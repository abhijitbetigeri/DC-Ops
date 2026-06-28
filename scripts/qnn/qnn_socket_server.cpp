/*
 * Persistent NPU inference server for the DC-Ops QNN app. Model-DRIVEN: input
 * dims, class/mask counts, mask resolution, detection scales and anchor count
 * are all DERIVED from the loaded .pte's tensor shapes (validated against an
 * optional meta.txt), instead of being hardcoded. The full YOLOv8-seg decode is
 * done here in C++ so the server returns ~KB of final detections instead of
 * ~7MB of raw tensors.
 *
 * Loads the .pte (QNN/HTP-delegated) ONCE via the ExecuTorch Module API and
 * keeps it resident, runs forward() per frame (~3ms NPU), then does
 * DFL -> dist2bbox, class sigmoid, area filter, per-class NMS, and mask->polygon
 * (convex hull) here in C++. Runs as the `shell` user in /data/local/tmp.
 *
 * Bundle layout (same dir as model.pte):
 *   model.pte  : the ExecuTorch QNN .pte
 *   meta.txt   : key=value lines (type=, reg_max=, input_size=)  [optional]
 *   labels.txt : one class label per line, class-index order     [optional]
 * meta.txt/labels.txt are hand-parsed (no json/3rd-party dep). Missing files
 * fall back to the historical defaults (yolov8_seg, reg_max=16, 640, 16 labels).
 *
 * Handshake (server -> client, sent once immediately on connect, all LE):
 *   int32 magic = 0x4D4F444C ("MODL")
 *   int32 input_w
 *   int32 input_h
 *   int32 num_classes
 *   repeat num_classes: int32 byte_len, then byte_len UTF-8 label bytes
 *
 * Per-frame request (client -> server): input_w*input_h*4 bytes, RGBA8888.
 * Per-frame response (server -> client): int32 payload_len (LE), then payload:
 *   int32 num_dets
 *   repeat: int32 class_id, float score, int32 n_pts,
 *           repeat n_pts: float x, float y   (normalized 0..1)
 * The server emits all detections >= CONF_FLOOR; the app filters by its slider.
 */
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;

#define PORT 8765
static const int32_t HANDSHAKE_MAGIC = 0x4D4F444C;  // "MODL"

// Decode tuning (model-independent).
static const float CONF_FLOOR = 0.20f;  // server floor; app slider filters further
static const float IOU_THRESH = 0.45f;
static const int MAX_DET = 50;
static const float MAX_BOX_AREA_FRAC = 0.30f;

// Historical defaults, used when meta.txt / labels.txt are absent.
static const char* DEFAULT_TYPE = "yolov8_seg";
static const int DEFAULT_REG_MAX = 16;
static const int DEFAULT_INPUT_SIZE = 640;
static const char* DEFAULT_LABELS[] = {
    "server rack", "compute tray", "NVLink switch tray", "network switch",
    "power shelf", "cable", "network port", "LED indicator", "label", "fan",
    "cooling manifold", "cable cartridge", "power connector", "drive bay",
    "management port", "DPU"};

// All values derived at startup from meta.txt + the .pte's tensor shapes.
struct ModelCfg {
  std::string type = DEFAULT_TYPE;
  int reg_max = DEFAULT_REG_MAX;
  int input_w = DEFAULT_INPUT_SIZE;
  int input_h = DEFAULT_INPUT_SIZE;
  int num_classes = 0;
  int num_mask = 0;
  int mask_res = 0;
  int anchors = 0;
  std::vector<std::pair<int, float>> scales;  // (grid, stride), grid descending
  std::vector<std::string> labels;
  // forward() output-tensor index mapping (positions are NOT assumed).
  std::vector<int> head_idx;  // detection heads, aligned with `scales`
  int mc_idx = -1;            // mask coefficients (1,M,A)
  int proto_idx = -1;         // mask prototypes  (1,M,Hm,Wm)
};

struct Det {
  float cx, cy, w, h, score;
  int cls;
  std::vector<float> coeff;
};
using Pt = std::pair<float, float>;

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static float iou(const Det& a, const Det& b) {
  float ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2, ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
  float bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2, bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
  float ix = std::max(0.f, std::min(ax2, bx2) - std::max(ax1, bx1));
  float iy = std::max(0.f, std::min(ay2, by2) - std::max(ay1, by1));
  float inter = ix * iy;
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-6f);
}

static std::vector<Pt> bbox_poly(const ModelCfg& cfg, float cx, float cy, float w, float h) {
  float sx = 1.0f / cfg.input_w, sy = 1.0f / cfg.input_h;
  float x1 = (cx - w / 2) * sx, y1 = (cy - h / 2) * sy;
  float x2 = (cx + w / 2) * sx, y2 = (cy + h / 2) * sy;
  return {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
}

// Andrew's monotone chain convex hull (CCW, no repeated endpoint).
static std::vector<Pt> convex_hull(std::vector<Pt> pts) {
  std::sort(pts.begin(), pts.end());
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
  int n = (int)pts.size();
  if (n < 3) return pts;
  auto cross = [](const Pt& o, const Pt& a, const Pt& b) {
    return (a.first - o.first) * (b.second - o.second) -
           (a.second - o.second) * (b.first - o.first);
  };
  std::vector<Pt> h(2 * n);
  int k = 0;
  for (int i = 0; i < n; i++) {
    while (k >= 2 && cross(h[k - 2], h[k - 1], pts[i]) <= 0) k--;
    h[k++] = pts[i];
  }
  for (int i = n - 2, t = k + 1; i >= 0; i--) {
    while (k >= t && cross(h[k - 2], h[k - 1], pts[i]) <= 0) k--;
    h[k++] = pts[i];
  }
  h.resize(k - 1);
  return h;
}

static std::vector<Pt> mask_poly(const ModelCfg& cfg, const float* coeff, const float* proto,
                                 float cx, float cy, float w, float h) {
  const int mres = cfg.mask_res, nmask = cfg.num_mask;
  const float cell = (float)cfg.input_w / mres;
  int bx1 = clampi((int)((cx - w / 2) / cell), 0, mres - 1);
  int by1 = clampi((int)((cy - h / 2) / cell), 0, mres - 1);
  int bx2 = clampi((int)((cx + w / 2) / cell), 0, mres - 1);
  int by2 = clampi((int)((cy + h / 2) / cell), 0, mres - 1);
  std::vector<Pt> pts;
  const int MM = mres * mres;
  for (int y = by1; y <= by2; y++) {
    for (int x = bx1; x <= bx2; x++) {
      float v = 0.f;
      for (int c = 0; c < nmask; c++) v += coeff[c] * proto[c * MM + y * mres + x];
      if (v > 0.f) pts.push_back({(float)x / mres, (float)y / mres});
    }
  }
  if (pts.size() < 3) return bbox_poly(cfg, cx, cy, w, h);
  auto hull = convex_hull(pts);
  return hull.size() >= 3 ? hull : bbox_poly(cfg, cx, cy, w, h);
}

// YOLOv8-seg decode (DFL -> dist2bbox, class sigmoid, area filter, per-class NMS).
static std::vector<Det> decode_yolov8_seg(const ModelCfg& cfg,
                                          const std::vector<const float*>& ps, const float* mc,
                                          float conf_floor) {
  const int reg = cfg.reg_max, ncls = cfg.num_classes, nmask = cfg.num_mask, A = cfg.anchors;
  const float maxArea = MAX_BOX_AREA_FRAC * cfg.input_w * cfg.input_h;
  std::vector<Det> dets;
  int anchorOffset = 0;
  for (size_t s = 0; s < cfg.scales.size(); s++) {
    int grid = cfg.scales[s].first;
    float stride = cfg.scales[s].second;
    int hw = grid * grid;
    const float* d = ps[s];
    for (int yy = 0; yy < grid; yy++) {
      for (int xx = 0; xx < grid; xx++) {
        int cell = yy * grid + xx;
        int bestC = 0;
        float bestLogit = -1e30f;
        for (int c = 0; c < ncls; c++) {
          float v = d[(4 * reg + c) * hw + cell];
          if (v > bestLogit) { bestLogit = v; bestC = c; }
        }
        float score = sigmoidf(bestLogit);
        if (score < conf_floor) continue;
        float dist[4];
        for (int kk = 0; kk < 4; kk++) {
          float mx = -1e30f;
          for (int b = 0; b < reg; b++) {
            float v = d[(kk * reg + b) * hw + cell];
            if (v > mx) mx = v;
          }
          float sum = 0.f, wsum = 0.f;
          for (int b = 0; b < reg; b++) {
            float ev = expf(d[(kk * reg + b) * hw + cell] - mx);
            sum += ev;
            wsum += b * ev;
          }
          dist[kk] = wsum / sum;
        }
        float ax = xx + 0.5f, ay = yy + 0.5f;
        float x1 = (ax - dist[0]) * stride, y1 = (ay - dist[1]) * stride;
        float x2 = (ax + dist[2]) * stride, y2 = (ay + dist[3]) * stride;
        if ((x2 - x1) * (y2 - y1) > maxArea) continue;
        int anchorIdx = anchorOffset + cell;
        Det det;
        det.cx = (x1 + x2) / 2;
        det.cy = (y1 + y2) / 2;
        det.w = x2 - x1;
        det.h = y2 - y1;
        det.cls = bestC;
        det.score = score;
        det.coeff.resize(nmask);
        for (int c = 0; c < nmask; c++) det.coeff[c] = mc[c * A + anchorIdx];
        dets.push_back(std::move(det));
      }
    }
    anchorOffset += hw;
  }
  std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
  std::vector<Det> kept;
  for (auto& cand : dets) {
    if ((int)kept.size() >= MAX_DET) break;
    bool sup = false;
    for (auto& k : kept) {
      if (k.cls == cand.cls && iou(k, cand) > IOU_THRESH) { sup = true; break; }
    }
    if (!sup) kept.push_back(std::move(cand));
  }
  return kept;
}

static bool read_full(int fd, char* b, long n) {
  long g = 0;
  while (g < n) { ssize_t r = read(fd, b + g, n - g); if (r <= 0) return false; g += r; }
  return true;
}
static bool write_full(int fd, const char* b, long n) {
  long s = 0;
  while (s < n) { ssize_t w = write(fd, b + s, n - s); if (w <= 0) return false; s += w; }
  return true;
}
static double ms(std::chrono::high_resolution_clock::time_point a,
                 std::chrono::high_resolution_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

template <typename T>
static void put(std::vector<char>& v, T x) {
  const char* p = reinterpret_cast<const char*>(&x);
  v.insert(v.end(), p, p + sizeof(T));
}

// ---- meta.txt / labels.txt parsing -----------------------------------------
static std::string dir_of(const std::string& path) {
  auto pos = path.find_last_of('/');
  return pos == std::string::npos ? std::string(".") : path.substr(0, pos);
}
static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}
static std::unordered_map<std::string, std::string> parse_meta(const std::string& path) {
  std::unordered_map<std::string, std::string> m;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    m[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
  }
  return m;
}
static std::vector<std::string> parse_labels(const std::string& path) {
  std::vector<std::string> v;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    std::string t = trim(line);
    if (!t.empty()) v.push_back(t);
  }
  return v;
}

int main(int argc, char** argv) {
  signal(SIGPIPE, SIG_IGN);
  executorch::runtime::runtime_init();
  const char* model_path = argc > 1 ? argv[1] : "model.pte";
  std::string dir = dir_of(model_path);

  ModelCfg cfg;

  // 1) meta.txt + labels.txt (optional; fall back to historical defaults).
  auto meta = parse_meta(dir + "/meta.txt");
  if (meta.count("type")) cfg.type = meta["type"];
  if (meta.count("reg_max")) cfg.reg_max = std::stoi(meta["reg_max"]);
  int meta_input = DEFAULT_INPUT_SIZE;
  if (meta.count("input_size")) meta_input = std::stoi(meta["input_size"]);
  cfg.labels = parse_labels(dir + "/labels.txt");
  if (cfg.labels.empty()) {
    for (const char* l : DEFAULT_LABELS) cfg.labels.emplace_back(l);
    fprintf(stderr, "meta: labels.txt missing/empty, using %zu default labels\n",
            cfg.labels.size());
  }
  fprintf(stderr, "meta: type=%s reg_max=%d input_size(meta)=%d labels=%zu\n",
          cfg.type.c_str(), cfg.reg_max, meta_input, cfg.labels.size());

  Module module(model_path);
  if (module.load() != Error::Ok) { fprintf(stderr, "FATAL: model load failed\n"); return 1; }
  if (module.load_forward() != Error::Ok) { fprintf(stderr, "FATAL: load_forward failed\n"); return 1; }

  // 2a) Input H/W from method_meta("forward") input (1,3,H,W); else meta.
  cfg.input_w = meta_input;
  cfg.input_h = meta_input;
  {
    auto mm = module.method_meta("forward");
    if (mm.ok() && mm->num_inputs() > 0) {
      auto ti = mm->input_tensor_meta(0);
      if (ti.ok() && ti->sizes().size() == 4) {
        cfg.input_h = (int)ti->sizes()[2];
        cfg.input_w = (int)ti->sizes()[3];
      }
    }
  }

  const long HW_IN = (long)cfg.input_w * cfg.input_h;
  const long IN_FLOATS = 3L * HW_IN;
  const long IN_BYTES = HW_IN * 4;  // RGBA8888
  std::vector<char> rawin(IN_BYTES), payload;
  std::vector<float> infloat(IN_FLOATS);

  // Warmup forward -- also our source of truth for output tensor shapes.
  size_t n_outputs = 0;
  {
    auto t = from_blob(infloat.data(), {1, 3, cfg.input_h, cfg.input_w}, ScalarType::Float);
    auto r = module.forward(std::vector<EValue>{*t});
    if (!r.ok()) { fprintf(stderr, "FATAL: warmup forward failed\n"); return 1; }
    n_outputs = r->size();
    fprintf(stderr, "warmup forward: ok, outputs=%zu\n", n_outputs);

    // 2b) Classify outputs by shape (positions are NOT assumed):
    //   mc    : the unique 3D output (1, M, A)
    //   proto : the unique 4D output with dim1 == M
    //   heads : the remaining 4D outputs (1, C, g, g)
    auto& outs = *r;
    std::vector<int> dim4;
    int n3d = 0;
    for (size_t i = 0; i < outs.size(); i++) {
      auto tt = outs[i].toTensor();
      if (tt.dim() == 3) {
        cfg.mc_idx = (int)i;
        cfg.num_mask = (int)tt.sizes()[1];
        cfg.anchors = (int)tt.sizes()[2];
        n3d++;
      } else if (tt.dim() == 4) {
        dim4.push_back((int)i);
      }
    }
    if (n3d != 1) {
      fprintf(stderr, "FATAL: expected exactly 1 3D (mask-coeff) output, found %d\n", n3d);
      return 1;
    }
    for (int i : dim4) {
      auto tt = outs[i].toTensor();
      if ((int)tt.sizes()[1] == cfg.num_mask && cfg.proto_idx < 0) {
        cfg.proto_idx = i;
        cfg.mask_res = (int)tt.sizes()[2];  // Hm (== Wm)
      } else {
        cfg.head_idx.push_back(i);  // detection head
      }
    }
    if (cfg.proto_idx < 0) { fprintf(stderr, "FATAL: no proto output (4D dim1==num_mask)\n"); return 1; }
    if (cfg.head_idx.size() != 3) {
      fprintf(stderr, "FATAL: expected 3 detection heads, found %zu\n", cfg.head_idx.size());
      return 1;
    }

    // Per-head (grid, stride); sort by grid descending and align head_idx with it.
    std::vector<std::pair<int, int>> tmp;  // (grid, output_idx)
    int head_channels = -1;
    for (int i : cfg.head_idx) {
      auto tt = outs[i].toTensor();
      int g = (int)tt.sizes()[2];
      if (head_channels < 0) head_channels = (int)tt.sizes()[1];
      tmp.emplace_back(g, i);
    }
    std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a.first > b.first; });
    cfg.scales.clear();
    cfg.head_idx.clear();
    int anchor_sum = 0;
    for (auto& gp : tmp) {
      int g = gp.first;
      cfg.scales.emplace_back(g, (float)(cfg.input_h / g));
      cfg.head_idx.push_back(gp.second);
      anchor_sum += g * g;
    }
    cfg.num_classes = head_channels - 4 * cfg.reg_max;

    // Cross-checks (loud on inconsistency).
    if (anchor_sum != cfg.anchors) {
      fprintf(stderr, "FATAL: anchor mismatch: sum(g^2)=%d but mask-coeff A=%d\n",
              anchor_sum, cfg.anchors);
      return 1;
    }
    if (cfg.num_classes <= 0) {
      fprintf(stderr, "FATAL: derived num_classes=%d (head_channels=%d reg_max=%d)\n",
              cfg.num_classes, head_channels, cfg.reg_max);
      return 1;
    }
    if ((int)cfg.labels.size() != cfg.num_classes) {
      fprintf(stderr, "WARNING: labels.size()=%zu != derived num_classes=%d (continuing)\n",
              cfg.labels.size(), cfg.num_classes);
    }
  }

  // 6) Startup line (verification confirms derivation against this).
  {
    std::string scales;
    for (size_t i = 0; i < cfg.scales.size(); i++) {
      if (i) scales += ",";
      scales += std::to_string(cfg.scales[i].first) + "/" +
                std::to_string((int)cfg.scales[i].second);
    }
    fprintf(stderr,
            "model: type=%s input=%dx%d classes=%d masks=%d mask_res=%d scales=%s anchors=%d\n",
            cfg.type.c_str(), cfg.input_w, cfg.input_h, cfg.num_classes, cfg.num_mask,
            cfg.mask_res, scales.c_str(), cfg.anchors);
    fflush(stderr);
  }

  int srv = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(PORT);
  if (bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
  listen(srv, 4);
  fprintf(stderr, "persistent NPU server (C++ decode) listening on 127.0.0.1:%d (model=%s)\n",
          PORT, model_path);
  fflush(stderr);

  // Precompute the handshake header (constant for the model's lifetime).
  std::vector<char> header;
  put<int32_t>(header, HANDSHAKE_MAGIC);
  put<int32_t>(header, (int32_t)cfg.input_w);
  put<int32_t>(header, (int32_t)cfg.input_h);
  put<int32_t>(header, (int32_t)cfg.num_classes);
  for (int c = 0; c < cfg.num_classes; c++) {
    const std::string& lbl = c < (int)cfg.labels.size() ? cfg.labels[c]
                                                        : ("class_" + std::to_string(c));
    put<int32_t>(header, (int32_t)lbl.size());
    header.insert(header.end(), lbl.begin(), lbl.end());
  }

  bool unknown_type_logged = false;
  long frame = 0;
  for (;;) {
    int cli = accept(srv, nullptr, nullptr);
    if (cli < 0) continue;
    int nd = 1;
    setsockopt(cli, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
    fprintf(stderr, "client connected\n");
    fflush(stderr);

    // 4) Send handshake header before reading any frame.
    if (!write_full(cli, header.data(), (long)header.size())) {
      close(cli);
      fprintf(stderr, "client disconnected (handshake)\n");
      fflush(stderr);
      continue;
    }

    int32_t ctrl = -1;
    while (read_full(cli, (char*)&ctrl, 4) && read_full(cli, rawin.data(), IN_BYTES)) {
      auto t0 = std::chrono::high_resolution_clock::now();
      // ctrl: server floor in 1/1000ths (0..1000), or -1 = use default CONF_FLOOR.
      float conf_floor = (ctrl < 0) ? CONF_FLOOR : (ctrl / 1000.0f);
      const unsigned char* px = reinterpret_cast<const unsigned char*>(rawin.data());
      float* f = infloat.data();
      for (long i = 0; i < HW_IN; i++) {
        f[i]             = px[4 * i + 0] * (1.0f / 255.0f);
        f[HW_IN + i]     = px[4 * i + 1] * (1.0f / 255.0f);
        f[2 * HW_IN + i] = px[4 * i + 2] * (1.0f / 255.0f);
      }
      auto tconv = std::chrono::high_resolution_clock::now();
      auto input = from_blob(infloat.data(), {1, 3, cfg.input_h, cfg.input_w}, ScalarType::Float);
      auto res = module.forward(std::vector<EValue>{*input});
      auto t1 = std::chrono::high_resolution_clock::now();
      if (!res.ok() || res->size() < n_outputs) { fprintf(stderr, "forward error\n"); break; }

      auto& outs = *res;
      std::vector<Det> kept;
      const float* proto = nullptr;
      if (cfg.type == "yolov8_seg") {
        std::vector<const float*> ps(cfg.head_idx.size());
        for (size_t s = 0; s < cfg.head_idx.size(); s++)
          ps[s] = outs[cfg.head_idx[s]].toTensor().const_data_ptr<float>();
        const float* mc = outs[cfg.mc_idx].toTensor().const_data_ptr<float>();
        proto = outs[cfg.proto_idx].toTensor().const_data_ptr<float>();
        kept = decode_yolov8_seg(cfg, ps, mc, conf_floor);
      } else if (!unknown_type_logged) {
        fprintf(stderr, "WARNING: unknown model type '%s' -> serving empty detections\n",
                cfg.type.c_str());
        unknown_type_logged = true;
      }

      payload.clear();
      put<int32_t>(payload, (int32_t)kept.size());
      for (auto& d : kept) {
        auto poly = mask_poly(cfg, d.coeff.data(), proto, d.cx, d.cy, d.w, d.h);
        put<int32_t>(payload, (int32_t)d.cls);
        put<float>(payload, d.score);
        put<int32_t>(payload, (int32_t)poly.size());
        for (auto& p : poly) { put<float>(payload, p.first); put<float>(payload, p.second); }
      }
      auto t2 = std::chrono::high_resolution_clock::now();

      int32_t len = (int32_t)payload.size();
      if (!write_full(cli, (char*)&len, 4)) break;
      if (!write_full(cli, payload.data(), len)) break;
      auto t3 = std::chrono::high_resolution_clock::now();

      if ((frame++ % 30) == 0) {
        fprintf(stderr,
                "frame %ld: conv=%.2f forward=%.2f decode=%.2f send=%.2f total=%.2fms dets=%zu bytes=%d\n",
                frame, ms(t0, tconv), ms(tconv, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3),
                kept.size(), len + 4);
        fflush(stderr);
      }
    }
    close(cli);
    fprintf(stderr, "client disconnected\n");
    fflush(stderr);
  }
}
