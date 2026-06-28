import os, io, json, socket, urllib.request, torch, numpy as np
socket.setdefaulttimeout(30)  # so a single stalled HF download can't hang the whole run forever
# WSL here has broken IPv6 -> HF connections stall ~30s in SYN-SENT on AAAA records.
# Force IPv4 so downloads are fast instead of crawling through timeouts.
_orig_gai = socket.getaddrinfo
def _ipv4_only(host, *a, **kw):
    return [r for r in _orig_gai(host, *a, **kw) if r[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_only
from PIL import Image
from ultralytics import YOLO
from executorch.backends.qualcomm.export_utils import build_executorch_binary, QnnConfig
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

ER = os.environ["EXECUTORCH_ROOT"]
HF = "https://huggingface.co/datasets/abhijitbetigeri/dc-ops-dataset"

def fetch(url, retries=5, backoff=3):
    """Download bytes with retries+backoff — HF is flaky from WSL (SSL record-layer / stalls)."""
    last = None
    for a in range(retries):
        try:
            return urllib.request.urlopen(url).read()
        except Exception as e:
            last = e
            print(f"  retry {a+1}/{retries} for {url.split('/')[-1]}: {repr(e)}", flush=True)
            import time; time.sleep(backoff * (a + 1))
    raise last

# Base model: MODEL_PT env overrides; else reuse the local cache; else download default.
CACHED_PT = os.path.expanduser("~/dcops_qnn/dc_ops_yolov8n_seg.pt")
model_pt = os.environ.get("MODEL_PT", "")
if model_pt and os.path.exists(model_pt):
    print(f"using MODEL_PT {model_pt}", flush=True)
elif os.path.exists(CACHED_PT) and os.path.getsize(CACHED_PT) > 1_000_000:
    print(f"using cached base model {CACHED_PT}", flush=True)
    model_pt = CACHED_PT
else:
    model_pt = "/tmp/dcops.pt"
    open(model_pt, "wb").write(fetch(f"{HF}/resolve/main/models/dc_ops_yolov8n_seg.pt"))
_yolo = YOLO(model_pt)
m = _yolo.model.eval()
# Emit the model's class names so the bundle's labels.txt/model.json can match exactly.
try:
    _names = _yolo.names if hasattr(_yolo, "names") else m.names
    _names = [_names[i] for i in range(len(_names))]
    print("CLASS_NAMES:", json.dumps(_names), flush=True)
except Exception as e:
    print("class-name introspection failed:", repr(e), flush=True)

# Return ONLY the raw pre-decode head outputs. The box-decode (dist2bbox/make_anchors ->
# unbind/full/slice that QNN rejects) feeds only the decoded det tensor, which we drop;
# DCE then removes it. NPU runs all conv; Kotlin does the lightweight decode + NMS.
class RawWrap(torch.nn.Module):
    def __init__(s, mm): super().__init__(); s.m = mm
    def forward(s, x):
        o = s.m(x)
        ps, mc, pr = o[1][0], o[1][1], o[1][2]   # 3x(1,80,H,W), (1,32,8400), (1,32,160,160)
        return ps[0], ps[1], ps[2], mc, pr
w = RawWrap(m).eval()

# sanity: outputs + confirm graph has no unbind/full after dropping decode
ex = (torch.randn(1, 3, 640, 640),)
with torch.no_grad():
    outs = w(*ex)
print("raw outputs:", [tuple(t.shape) for t in outs])
gm = torch.export.export(w, ex).graph_module
bad = [n.target for n in gm.graph.nodes if any(k in str(n.target) for k in ("unbind", "full", "arange"))]
print("decode ops still present (want []):", bad)

# calibration — use a large slice of the real HF dataset for better PTQ.
# Cache each image to disk so a re-run reuses them instead of re-downloading from flaky HF.
CALIB_DIR = os.path.expanduser("~/et_setup/calib_imgs")
os.makedirs(CALIB_DIR, exist_ok=True)
api = json.loads(fetch("https://huggingface.co/api/datasets/abhijitbetigeri/dc-ops-dataset/tree/main/images?recursive=false&limit=1000"))
imgs_all = [f["path"] for f in api if f["path"].lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
print(f"calibration images available: {len(imgs_all)}", flush=True)
calib = []
TARGET = 200
for i, p in enumerate(imgs_all[:TARGET]):
    local = os.path.join(CALIB_DIR, os.path.basename(p))
    try:
        if os.path.exists(local) and os.path.getsize(local) > 0:
            b = open(local, "rb").read()
        else:
            b = fetch(f"{HF}/resolve/main/{p}")
            open(local, "wb").write(b)
        im = Image.open(io.BytesIO(b)).convert("RGB").resize((640, 640))
        calib.append((torch.from_numpy(np.asarray(im).astype("float32") / 255.).permute(2, 0, 1).unsqueeze(0),))
    except Exception as e:
        print("skip", p, repr(e), flush=True)
    if (i + 1) % 10 == 0:
        print(f"  calib {len(calib)}/{i+1} (of {TARGET})", flush=True)
print("calib:", len(calib), flush=True)
if len(calib) < 8:
    raise SystemExit(f"too few calibration images ({len(calib)}) — HF download likely failing")

cfg = QnnConfig(soc_model="SM8750", build_folder=f"{ER}/build-android", backend="htp", compile_only=True)
out = os.environ.get("OUT_PTE_BASE", f"{ER}/dcops_qnn/dc_ops_yolov8n_seg_backbone_qnn")
os.makedirs(os.path.dirname(out), exist_ok=True)
# w8a16 (16-bit activations) — YOLO's recommended HTP precision; 8a8w collapsed scores -> 0 detections
build_executorch_binary(model=w, qnn_config=cfg, file_name=out, dataset=calib, quant_dtype=QuantDtype.use_16a8w)
print("WROTE", out + ".pte", os.path.getsize(out + ".pte"), "bytes")
