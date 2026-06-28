import os, io, json, socket, urllib.request, torch, numpy as np
socket.setdefaulttimeout(30)  # so a single stalled HF download can't hang the whole run forever
# WSL here has broken IPv6 -> HF connections stall ~30s in SYN-SENT on AAAA records.
# Force IPv4 so downloads are fast instead of crawling through timeouts.
_orig_gai = socket.getaddrinfo
def _ipv4_only(host, *a, **kw):
    return [r for r in _orig_gai(host, *a, **kw) if r[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_only
from PIL import Image
from executorch.backends.qualcomm.export_utils import build_executorch_binary, QnnConfig
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

ER = os.environ["EXECUTORCH_ROOT"]
HF = "https://huggingface.co/datasets/abhijitbetigeri/dc-ops-dataset"
N = int(os.environ.get("INPUT_SIZE", "640"))  # square network input NxN

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

# Base model: torchvision RetinaNet (ResNet50-FPN), saved as a FULL model object (not a state_dict).
# MODEL_PT env overrides the default path. Trained on CUDA -> MUST map_location="cpu".
MODEL_PT = os.environ.get("MODEL_PT", "/home/rashi/dcops_qnn/retinanet_dc_ops.pt")
print(f"loading RetinaNet base model {MODEL_PT}", flush=True)
m = torch.load(MODEL_PT, weights_only=False, map_location="cpu").eval()
ch = m.head.classification_head
rh = m.head.regression_head
NUM_CLASSES = ch.num_classes
NUM_ANCHORS = ch.num_anchors
print(f"num_classes={NUM_CLASSES} num_anchors={NUM_ANCHORS}", flush=True)

# RetinaNet's own GeneralizedRCNNTransform applies ImageNet normalization (mean/std). The NPU
# server feeds px/255 only (identical to the YOLO path) with NO mean/std. So we BAKE the ImageNet
# normalization into the wrapper's forward: the .pte accepts [0,1] RGB NCHW and normalizes
# internally, keeping the C++ input feed byte-for-byte identical to the YOLO bundle.
#
# Wrapper returns ONLY the raw classification + box-regression conv head tensors per FPN level,
# bypassing GeneralizedRCNNTransform, AnchorGenerator and postprocess entirely. Those un-lowerable
# ops (anchor arange/meshgrid, box decode, batched_nms, topk) feed nothing in this graph, so DCE
# drops them. The NPU runs every conv; the C++ server does anchor-gen + box-decode + NMS.
#
# OUTPUT CONTRACT (deterministic order, 10 tensors):
#   idx 0..4 = cls_logits per level, grid DESC: (1,153,80,80),(1,153,40,40),(1,153,20,20),
#                                                (1,153,10,10),(1,153,5,5)   [153 = num_anchors*num_classes = 9*17]
#   idx 5..9 = bbox_reg   per level, grid DESC: (1,36,80,80), (1,36,40,40), (1,36,20,20),
#                                                (1,36,10,10), (1,36,5,5)    [36  = num_anchors*4 = 9*4]
# Head channel layout (torchvision): channel c -> anchor a = c // K, attr k = c % K
#   (cls K=17: channel = anchor*17 + class ; reg K=4: channel = anchor*4 + {dx,dy,dw,dh}).
# The C++ distinguishes cls (C=153) vs reg (C=36) by channel count and level by grid size, so it
# is robust to ExecuTorch reordering, but this order is the documented intent.
class RawWrap(torch.nn.Module):
    def __init__(s, mm):
        super().__init__()
        s.backbone = mm.backbone
        s.cls_conv = mm.head.classification_head.conv
        s.cls_logits = mm.head.classification_head.cls_logits
        s.reg_conv = mm.head.regression_head.conv
        s.reg_bbox = mm.head.regression_head.bbox_reg
        # ImageNet norm baked in (shape (1,3,1,1) so it broadcasts over NCHW).
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(s, x):
        xn = (x - s.mean) / s.std                     # x is [0,1] RGB NCHW (server feeds px/255)
        feats = s.backbone(xn)                        # OrderedDict: '0','1','2','p6','p7' (grid 80..5)
        levels = [feats[k] for k in ("0", "1", "2", "p6", "p7")]
        cls = [s.cls_logits(s.cls_conv(f)) for f in levels]   # (1,153,g,g) per level
        reg = [s.reg_bbox(s.reg_conv(f)) for f in levels]     # (1,36,g,g)  per level
        return tuple(cls) + tuple(reg)                # 10 raw conv tensors, cls[0..4] then reg[0..4]

w = RawWrap(m).eval()

# sanity: print raw output shapes + confirm no decode/anchor ops survive in the exported graph
ex = (torch.randn(1, 3, N, N),)
with torch.no_grad():
    outs = w(*ex)
print("raw outputs:", [tuple(t.shape) for t in outs])
assert len(outs) == 10, f"expected 10 raw head tensors, got {len(outs)}"
gm = torch.export.export(w, ex).graph_module
bad = [str(n.target) for n in gm.graph.nodes
       if any(k in str(n.target) for k in ("unbind", "full", "arange", "meshgrid", "nms", "topk", "nonzero"))]
print("decode/anchor ops still present (want []):", bad)

# calibration — use a large slice of the real HF dc-ops dataset for better PTQ.
# Cache each image to disk so a re-run reuses them instead of re-downloading from flaky HF.
# Calib tensors are [0,1] RGB (np/255), SAME as the YOLO export — the wrapper normalizes
# internally, so do NOT pre-apply ImageNet norm here or it double-normalizes.
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
        im = Image.open(io.BytesIO(b)).convert("RGB").resize((N, N))
        calib.append((torch.from_numpy(np.asarray(im).astype("float32") / 255.).permute(2, 0, 1).unsqueeze(0),))
    except Exception as e:
        print("skip", p, repr(e), flush=True)
    if (i + 1) % 10 == 0:
        print(f"  calib {len(calib)}/{i+1} (of {TARGET})", flush=True)
print("calib:", len(calib), flush=True)
if len(calib) < 8:
    raise SystemExit(f"too few calibration images ({len(calib)}) — HF download likely failing")

cfg = QnnConfig(soc_model="SM8750", build_folder=f"{ER}/build-android", backend="htp", compile_only=True)
out = os.environ.get("OUT_PTE_BASE", f"{ER}/dcops_qnn/retinanet_dc_ops_qnn")
os.makedirs(os.path.dirname(out), exist_ok=True)
# w8a16 (16-bit activations) — same recommended HTP precision as YOLO; 8a8w collapsed YOLO scores
# to 0, and RetinaNet's sigmoid class logits are similarly sensitive, so keep use_16a8w.
build_executorch_binary(model=w, qnn_config=cfg, file_name=out, dataset=calib, quant_dtype=QuantDtype.use_16a8w)
print("WROTE", out + ".pte", os.path.getsize(out + ".pte"), "bytes")
