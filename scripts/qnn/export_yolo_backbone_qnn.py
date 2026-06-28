import os, io, json, urllib.request, torch, numpy as np
from PIL import Image
from ultralytics import YOLO
from executorch.backends.qualcomm.export_utils import build_executorch_binary, QnnConfig
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

ER = os.environ["EXECUTORCH_ROOT"]
HF = "https://huggingface.co/datasets/abhijitbetigeri/dc-ops-dataset"
open("/tmp/dcops.pt", "wb").write(urllib.request.urlopen(f"{HF}/resolve/main/models/dc_ops_yolov8n_seg.pt").read())
m = YOLO("/tmp/dcops.pt").model.eval()

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

# calibration — use a large slice of the real HF dataset for better PTQ
api = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/abhijitbetigeri/dc-ops-dataset/tree/main/images?recursive=false&limit=1000"))
imgs_all = [f["path"] for f in api if f["path"].lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
print(f"calibration images available: {len(imgs_all)}")
calib = []
for p in imgs_all[:200]:
    try:
        b = urllib.request.urlopen(f"{HF}/resolve/main/{p}").read()
        im = Image.open(io.BytesIO(b)).convert("RGB").resize((640, 640))
        calib.append((torch.from_numpy(np.asarray(im).astype("float32") / 255.).permute(2, 0, 1).unsqueeze(0),))
    except Exception as e:
        print("skip", p, e)
print("calib:", len(calib))

cfg = QnnConfig(soc_model="SM8750", build_folder=f"{ER}/build-android", backend="htp", compile_only=True)
out = f"{ER}/dcops_qnn/dc_ops_yolov8n_seg_backbone_qnn"
os.makedirs(os.path.dirname(out), exist_ok=True)
# w8a16 (16-bit activations) — YOLO's recommended HTP precision; 8a8w collapsed scores -> 0 detections
build_executorch_binary(model=w, qnn_config=cfg, file_name=out, dataset=calib, quant_dtype=QuantDtype.use_16a8w)
print("WROTE", out + ".pte", os.path.getsize(out + ".pte"), "bytes")
