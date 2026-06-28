import socket, time, numpy as np
from PIL import Image

HOST, PORT = "127.0.0.1", 18765  # adb-forwarded to device 8765
IN_BYTES = 1*3*640*640*4
OUT_SIZES = [1*80*80*80*4, 1*80*40*40*4, 1*80*20*20*4, 1*32*8400*4, 1*32*160*160*4]
OUT_TOTAL = sum(OUT_SIZES)
CLASSES = ["server rack","compute tray","NVLink switch tray","network switch","power shelf","cable",
           "network port","LED indicator","label","fan","cooling manifold","cable cartridge",
           "power connector","drive bay","management port","DPU"]

img = "/mnt/c/Users/Rashi/AndroidStudioProjects/DC-Ops/android-app-qnn/app/src/main/assets/test_rack.jpg"
im = Image.open(img).convert("RGB").resize((640,640))
x = (np.asarray(im, dtype=np.float32)/255.).transpose(2,0,1)[None].copy()
payload = x.tobytes()
assert len(payload) == IN_BYTES, len(payload)

def recv_full(s, n):
    buf = bytearray()
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c: raise IOError("closed")
        buf += c
    return bytes(buf)

s = socket.create_connection((HOST, PORT), timeout=30)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
# time a few frames
for it in range(3):
    t0 = time.time()
    s.sendall(payload)
    out = recv_full(s, OUT_TOTAL)
    dt = (time.time()-t0)*1000
    print(f"frame {it}: round-trip {dt:.1f} ms")
s.close()

# decode last result
arrs, off = [], 0
for n in OUT_SIZES:
    arrs.append(np.frombuffer(out[off:off+n], dtype=np.float32)); off += n
ps = [arrs[0].reshape(1,80,80,80)[0], arrs[1].reshape(1,80,40,40)[0], arrs[2].reshape(1,80,20,20)[0]]
def sig(z): return 1/(1+np.exp(-z))
SCAL=[(80,8),(40,16),(20,32)]; dets=[]
for si,(g,st) in enumerate(SCAL):
    d=ps[si]
    for yy in range(g):
        for xx in range(g):
            logits=d[64:80,yy,xx]; bc=int(logits.argmax()); sc=float(sig(logits[bc]))
            if sc<0.35: continue
            dist=[]
            for k in range(4):
                z=d[k*16:(k+1)*16,yy,xx]; e=np.exp(z-z.max()); p=e/e.sum(); dist.append(float((np.arange(16)*p).sum()))
            ax,ay=xx+0.5,yy+0.5
            x1=(ax-dist[0])*st; y1=(ay-dist[1])*st; x2=(ax+dist[2])*st; y2=(ay+dist[3])*st
            dets.append((sc,bc,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1))
dets.sort(reverse=True)
def iou(A,B):
    ax1,ay1,ax2,ay2=A[2]-A[4]/2,A[3]-A[5]/2,A[2]+A[4]/2,A[3]+A[5]/2
    bx1,by1,bx2,by2=B[2]-B[4]/2,B[3]-B[5]/2,B[2]+B[4]/2,B[3]+B[5]/2
    ix=max(0,min(ax2,bx2)-max(ax1,bx1)); iy=max(0,min(ay2,by2)-max(ay1,by1)); i=ix*iy
    return i/(A[4]*A[5]+B[4]*B[5]-i+1e-6)
keep=[]
for dd in dets:
    if all(not(dd[1]==k[1] and iou(dd,k)>0.45) for k in keep): keep.append(dd)
print(f"raw>0.35: {len(dets)}  | after NMS: {len(keep)}")
for sc,bc,cx,cy,w,h in keep[:15]:
    print(f"  {CLASSES[bc]:18s} {sc:.2f}  box(px)=({cx:.0f},{cy:.0f},{w:.0f},{h:.0f})")
