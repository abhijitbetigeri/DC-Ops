import numpy as np
CLASSES=["server rack","compute tray","NVLink switch tray","network switch","power shelf","cable",
         "network port","LED indicator","label","fan","cooling manifold","cable cartridge",
         "power connector","drive bay","management port","DPU"]
D="/tmp/qnn_out"
ps=[np.fromfile(f"{D}/output_0_{i}.raw",dtype=np.float32).reshape(1,80,g,g)[0] for i,g in [(0,80),(1,40),(2,20)]]
mc=np.fromfile(f"{D}/output_0_3.raw",dtype=np.float32).reshape(1,32,8400)[0]
proto=np.fromfile(f"{D}/output_0_4.raw",dtype=np.float32).reshape(1,32,160,160)[0]
def sig(x): return 1/(1+np.exp(-x))
SCAL=[(80,8),(40,16),(20,32)]; dets=[]; off=0
for s,(g,st) in enumerate(SCAL):
    d=ps[s]
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
    off+=g*g
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
