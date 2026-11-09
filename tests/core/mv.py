import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'd')
sys.argv = ['x', 'none']; sys.path.insert(0, HERE)
import render_seq2 as r
FOV = 39.3077
def proj(P, f):
    o, t = r.cam(f); o=np.array(o); t=np.array(t); up=np.array([0,1.,0])
    d=(t-o)/np.linalg.norm(t-o); left=np.cross(up,d); left/=np.linalg.norm(left); nu=np.cross(d,left)
    v=P-o; zc=v@d; xc=v@left; yc=v@nu
    th=np.tan(np.radians(FOV)/2)
    return 128-128*xc/zc/th, 128-128*yc/zc/th, zc
def flow(f, g, ballsid):
    """pixel displacement (top-down px) from frame f to frame g, plus depth at f"""
    pos=np.load(f'{D}/pos_{f}.npy').reshape(-1,3); sid=np.load(f'{D}/sid_{f}.npy').ravel()
    H=W=r.RES
    P=pos.copy(); mb = np.abs(sid-ballsid)<0.5
    P[mb]+=np.array(r.ball(g))-np.array(r.ball(f))
    x0,y0,z=proj(pos,f); x1,y1,_=proj(P,g)
    return np.stack([x1-x0,y1-y0],-1).reshape(H,W,2).astype(np.float32), z.reshape(H,W).astype(np.float32), mb.reshape(H,W)
if __name__=='__main__':
    sid=np.load(f'{D}/sid_4.npy'); x,y,_=proj(np.array(r.ball(4)),4); bs=sid[int(y),int(x)]; print('ball sid', bs)
    for f in range(r.NF):
        g=f+1 if f+1<r.NF else f-1
        fl,z,mb=flow(f, min(f+1,r.NF-1) if f+1<r.NF else f, bs) if f+1<r.NF else (None,None,None)
        if f+1<r.NF:
            np.save(f'{D}/mvf_{f}.npy', fl)
        # backward vector (f -> f-1)
        if f>0:
            fb,_,_=flow(f,f-1,bs); np.save(f'{D}/mvb_{f}.npy', fb)
        _,z,mb=flow(f,f,bs); np.save(f'{D}/depth_{f}.npy', z); np.save(f'{D}/ballmask_{f}.npy', mb)
    m=np.load(f'{D}/mvf_4.npy'); print('mv stats x', np.percentile(m[...,0],[1,50,99]), 'y', np.percentile(m[...,1],[1,50,99]))
    print('ball px', np.load(f'{D}/ballmask_4.npy').sum())
