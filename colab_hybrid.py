"""Self-contained Colab runner for the PatchMatch HYBRID validation.

Runs entirely on Colab (T4 for the face detector, CPU for PatchMatch). Needs NO
local GPU. Fetches orig+mask for the 4 test scenes from the asset repo, then per
scene produces:
  - pure PatchMatch (feather composite, no face-router)        -> shows raw PM
  - PatchMatch + global_mask(faces) + seamlessClone (the recipe) -> the fixes
plus the ring-band texture-gate decision (ring_ed vs PM_TEXTURE_THRESH) and a
face-hallucination count. Uploads montages to tmpfiles and reports URLs via ntfy
(topic below). PatchMatch runs in a numpy-only subprocess (no cv2) to avoid the
apt-opencv vs pip-cv2 ABI clash — same isolation trick as the local pm_worker.

Cell to run it (after Runtime->GPU):
    !wget -qO /content/colab_hybrid.py https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/colab_hybrid.py && python /content/colab_hybrid.py
"""
import os, sys, json, time, subprocess, tempfile, traceback

TOPIC = "manga-infill-colab-7x2"
BASE = "https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/"
SCENES = [17, 49, 42, 18]
PS, CTX, WORK, TEX_THRESH = 15, 256, 768, 0.035

PM_WORKER_SRC = r'''
import sys, os
sys.path.insert(0, "/content/PyPatchMatch")
import numpy as np, patch_match
def main():
    inp, outp, ps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    d = np.load(inp); n = int(d["n"]); res = {}
    for i in range(n):
        crop = np.ascontiguousarray(d[f"crop_{i}"]); cm = np.ascontiguousarray(d[f"cm_{i}"])
        gm = d[f"gm_{i}"] if f"gm_{i}" in d else None
        if gm is not None and gm.any():
            res[f"res_{i}"] = patch_match.inpaint(crop, cm, global_mask=np.ascontiguousarray(gm), patch_size=ps)
        else:
            res[f"res_{i}"] = patch_match.inpaint(crop, cm, patch_size=ps)
    np.savez(outp, **res); print(n)
main()
'''


def notify(t):
    import requests
    try:
        requests.post("https://ntfy.sh/" + TOPIC, data=t.encode()[:3500])
    except Exception as e:
        print("notify failed", e, flush=True)


def upload(path):
    import requests
    r = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": open(path, "rb")})
    u = json.loads(r.text)["data"]["url"]
    return u.replace("tmpfiles.org/", "tmpfiles.org/dl/")


def main():
    # ---- setup ----
    subprocess.run("apt-get -qq install -y libopencv-dev", shell=True)
    subprocess.run(f"{sys.executable} -m pip -q install ultralytics requests", shell=True)
    if not os.path.exists("/content/PyPatchMatch"):
        subprocess.run("git clone -q https://github.com/vacancy/PyPatchMatch /content/PyPatchMatch", shell=True)
    subprocess.run("sed -i 's/pkg-config --cflags opencv)/pkg-config --cflags opencv4)/g; "
                   "s/pkg-config --cflags --libs opencv)/pkg-config --cflags --libs opencv4)/g' "
                   "/content/PyPatchMatch/Makefile", shell=True)
    subprocess.run("make -C /content/PyPatchMatch", shell=True, check=True)
    open("/content/pm_worker.py", "w").write(PM_WORKER_SRC)
    notify("SETUP done (patchmatch built)")

    import cv2
    import numpy as np
    import urllib.request
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    face = YOLO(hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt"))

    def dl(name):
        return cv2.imdecode(np.frombuffer(urllib.request.urlopen(BASE + name, timeout=60).read(), np.uint8),
                            cv2.IMREAD_COLOR if "mask" not in name else cv2.IMREAD_GRAYSCALE)

    def face_boxes(im):
        r = face.predict(im, verbose=False, conf=0.30)[0]
        return [[int(v) for v in b] for b in (r.boxes.xyxy.tolist() if r.boxes is not None else [])]

    def new_faces(base, cand, mask):
        fa, fb = face_boxes(base), face_boxes(cand)
        def ov(bx, lst, fr=0.2):
            ar = (bx[2]-bx[0])*(bx[3]-bx[1])
            for c in lst:
                ix = max(0, min(bx[2],c[2])-max(bx[0],c[0])); iy = max(0, min(bx[3],c[3])-max(bx[1],c[1]))
                if ix*iy > fr*max(1,ar): return True
            return False
        m = mask > 127; cnt = 0
        for x in fb:
            if ov(x, fa): continue
            sub = m[max(0,x[1]):x[3], max(0,x[0]):x[2]]
            if sub.size and sub.mean() > 0.15: cnt += 1
        return cnt

    def run_pm(crop_rgb, cm, gm):
        with tempfile.TemporaryDirectory() as td:
            ip, rp = f"{td}/in.npz", f"{td}/out.npz"
            d = {"n": 1, "crop_0": np.ascontiguousarray(crop_rgb), "cm_0": cm}
            if gm is not None:
                d["gm_0"] = gm
            np.savez(ip, **d)
            r = subprocess.run([sys.executable, "/content/pm_worker.py", ip, rp, str(PS)],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(rp):
                raise RuntimeError("pm_worker: " + r.stderr[-400:])
            return np.load(rp)["res_0"]

    def face_mask_in_crop(fb, x0, y0, x1, y1):
        fm = np.zeros((y1-y0, x1-x0), np.uint8)
        for (fx0, fy0, fx1, fy1) in fb:
            ix0, iy0, ix1, iy1 = max(fx0,x0), max(fy0,y0), min(fx1,x1), min(fy1,y1)
            if ix1 > ix0 and iy1 > iy0:
                fm[iy0-y0:iy1-y0, ix0-x0:ix1-x0] = 255
        return fm

    def composite(src, dst, cm, seamless):
        m = (cm > 127).astype(np.uint8); ys, xs = np.where(m > 0)
        if len(xs) == 0: return dst
        touches = bool(m[0,:].any() or m[-1,:].any() or m[:,0].any() or m[:,-1].any())
        if seamless and not touches:
            try:
                return cv2.seamlessClone(src.astype(np.uint8), dst.astype(np.uint8),
                                         (m*255).astype(np.uint8), (int(xs.mean()), int(ys.mean())), cv2.NORMAL_CLONE)
            except cv2.error:
                pass
        al = cv2.GaussianBlur(m.astype(np.float32), (0,0), 3.0)[..., None]
        return (src.astype(np.float32)*al + dst.astype(np.float32)*(1-al)).astype(np.uint8)

    def fill(orig, mask, fb, use_gm, seamless):
        out = orig.copy(); H, W = orig.shape[:2]
        n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
        regimes = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < 200: continue
            x0, y0 = max(0,x-CTX), max(0,y-CTX); x1, y1 = min(W,x+w+CTX), min(H,y+h+CTX)
            crop = out[y0:y1, x0:x1]; cm = ((lab[y0:y1,x0:x1]==i).astype(np.uint8))*255
            facem = face_mask_in_crop(fb, x0, y0, x1, y1)
            edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
            hole = cm > 127
            dil = cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(97,97))) > 0
            band = dil & (~hole) & (facem == 0)
            ed = float((edges[band] > 0).mean()) if band.any() else 0.0
            regimes.append((int(a), ed))
            ch, cw = crop.shape[:2]; s = min(1.0, WORK/max(ch,cw))
            if s < 1.0:
                dw, dh = max(8,int(round(cw*s))), max(8,int(round(ch*s)))
                dcrop = cv2.resize(crop,(dw,dh),interpolation=cv2.INTER_AREA)
                dcm = cv2.resize(cm,(dw,dh),interpolation=cv2.INTER_NEAREST)
                dgm = cv2.resize(facem,(dw,dh),interpolation=cv2.INTER_NEAREST) if use_gm else None
            else:
                dcrop, dcm, dgm = crop, cm, (facem if use_gm else None)
            res = run_pm(cv2.cvtColor(dcrop, cv2.COLOR_BGR2RGB), dcm, dgm)
            res_bgr = cv2.resize(cv2.cvtColor(res, cv2.COLOR_RGB2BGR), (cw, ch), interpolation=cv2.INTER_CUBIC)
            out[y0:y1, x0:x1] = composite(res_bgr, out[y0:y1, x0:x1], cm, seamless)
        return out, regimes

    def lbl(im, t):
        im = im.copy(); cv2.rectangle(im,(0,0),(im.shape[1],28),(0,0,0),-1)
        cv2.putText(im,t,(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA); return im

    for idx in SCENES:
        t0 = time.time()
        orig = dl(f"orig_{idx:03d}.png"); mask = dl(f"mask_{idx:03d}.png")
        fb = face_boxes(orig)
        pm_pure, regimes = fill(orig, mask, fb, use_gm=False, seamless=False)
        pm_hyb, _ = fill(orig, mask, fb, use_gm=True, seamless=True)
        nf_pure, nf_hyb = new_faces(orig, pm_pure, mask), new_faces(orig, pm_hyb, mask)
        # zoom montage
        ys, xs = np.where(mask > 127); mg = 80
        x0, y0 = max(0,xs.min()-mg), max(0,ys.min()-mg); x1, y1 = min(orig.shape[1],xs.max()+mg), min(orig.shape[0],ys.max()+mg)
        sc = min(3.0, 640/max(1,y1-y0))
        def zc(im,t): return lbl(cv2.resize(im[y0:y1,x0:x1],None,fx=sc,fy=sc,interpolation=cv2.INTER_NEAREST), t)
        sep = np.full((int((y1-y0)*sc),4,3),(0,255,255),np.uint8)
        z = np.hstack([zc(orig,"ORIG"),sep, zc(pm_pure,f"PM-pure +{nf_pure}f"),sep, zc(pm_hyb,f"PM+gm+seamless +{nf_hyb}f")])
        p = f"/content/hyb_{idx:03d}.png"; cv2.imwrite(p, z)
        url = upload(p)
        reg = " ".join(f"{a}:ed{ed:.3f}{'PM' if ed>=TEX_THRESH else 'LaMa'}" for a,ed in regimes)
        notify(f"scene_{idx:03d} {time.time()-t0:.0f}s facesPure={nf_pure} facesHyb={nf_hyb} | {reg} | {url}")
        print(f"scene_{idx:03d} -> {url}", flush=True)
    notify("ALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        notify("ERROR\n" + traceback.format_exc())
        raise
