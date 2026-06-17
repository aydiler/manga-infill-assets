"""Colab GPU — R3: full-regenerate INPAINT with an in-style (anime) prior + soft Tile-CN anchor.

Diffusion REFINE of the blurry LaMa fill is a dead end (R1/R2): anchored=blurry, unanchored=invents.
New angle: FULL-regenerate inpaint (so the hole gets fresh SHARP pixels) but with (a) an ANIME base
model (its prior is anime art, so "invention" should be in-style, not photoreal meadows/castles) and
(b) a SOFT Tile-ControlNet on the LaMa fill (color/layout anchor, not a hard blur-lock). Sweep the
CN scale [0.4/0.7/1.0] to find sharp-but-plausible. Anti-subject negative to suppress invented
faces/people. The hole regen is composited only inside the hull mask (0-leak; out base = LaMa).

Per scene -> montage ORIG | LaMa | inpaint@cn0.4 | @cn0.7 | @cn1.0 cropped to bbox at FULL res.
Methodology defect-hunt: is it SHARP? does it INVENT subjects? does it REPEAT? does it fit context?

Run cell:
  !wget -qO /content/colab_inpaint.py https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/colab_inpaint.py && python /content/colab_inpaint.py
"""
import os, sys, json, time, traceback

TOPIC = "manga-infill-colab-7x2"
BASE = "https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/"
SCENES = [int(x) for x in os.environ.get("SCENES", "17,49,42,18,58,3,27,44").split(",")]
MODEL = os.environ.get("MODEL", "xyn-ai/anything-v4.0")
TILE = "lllyasviel/control_v11f1e_sd15_tile"
CNS = [float(x) for x in os.environ.get("CNS", "0.4,0.7,1.0").split(",")]
CTX, WORK, STEPS, GUID = 256, 768, int(os.environ.get("STEPS", "28")), float(os.environ.get("GUID", "7.0"))
HULL = 22
TAG = os.environ.get("TAG", "R3")
PROMPT = ("manga webtoon panel, painted background, seamless continuation of the surrounding scene, "
          "sharp clean detail, consistent shading, no characters")
NEG = ("person, people, man, woman, face, eyes, portrait, head, human, body, figure, character, hand, "
       "building, castle, house, mountain, landscape, tree, creature, animal, object, "
       "speech bubble, text, letters, watermark, blurry, lowres, deformed, duplicated, frame border")


def notify(t):
    import requests
    try: requests.post("https://ntfy.sh/" + TOPIC, data=t.encode()[:3500])
    except Exception as e: print("notify failed", e, flush=True)


def upload(p):
    import requests
    r = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": open(p, "rb")})
    return json.loads(r.text)["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")


def main():
    os.system(f"{sys.executable} -m pip -q install diffusers transformers accelerate safetensors requests")
    import torch, cv2, numpy as np, urllib.request
    from PIL import Image
    from diffusers import (StableDiffusionControlNetInpaintPipeline, ControlNetModel,
                           UniPCMultistepScheduler)
    notify(f"{TAG} SETUP torch {torch.__version__} cuda={torch.cuda.is_available()} model={MODEL}")
    cn = ControlNetModel.from_pretrained(TILE, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        MODEL, controlnet=cn, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

    def dl(name, gray=False):
        b = urllib.request.urlopen(BASE + name, timeout=60).read()
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)

    def hull_mask(comp):
        pts = cv2.findNonZero(comp); m = np.zeros(comp.shape, np.uint8)
        if pts is not None:
            cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*HULL+1, 2*HULL+1)))
        return m

    def inpaint_one(lama_crop, cm, cnscale):
        ch, cw = lama_crop.shape[:2]
        s = min(1.0, WORK / max(ch, cw))
        rw, rh = max(64, int(round(cw*s/8)*8)), max(64, int(round(ch*s/8)*8))
        img = Image.fromarray(cv2.cvtColor(cv2.resize(lama_crop, (rw, rh)), cv2.COLOR_BGR2RGB))
        msk = Image.fromarray(cv2.resize(cm, (rw, rh), interpolation=cv2.INTER_NEAREST))
        g = torch.Generator("cuda").manual_seed(0)
        res = pipe(prompt=PROMPT, negative_prompt=NEG, image=img, mask_image=msk, control_image=img,
                   num_inference_steps=STEPS, guidance_scale=GUID, controlnet_conditioning_scale=cnscale,
                   generator=g, width=rw, height=rh).images[0]
        return cv2.resize(cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR), (cw, ch), interpolation=cv2.INTER_CUBIC)

    def lbl(im, t):
        im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(im, t, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); return im

    for idx in SCENES:
        t0 = time.time()
        orig = dl(f"orig_{idx:03d}.png"); lama = dl(f"lama_{idx:03d}.png"); mask = dl(f"mask_{idx:03d}.png", gray=True)
        H, W = orig.shape[:2]
        n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
        outs = {c: lama.copy() for c in CNS}
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < 200: continue
            x0, y0 = max(0, x-CTX), max(0, y-CTX); x1, y1 = min(W, x+w+CTX), min(H, y+h+CTX)
            comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8); cm = hull_mask(comp)
            al = cv2.GaussianBlur((cm > 127).astype(np.float32), (0, 0), 4.0)[..., None]
            lama_crop = lama[y0:y1, x0:x1]
            for c in CNS:
                ref = inpaint_one(lama_crop, cm, c).astype(np.float32)
                outs[c][y0:y1, x0:x1] = (ref*al + lama_crop.astype(np.float32)*(1-al)).astype(np.uint8)
        ys, xs = np.where(mask > 127); mg = 100
        bx0, by0 = max(0, xs.min()-mg), max(0, ys.min()-mg); bx1, by1 = min(W, xs.max()+mg), min(H, ys.max()+mg)
        sep = np.full((by1-by0, 5, 3), (0, 255, 255), np.uint8)
        cols = [lbl(orig[by0:by1, bx0:bx1], "ORIG"), sep, lbl(lama[by0:by1, bx0:bx1], "LaMa")]
        for c in CNS:
            cols += [sep, lbl(outs[c][by0:by1, bx0:bx1], f"inpaint cn{c}")]
        p = f"/content/{TAG.lower()}_{idx:03d}.png"; cv2.imwrite(p, np.hstack(cols))
        notify(f"{TAG} scene_{idx:03d} {time.time()-t0:.0f}s -> {upload(p)}")
        print(f"scene_{idx:03d} done", flush=True)
    notify(f"{TAG} ALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        notify(f"{TAG} ERROR\n" + traceback.format_exc()); raise
