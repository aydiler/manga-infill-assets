"""Colab GPU — R1: structure-locked LOW-STRENGTH diffusion refine of the LaMa fill.

Hypothesis: full-regenerate diffusion hallucinates (meadows/castles/faces); but anchoring
to LaMa's correct colors+structure (img2img init = LaMa fill, Tile-ControlNet control =
LaMa fill) at LOW denoise strength should let the model add only HIGH-FREQUENCY detail —
sharpening LaMa's blur without inventing (no free regenerate) or repeating (not exemplar).
Sweep strength to find where it sharpens but doesn't drift.

Per scene -> montage: ORIG | LaMa | refine@s0.25 | refine@s0.40 | refine@s0.55, cropped to
the bubble bbox at FULL RES (for methodology defect-hunt). Uploaded to tmpfiles, URL via ntfy.

0-leak: out starts as the clean LaMa image; only the hole region (feathered) is overwritten
with the refined crop. Anti-subject negative to discourage faces/people.

Run cell:
  !wget -qO /content/colab_refine.py https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/colab_refine.py && python /content/colab_refine.py
"""
import os, sys, json, time, traceback

TOPIC = "manga-infill-colab-7x2"
BASE = "https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/"
SCENES = [17, 49, 42, 18, 58, 3, 27, 44]
MODEL = os.environ.get("REFINE_MODEL", "xyn-ai/anything-v4.0")  # colored-anime SD1.5
TILE = "lllyasviel/control_v11f1e_sd15_tile"
STRENGTHS = [float(x) for x in os.environ.get("STRENGTHS", "0.25,0.40,0.55").split(",")]
CTX, WORK, STEPS = 256, 768, int(os.environ.get("STEPS", "24"))
GUID = float(os.environ.get("GUID", "6.0"))
CN = float(os.environ.get("CN", "1.1"))
TAG = os.environ.get("TAG", "R1")
HULL = 22
PROMPT = ("detailed manga webtoon panel, sharp clean linework, crisp painted background, "
          "high quality, consistent shading")
NEG = ("blurry, soft, smear, lowres, jpeg artifacts, speech bubble, text, letters, watermark, "
       "deformed, extra face, person, people, portrait, head, human, figure, character, hand")


def notify(t):
    import requests
    try: requests.post("https://ntfy.sh/" + TOPIC, data=t.encode()[:3500])
    except Exception as e: print("notify failed", e, flush=True)


def upload(p):
    import requests
    r = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": open(p, "rb")})
    return json.loads(r.text)["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")


def main():
    subprocess_pip = f"{sys.executable} -m pip -q install diffusers transformers accelerate safetensors requests"
    os.system(subprocess_pip)
    import torch, cv2, numpy as np, urllib.request
    from PIL import Image
    from diffusers import (StableDiffusionControlNetImg2ImgPipeline, ControlNetModel,
                           UniPCMultistepScheduler)
    notify(f"{TAG} SETUP torch {torch.__version__} cuda={torch.cuda.is_available()} model={MODEL}")
    cn = ControlNetModel.from_pretrained(TILE, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        MODEL, controlnet=cn, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

    def dl(name, gray=False):
        b = urllib.request.urlopen(BASE + name, timeout=60).read()
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)

    def hull_mask(comp):
        pts = cv2.findNonZero(comp)
        m = np.zeros(comp.shape, np.uint8)
        if pts is not None:
            cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*HULL+1, 2*HULL+1)))
        return m

    def refine_one(lama_crop, strength):
        ch, cw = lama_crop.shape[:2]
        s = min(1.0, WORK / max(ch, cw))
        rw, rh = max(8, int(round(cw*s/8)*8)), max(8, int(round(ch*s/8)*8))
        small = cv2.resize(lama_crop, (rw, rh), interpolation=cv2.INTER_CUBIC)
        init = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        g = torch.Generator("cuda").manual_seed(0)
        res = pipe(prompt=PROMPT, negative_prompt=NEG, image=init, control_image=init,
                   strength=strength, num_inference_steps=STEPS, guidance_scale=GUID,
                   controlnet_conditioning_scale=CN, generator=g).images[0]
        return cv2.resize(cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR), (cw, ch), interpolation=cv2.INTER_CUBIC)

    def lbl(im, t):
        im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(im, t, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); return im

    for idx in SCENES:
        t0 = time.time()
        orig = dl(f"orig_{idx:03d}.png"); lama = dl(f"lama_{idx:03d}.png"); mask = dl(f"mask_{idx:03d}.png", gray=True)
        H, W = orig.shape[:2]
        n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
        outs = {s: lama.copy() for s in STRENGTHS}
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < 200: continue
            x0, y0 = max(0, x-CTX), max(0, y-CTX); x1, y1 = min(W, x+w+CTX), min(H, y+h+CTX)
            comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
            cm = hull_mask(comp)
            al = cv2.GaussianBlur((cm > 127).astype(np.float32), (0, 0), 4.0)[..., None]
            lama_crop = lama[y0:y1, x0:x1]
            for s in STRENGTHS:
                ref = refine_one(lama_crop, s).astype(np.float32)
                outs[s][y0:y1, x0:x1] = (ref*al + lama_crop.astype(np.float32)*(1-al)).astype(np.uint8)
        # montage cropped to mask bbox (+margin) at FULL res
        ys, xs = np.where(mask > 127); mg = 100
        bx0, by0 = max(0, xs.min()-mg), max(0, ys.min()-mg); bx1, by1 = min(W, xs.max()+mg), min(H, ys.max()+mg)
        sep = np.full((by1-by0, 5, 3), (0, 255, 255), np.uint8)
        cols = [lbl(orig[by0:by1, bx0:bx1], "ORIG"), sep, lbl(lama[by0:by1, bx0:bx1], "LaMa")]
        for s in STRENGTHS:
            cols += [sep, lbl(outs[s][by0:by1, bx0:bx1], f"refine s{s}")]
        p = f"/content/{TAG.lower()}_{idx:03d}.png"; cv2.imwrite(p, np.hstack(cols))
        notify(f"{TAG} scene_{idx:03d} {time.time()-t0:.0f}s -> {upload(p)}")
        print(f"scene_{idx:03d} done", flush=True)
    notify(f"{TAG} ALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        notify(f"{TAG} ERROR\n" + traceback.format_exc()); raise
