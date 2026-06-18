"""Colab GPU (T4) — FULL-BLEED outpaint: fill the empty panel gutters/margins with generated scenery.

Operator reframe: Qwen-Image-EDIT won't draw into large empty voids (it's an editor, not an outpainter —
left 54-83% of gutters black/white, §33). A true INPAINT model DOES generate into masked regions, so use
SDXL-inpaint to extend the scene edge-to-edge into the empty black/white gutters (+ remove bubbles), turning
the unsolvable panel-boundary problem into scene extension.

Self-contained: downloads orig_NNN + mask_NNN (bubble) from the assets repo, computes the gutter+bubble mask
on the fly (large near-black/near-white regions TOUCHING the image border = gutters; union the bubble mask),
SDXL-inpaints that whole region with an "extend the scene" prompt, composites 0-leak, montages ORIG | OUTPAINT
at full res, and posts each result URL via ntfy. Defect-hunt per methodology: does it fill the WHOLE gutter
(vs leave black/white)? sharp? plausible/in-style? hallucinated subjects? seam at the gutter<->content join?

Run cell (Colab, T4):
  !wget -qO /content/colab_outpaint.py https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/colab_outpaint.py && python /content/colab_outpaint.py
"""
import os, sys, json, time, traceback

TOPIC = "manga-infill-colab-7x2"
BASE = "https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/"
SCENES = [int(x) for x in os.environ.get("SCENES", "42,17,27,44").split(",")]
MODEL = os.environ.get("MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
WORK = int(os.environ.get("WORK", "1024"))          # long-edge working res for the inpaint pass
STEPS = int(os.environ.get("STEPS", "30"))
GUID = float(os.environ.get("GUID", "7.5"))
STRENGTH = float(os.environ.get("STRENGTH", "0.99"))
TAG = os.environ.get("TAG", "OUTPAINT")
PROMPT = os.environ.get("PROMPT",
    "manga webtoon illustration, extend the existing scene's background and scenery edge to edge, "
    "matching the surrounding art style, colours, lighting and shading, seamless full-bleed, "
    "sharp clean detail, atmospheric background only")
NEG = os.environ.get("NEG",
    "new character, person, face, figure, hand, speech bubble, text, letters, watermark, "
    "blurry, lowres, jpeg artifacts, deformed, duplicated, frame border, black bars, white border, panel gap")


def notify(t):
    import requests
    try: requests.post("https://ntfy.sh/" + TOPIC, data=t.encode()[:3500])
    except Exception as e: print("notify failed", e, flush=True)


def upload(p):
    import requests
    r = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": open(p, "rb")})
    return json.loads(r.text)["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")


def gutter_mask(orig, bubble):
    """bubble mask UNION the empty gutters/margins (large near-black/near-white uniform regions that
    TOUCH the image border). Border-touching restriction keeps INTERIOR panel separators intact (so a
    multi-sub-panel page is not merged into one scene)."""
    import cv2, numpy as np
    g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY); H, W = g.shape
    uniform = cv2.erode(((g < 20) | (g > 235)).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(uniform, 8)
    gut = np.zeros((H, W), np.uint8)
    for j in range(1, n):
        x, y, w, h, a = stats[j]
        if a < 0.01 * H * W:
            continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:   # touches a frame edge => gutter
            gut[lab == j] = 255
    gut = cv2.dilate(gut, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.bitwise_or(gut, bubble)


def main():
    os.system(f"{sys.executable} -m pip -q install diffusers transformers accelerate safetensors requests")
    import torch, cv2, numpy as np, urllib.request
    from PIL import Image
    from diffusers import AutoPipelineForInpainting
    notify(f"{TAG} SETUP torch {torch.__version__} cuda={torch.cuda.is_available()} model={MODEL}")
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16")
    except Exception:
        pipe = AutoPipelineForInpainting.from_pretrained(MODEL, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()          # T4-safe (SDXL won't fully fit 16GB otherwise)
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    pipe.set_progress_bar_config(disable=True)

    def dl(name, gray=False):
        b = urllib.request.urlopen(BASE + name, timeout=60).read()
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)

    def lbl(im, t):
        im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(im, t, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); return im

    for idx in SCENES:
        t0 = time.time()
        orig = dl(f"orig_{idx:03d}.png"); bub = dl(f"mask_{idx:03d}.png", gray=True)
        H, W = orig.shape[:2]
        m = gutter_mask(orig, bub)
        # working resolution: long edge -> WORK, multiple of 8
        s = min(1.0, WORK / max(H, W))
        rw, rh = max(64, int(round(W * s / 8) * 8)), max(64, int(round(H * s / 8) * 8))
        img = Image.fromarray(cv2.cvtColor(cv2.resize(orig, (rw, rh)), cv2.COLOR_BGR2RGB))
        msk = Image.fromarray(cv2.resize(m, (rw, rh), interpolation=cv2.INTER_NEAREST))
        gen = torch.Generator("cuda").manual_seed(0)
        res = pipe(prompt=PROMPT, negative_prompt=NEG, image=img, mask_image=msk,
                   num_inference_steps=STEPS, guidance_scale=GUID, strength=STRENGTH,
                   width=rw, height=rh, generator=gen).images[0]
        ref = cv2.resize(cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR), (W, H), interpolation=cv2.INTER_CUBIC)
        al = cv2.GaussianBlur((m > 127).astype(np.float32), (0, 0), 4.0)[..., None]
        out = (ref.astype(np.float32) * al + orig.astype(np.float32) * (1 - al)).astype(np.uint8)
        # how much of the masked gutter did it actually FILL (vs leave near-black/white)?
        gf = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY); mb = m > 127
        still = float((((gf < 25) | (gf > 235)) & mb).sum()) / max(int(mb.sum()), 1)
        Hs = min(1400, H); sc = Hs / H
        rs = lambda im: cv2.resize(im, None, fx=sc, fy=sc)
        sep = np.full((rs(orig).shape[0], 5, 3), (0, 255, 255), np.uint8)
        p = f"/content/{TAG.lower()}_{idx:03d}.png"
        cv2.imwrite(p, np.hstack([lbl(rs(orig), "ORIG"), sep, lbl(rs(out), f"OUTPAINT still-empty={still:.0%}")]))
        notify(f"{TAG} scene_{idx:03d} {time.time()-t0:.0f}s still-empty={still:.0%} -> {upload(p)}")
        print(f"scene_{idx:03d} done still-empty={still:.0%}", flush=True)
    notify(f"{TAG} ALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        notify(f"{TAG} ERROR\n" + traceback.format_exc()); raise
