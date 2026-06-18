"""Colab v6e RPC bootstrap — paste this whole file into ONE Colab cell and run it.

Turns the connected TPU runtime into a persistent-namespace exec server reachable from a
laptop over a free cloudflared "quick tunnel" (no account/token). After it prints
TUNNEL_URL + TOKEN, all further experimentation is driven from the local machine via
scripts/colab_rpc.py — no more browser interaction needed.

Design notes:
  * /exec runs Python in ONE persistent global dict G, so a model loaded in call N is still
    resident in call N+1 (load SD3/Flux once, iterate masks/prompts for free).
  * stdout+stderr+traceback are captured and returned as JSON.
  * If a call assigns `RESULT = ["/content/out/foo.png", ...]`, those files are streamable
    back through /get?path=... (driver downloads them to ./colab_results/).
  * cloudflared runs as a detached subprocess; a drain thread keeps its pipe from filling.
  * X-Token header gates every endpoint (the tunnel URL is public).
"""
import os, sys, io, re, time, json, secrets, threading, platform, subprocess, contextlib, traceback

print("[1/5] Verifying TPU is visible to JAX ...", flush=True)
import jax
print("      jax", jax.__version__, "devices:", jax.devices(), flush=True)

print("[2/5] Installing flask + cloudflared ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "flask"], check=False)
CF = "/usr/local/bin/cloudflared"
if not os.path.exists(CF):
    arch = "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64"
    subprocess.run(
        f"wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
        f"cloudflared-linux-{arch} -O {CF} && chmod +x {CF}", shell=True, check=True)
print("      cloudflared ready:", os.path.exists(CF), flush=True)

print("[3/5] Starting exec server on :8000 ...", flush=True)
from flask import Flask, request, jsonify, send_file
TOKEN = secrets.token_hex(8)
G = {"__name__": "colab", "jax": jax}            # persistent experiment namespace
app = Flask(__name__)


def _auth():
    return request.headers.get("X-Token") == TOKEN


@app.post("/exec")
def _exec():
    if not _auth():
        return jsonify(error="auth"), 403
    code = request.get_json(force=True)["code"]
    buf = io.StringIO()
    err = None
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, G)
    except Exception:
        err = traceback.format_exc()
        buf.write(err)
    arts = G.pop("RESULT", None) or []
    if isinstance(arts, str):
        arts = [arts]
    return jsonify(stdout=buf.getvalue(), error=err, artifacts=arts, secs=round(time.time() - t0, 1))


@app.get("/get")
def _get():
    if not _auth():
        return "auth", 403
    return send_file(request.args["path"])


@app.get("/ping")
def _ping():
    if not _auth():
        return "auth", 403
    return jsonify(ok=True, devices=str(jax.devices()))


threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8000, threaded=True), daemon=True).start()
time.sleep(2)

print("[4/5] Opening cloudflared quick tunnel ...", flush=True)
proc = subprocess.Popen([CF, "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
URL = None
for line in proc.stdout:
    if "trycloudflare.com" in line:
        m = re.search(r"https://[\w-]+\.trycloudflare\.com", line)
        if m:
            URL = m.group(0)
            break
# keep draining cloudflared's stdout so its pipe never blocks
threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()

print("[5/5] READY. Configure the local driver with these two lines:\n", flush=True)
print("TUNNEL_URL:", URL, flush=True)
print("TOKEN:", TOKEN, flush=True)
print('\nLocal: echo \'{"url":"%s","token":"%s"}\' > ~/.colab_v6e_rpc.json' % (URL, TOKEN), flush=True)

# Also push url+token out-of-band so the laptop can pick it up without scraping Colab's output iframe.
try:
    import urllib.request as _u
    _u.urlopen(_u.Request("https://ntfy.sh/manga-v6e-rpc-7x2",
                          data=("RPC %s %s" % (URL, TOKEN)).encode()), timeout=10)
    print("      posted url+token to ntfy topic manga-v6e-rpc-7x2", flush=True)
except Exception as e:
    print("      ntfy post failed:", e, flush=True)
