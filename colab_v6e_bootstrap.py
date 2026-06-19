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

print("[1/5] Verifying accelerator is visible to JAX ...", flush=True)
# GPU runtimes: stop jax preallocating 75% of VRAM up front (starves big models like SD3+T5-XXL);
# on-demand allocation lets a ~12GB model + render fit a 24GB card. No-op on TPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
print("      jax", jax.__version__, "devices:", jax.devices(), flush=True)

print("[2/5] Installing cloudflared ...", flush=True)
CF = "/usr/local/bin/cloudflared"
if not os.path.exists(CF):
    arch = "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64"
    subprocess.run(
        f"wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
        f"cloudflared-linux-{arch} -O {CF} && chmod +x {CF}", shell=True, check=True)
print("      cloudflared ready:", os.path.exists(CF), flush=True)

print("[3/5] Starting stdlib exec server on :8000 ...", flush=True)
import json, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
TOKEN = secrets.token_hex(8)
G = {"__name__": "colab", "jax": jax}            # persistent experiment namespace
G["G"] = G                                       # self-ref so exec'd code can use `G[...]` for resident state


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):                   # silence per-request logging
        pass

    def _auth(self):
        return self.headers.get("X-Token") == TOKEN

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if not self._auth():
            return self._send(403, {"error": "auth"})
        if u.path == "/ping":
            return self._send(200, {"ok": True, "devices": str(jax.devices())})
        if u.path == "/get":
            p = urllib.parse.parse_qs(u.query).get("path", [""])[0]
            try:
                with open(p, "rb") as f:
                    return self._send(200, f.read(), "application/octet-stream")
            except Exception as e:
                return self._send(404, {"error": str(e)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._auth():
            return self._send(403, {"error": "auth"})
        ln = int(self.headers.get("Content-Length", "0"))
        code = json.loads(self.rfile.read(ln))["code"]
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
        return self._send(200, {"stdout": buf.getvalue(), "error": err,
                                "artifacts": arts, "secs": round(time.time() - t0, 1)})


srv = ThreadingHTTPServer(("0.0.0.0", 8000), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
time.sleep(1)

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

# KEEP-ALIVE: hold this cell open forever. An actively-running cell registers as Colab activity, so
# the runtime won't idle-disconnect (background-thread RPC alone does NOT count as activity). The HTTP
# server + cloudflared run in their own threads; all work is driven via the tunnel, not via cells.
print("[keepalive] cell held open to keep the runtime active; drive everything via the tunnel.", flush=True)
while True:
    time.sleep(60)
