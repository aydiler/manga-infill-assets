"""colab_fleet_boot.py — the ONE universal worker. Same file runs identically on:

  * Colab     (FLEET_EXPOSE=tunnel    -> opens a free cloudflared quick tunnel)
  * Colab/TS  (FLEET_EXPOSE=tailscale -> stable MagicDNS url via `tailscale serve`; survives restarts —
               no rotating url / NXDOMAIN / 530 like quick tunnels. Set FLEET_TS_URL to skip discovery.)
  * any host  (FLEET_EXPOSE=host      -> binds 0.0.0.0, announces FLEET_HOST ip — e.g. a Tailscale IP)
  * localhost (FLEET_EXPOSE=local     -> binds 127.0.0.1, announces it — for same-machine fleet tests)

It is a persistent-namespace exec server (HTTP) that DIALS HOME: it announces its
{name,url,token,device} to an ntfy topic AND drops ~/.colab/announce/<name>.json, so the
local client never has to scrape a browser to learn where the runtime is. A heartbeat
re-announces every 30s (the registry tracks last_seen). HTTP carries the actual RPC
(large payloads, synchronous replies); ntfy only carries the tiny discovery handshake —
so we keep the proven tunnel transport but lose the "how do I find it / which instance is
which" pain, and multi-instance becomes free (one topic, N self-identifying workers).

Endpoints (all gated by the X-Token header):
  GET  /ping              -> {ok, name, device}
  POST /exec   {code}     -> {stdout, error, artifacts, secs}      run in the persistent dict G
  POST /bg     {code,job} -> {started}                             run in a daemon thread (long ops)
  GET  /poll?job=         -> {log, done, err, secs, artifacts}     fetch a bg job's live state
  GET  /get?path=         -> raw bytes                             stream an artifact back

Config via env (all optional):
  FLEET_NAME   instance name              (default: hostname)
  FLEET_EXPOSE tunnel|tailscale|host|local (default: local)
  FLEET_HOST   ip to announce for host    (default: auto-detected)
  FLEET_TS_URL stable url for tailscale    (default: derived from `tailscale serve` + MagicDNS)
  FLEET_PORT   bind port                  (default: 8077)
  FLEET_TOKEN  fixed auth token           (default: random)
  FLEET_TOPIC  ntfy announce topic        (default: colab-fleet-ahmet-9z3)
  FLEET_NTFY   ntfy base url              (default: https://ntfy.sh)
"""
import os, io, re, json, time, socket, secrets, threading, platform, subprocess, contextlib, traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAME   = os.environ.get("FLEET_NAME") or platform.node() or "worker"
EXPOSE = os.environ.get("FLEET_EXPOSE", "local")
PORT   = int(os.environ.get("FLEET_PORT", "8077"))
TOKEN  = os.environ.get("FLEET_TOKEN") or secrets.token_hex(8)
TOPIC  = os.environ.get("FLEET_TOPIC", "colab-fleet-ahmet-9z3")
NTFY   = os.environ.get("FLEET_NTFY", "https://ntfy.sh").rstrip("/")
ANNOUNCE_DIR = os.path.expanduser("~/.colab/announce")


def detect_device():
    """Self-identify so the registry never lies about which runtime this is
    (kills the 'wrong runtime / 2 active sessions / toolbar says X' trap)."""
    d = {"host": platform.node(), "py": platform.python_version(), "machine": platform.machine()}
    try:
        import torch
        if torch.cuda.is_available():
            d["accel"] = torch.cuda.get_device_name(0); d["kind"] = "cuda"
    except Exception:
        pass
    if "accel" not in d:
        try:
            import jax
            devs = jax.devices()
            d["accel"] = str(devs); d["kind"] = devs[0].platform if devs else "cpu"
        except Exception:
            pass
    d.setdefault("accel", "cpu"); d.setdefault("kind", "cpu")
    try:
        d["env"] = ("kaggle" if os.path.exists("/kaggle") else
                    "colab" if os.path.exists("/content") else "host")
    except Exception:
        pass
    return d


DEVICE = detect_device()
# persistent experiment namespace — a model loaded in call N is still resident in call N+1
G = {"__name__": "fleet", "G": None, "_jobs": {}}
G["G"] = G


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "127.0.0.1"


_BG_DONE = "__done__"


def _run_bg(job, code):
    log = io.StringIO()
    G["_jobs"][job] = {"done": False, "log": log, "err": None, "secs": None, "artifacts": []}

    def runner():
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                exec(code, G)
            arts = G.pop("RESULT", None) or []
            G["_jobs"][job]["artifacts"] = [arts] if isinstance(arts, str) else list(arts)
        except BaseException:
            # BaseException, not Exception: sys.exit()/SystemExit in a fleet job
            # used to vanish here, leaving the job "running" forever (wedged).
            tb = traceback.format_exc(); G["_jobs"][job]["err"] = tb; log.write(tb)
        G["_jobs"][job]["secs"] = round(time.time() - t0, 1)
        G["_jobs"][job]["done"] = True

    threading.Thread(target=runner, daemon=True).start()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
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
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_GET(self):
        import urllib.parse as up
        u = up.urlparse(self.path)
        if not self._auth():
            return self._send(403, {"error": "auth"})
        if u.path == "/ping":
            return self._send(200, {"ok": True, "name": NAME, "device": DEVICE})
        if u.path == "/poll":
            job = up.parse_qs(u.query).get("job", [""])[0]
            j = G["_jobs"].get(job)
            if not j:
                return self._send(404, {"error": "no such job", "known": list(G["_jobs"])})
            return self._send(200, {"log": j["log"].getvalue(), "done": j["done"],
                                    "err": j["err"], "secs": j["secs"], "artifacts": j["artifacts"]})
        if u.path == "/get":
            p = up.parse_qs(u.query).get("path", [""])[0]
            try:
                with open(p, "rb") as f:
                    return self._send(200, f.read(), "application/octet-stream")
            except Exception as e:
                return self._send(404, {"error": str(e)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        import urllib.parse as up
        u = up.urlparse(self.path)
        if not self._auth():
            return self._send(403, {"error": "auth"})
        ln = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(ln) or b"{}")
        if u.path == "/bg":
            job = body.get("job", "job")
            _run_bg(job, body["code"])
            return self._send(200, {"started": job})
        if u.path == "/exec":
            buf = io.StringIO(); err = None; t0 = time.time()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(body["code"], G)
            except Exception:
                err = traceback.format_exc(); buf.write(err)
            arts = G.pop("RESULT", None) or []
            if isinstance(arts, str):
                arts = [arts]
            return self._send(200, {"stdout": buf.getvalue(), "error": err,
                                    "artifacts": arts, "secs": round(time.time() - t0, 1)})
        return self._send(404, {"error": "not found"})


def announce(url):
    rec = {"name": NAME, "url": url, "token": TOKEN, "device": DEVICE, "ts": int(time.time())}
    body = json.dumps(rec).encode()
    # 1) same-machine drop file (works with zero network)
    try:
        os.makedirs(ANNOUNCE_DIR, exist_ok=True)
        tmp = os.path.join(ANNOUNCE_DIR, "." + NAME + ".tmp")
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, os.path.join(ANNOUNCE_DIR, NAME + ".json"))
    except Exception:
        pass
    # 2) ntfy announce (cross-network discovery) — SKIP when FLEET_NTFY=off
    #    (Colab IPs get ntfy-429'd for hours anyway; the autobot reads the URL
    #    from the notebook DOM, so ntfy is pure noise there.)
    if NTFY in ("off", "none", "disabled", ""):
        return
    try:
        req = urllib.request.Request(NTFY + "/" + TOPIC, data=body,
                                     headers={"Title": "fleet-announce", "Tags": NAME})
        urllib.request.urlopen(req, timeout=10).read()
        announce._n = 0
    except Exception as e:
        # bounded printing: Colab IPs stay ntfy-429'd for hours, and one print per
        # 30s heartbeat = thousands of saved notebook lines, which wedges the tab
        # renderer on slow machines when the notebook reloads (seen 2026-10-11).
        announce._n = getattr(announce, "_n", 0) + 1
        if announce._n <= 1 or announce._n % 100 == 0:
            print(f"[announce] ntfy failed ({announce._n}x):", e, flush=True)


def start_tunnel():
    import shutil
    cf = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if not os.path.exists(cf):
        arch = "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64"
        dst = cf if os.access(os.path.dirname(cf) or "/", os.W_OK) else \
            os.path.expanduser("~/.local/bin/cloudflared")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/"
                       f"download/cloudflared-linux-{arch} -O {dst} && chmod +x {dst}",
                       shell=True, check=True)
        cf = dst
    proc = subprocess.Popen([cf, "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = None
    for line in proc.stdout:
        m = re.search(r"https://[\w-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0); break
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
    return url, proc


def _tailscale_url():
    """Stable MagicDNS url for the worker — survives PC/runtime restarts (unlike a cloudflared
    quick tunnel's rotating, NXDOMAIN-prone hostname). FLEET_TS_URL wins (explicit override / skips
    discovery). Else `tailscale serve` the local port over HTTPS and read Self.DNSName from
    `tailscale status --json`. Returns None when neither is available (no FLEET_TS_URL and no
    usable `tailscale` CLI) so main() can fail with a clear message."""
    env = os.environ.get("FLEET_TS_URL")
    if env:
        return env.rstrip("/")
    import shutil
    if not shutil.which("tailscale"):
        return None
    try:
        subprocess.run(["tailscale", "serve", "--bg", "--https=443", f"http://localhost:{PORT}"],
                       check=False, capture_output=True, text=True, timeout=20)
        st = json.loads(subprocess.run(["tailscale", "status", "--json"],
                                       capture_output=True, text=True, timeout=15).stdout or "{}")
        dns = ((st.get("Self") or {}).get("DNSName") or "").rstrip(".")
        return f"https://{dns}" if dns else None
    except Exception:
        return None


def _expose_url(expose):
    """Compute the announce URL for a NON-tunnel expose mode WITHOUT binding the HTTP server (pure
    enough to unit-test). 'tunnel' is excluded — it needs a live cloudflared subprocess (main())."""
    if expose == "tailscale":
        return _tailscale_url()
    if expose == "host":
        return "http://%s:%d" % (os.environ.get("FLEET_HOST") or _lan_ip(), PORT)
    return "http://127.0.0.1:%d" % PORT          # local


def main():
    bind = "127.0.0.1" if EXPOSE == "local" else "0.0.0.0"
    srv = ThreadingHTTPServer((bind, PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.5)

    if EXPOSE == "tunnel":
        url, proc = start_tunnel()
    elif EXPOSE == "tailscale":
        proc = None
        url = _expose_url("tailscale")
        if not url:
            raise SystemExit("FLEET_EXPOSE=tailscale but no FLEET_TS_URL set and `tailscale` CLI "
                             "unavailable — cannot derive a stable url")
    else:
        proc = None
        url = _expose_url(EXPOSE)

    print("[fleet] name=%s expose=%s url=%s token=%s device=%s"
          % (NAME, EXPOSE, url, TOKEN, DEVICE.get("accel")), flush=True)
    announce(url)
    print("[fleet] announced to %s/%s and %s/%s.json" % (NTFY, TOPIC, ANNOUNCE_DIR, NAME), flush=True)

    # heartbeat: re-announce (keeps last_seen fresh), RESTART the tunnel if the cloudflared
    # subprocess died (quick tunnels die randomly; the old worker left a live-but-unreachable
    # url in the registry — seen on kaggle1 2026-10-11), and keeps this foreground loop
    # running = Colab "activity" so the runtime won't idle-disconnect.
    while True:
        time.sleep(30)
        if proc is not None and proc.poll() is not None:
            print(f"[fleet] cloudflared died (rc={proc.returncode}) — restarting tunnel", flush=True)
            try:
                url, proc = start_tunnel()
                print(f"[fleet] new url: {url}", flush=True)
            except Exception as e:
                print(f"[fleet] tunnel restart failed: {e}", flush=True)
        announce(url)


if __name__ == "__main__":
    main()
