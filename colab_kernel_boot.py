"""colab_kernel_boot.py — expose Colab's OWN jupyter-server for the stable kernel-websocket route.

Paste into ONE Colab cell (or %run via the launcher) and run it. It:
  1. finds the runtime's standard `jupyter-server` (v2.x, tokenless, bound to the VM-internal
     IP, normally 172.28.0.12:9000 — discovered from `ss`, falls back to COLAB_JUPYTER_IP),
  2. opens a free cloudflared quick tunnel to it,
  3. announces the tunnel URL to an ntfy topic (dial-home; kind="kernel"),
  4. holds the cell open (foreground loop = Colab "activity" so the VM won't idle-disconnect).

Then drive it from the laptop with colab_kernel.py over the documented Jupyter REST + `/channels`
websocket protocol — POST /api/kernels for a fresh kernel, execute_request to run code. No CDP,
no Monaco, no browser after this one cell. The notebook's own kernel stays untouched (this cell
just keeps it busy as the keepalive); your driving happens on a separate API-created kernel.

SECURITY: Colab's :9000 is tokenless, so the tunnel URL is the only secret — it's random and
short-lived (like any trycloudflare quick tunnel). Don't share the URL.

Env (optional): KGW_NAME (default 'colab'), KGW_TOPIC (default 'colab-kernel-ahmet-9z3'), KGW_NTFY.
"""
import os, re, json, time, shutil, platform, threading, subprocess, urllib.request

NAME  = os.environ.get("KGW_NAME", "colab")
TOPIC = os.environ.get("KGW_TOPIC", "colab-kernel-ahmet-9z3")
NTFY  = os.environ.get("KGW_NTFY", "https://ntfy.sh").rstrip("/")


def jupyter_addr():
    """Bind addr of the running jupyter-server, read from `ss` so we adapt if Colab moves it."""
    try:
        out = subprocess.getoutput("ss -ltnp 2>/dev/null")
        for line in out.splitlines():
            if "jupyter-server" in line:
                m = re.search(r"(\d+\.\d+\.\d+\.\d+):(\d+)", line)
                if m:
                    return m.group(1), int(m.group(2))
    except Exception:
        pass
    return os.environ.get("COLAB_JUPYTER_IP", "172.28.0.12"), 9000


def start_tunnel(host, port):
    cf = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if not os.path.exists(cf):
        arch = "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64"
        subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/"
                       f"download/cloudflared-linux-{arch} -O {cf} && chmod +x {cf}",
                       shell=True, check=True)
    proc = subprocess.Popen([cf, "tunnel", "--url", f"http://{host}:{port}", "--no-autoupdate"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = None
    for line in proc.stdout:
        m = re.search(r"https://[\w-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0); break
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
    return url, proc


def announce(url):
    rec = json.dumps({"name": NAME, "kernel_url": url, "ts": int(time.time()),
                      "kind": "kernel"}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            NTFY + "/" + TOPIC, data=rec, headers={"Title": "kernel-announce", "Tags": NAME}),
            timeout=10).read()
    except Exception as e:
        print("[kgw] announce failed:", e, flush=True)


host, port = jupyter_addr()
print(f"[kgw] jupyter-server at {host}:{port}", flush=True)
URL, _proc = start_tunnel(host, port)
print(f"[kgw] tunnel {URL}  ->  {host}:{port}", flush=True)
# sanity check the tunnel actually reaches the server
try:
    v = urllib.request.urlopen(URL + "/api", timeout=20).read().decode()
    print("[kgw] reachable, jupyter-server", v, flush=True)
except Exception as e:
    print("[kgw] WARNING tunnel not reachable yet:", e, flush=True)
announce(URL)
print(f"[kgw] announced '{NAME}' kernel_url={URL} to {NTFY}/{TOPIC}", flush=True)
print("[kgw] drive from laptop:  colab-kernel info  /  colab-kernel exec -", flush=True)
print("[keepalive] cell held open so the runtime stays active; drive via the tunnel.", flush=True)
while True:
    time.sleep(30)
    announce(URL)
