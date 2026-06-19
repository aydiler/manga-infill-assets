# Bootstraps the STABLE kernel-websocket route on Colab. One-cell paste:
#   !wget -qO /content/kl.py https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/colab_kernel_launch.py
#   %run /content/kl.py
import os, urllib.request
os.environ.setdefault("KGW_NAME", "colab")
RAW = "https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/colab_kernel_boot.py"
urllib.request.urlretrieve(RAW, "/content/kboot.py")
print("[kl] launching kernel-gateway bootstrap", flush=True)
exec(compile(open("/content/kboot.py").read(), "kboot.py", "exec"), {"__name__": "__main__"})
