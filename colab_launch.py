# Bootstraps the universal fleet worker on a Colab runtime. Cell that runs this needs only:
#   !wget -qO /content/cl.py https://raw.githubusercontent.com/aydiler/manga-infill-assets/main/colab_launch.py
#   %run /content/cl.py
import os, urllib.request
os.environ.setdefault("FLEET_NAME", "colab")
os.environ["FLEET_EXPOSE"] = "tunnel"
os.environ.setdefault("FLEET_TOPIC", "colab-fleet-ahmet-9z3")
RAW = "https://raw.githubusercontent.com/aydiler/manga-infill-assets/master/colab_fleet_boot.py"
urllib.request.urlretrieve(RAW, "/content/boot.py")
print("[cl] fetched worker, launching as", os.environ["FLEET_NAME"], flush=True)
exec(compile(open("/content/boot.py").read(), "boot.py", "exec"), {"__name__": "__main__"})
