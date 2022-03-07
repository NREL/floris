
import subprocess
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent

status = subprocess.call(["jupyter", "jekyllnb", f"{root_dir}/examples/00_getting_started.ipynb", "--site-dir", f"{root_dir}/docs", "--image-dir", "assets/images"])
if status != 0:
    sys.exit()

subprocess.call(["mv", f"{root_dir}/docs/00_getting_started.md", f"{root_dir}/docs/_tutorials/index.md"])

