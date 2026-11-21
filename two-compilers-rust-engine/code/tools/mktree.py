"""mktree.py SIDE SHA DEST: the standalone tree of one recorded checkpoint (what `measure --official` snapshotted):
DEST/torch to import from, DEST/hill_config.py to build with (for vm also DEST/rust, the engine's crate, and the
engine library the checkpoint was measured with, in DEST/torch)."""
import os, subprocess, sys

side, sha, dest = sys.argv[1:4]
H = "/home/dev/local/hill2"
if os.path.exists(dest):
    sys.exit(f"{dest} exists")
os.makedirs(dest)
archive = subprocess.Popen(["git", "-C", f"{H}/{side}/tree", "archive", sha], stdout=subprocess.PIPE)
subprocess.run(["tar", "-x", "-C", dest], stdin=archive.stdout, check=True)
print(dest)
