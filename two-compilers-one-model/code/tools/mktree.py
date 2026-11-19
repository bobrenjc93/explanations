"""mktree.py SIDE SHA DEST: the standalone tree of one recorded checkpoint (what `measure --official` snapshotted):
DEST/torch to import from, DEST/hill_config.py to build with."""
import os, shutil, subprocess, sys

side, sha, dest = sys.argv[1:4]
H = "/home/dev/local/hill"
if os.path.exists(dest):
    sys.exit(f"{dest} exists")
if side == "tc":
    os.makedirs(dest)
    archive = subprocess.Popen(["git", "-C", f"{H}/tc/tree", "archive", sha], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", dest], stdin=archive.stdout, check=True)
else:
    subprocess.run(["cp", "-a", f"{H}/pristine", dest], check=True)
    archive = subprocess.Popen(["git", "-C", f"{H}/vm/snap", "archive", sha, "torch", "hill_config.py", "_cfg_rs.so"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", dest], stdin=archive.stdout, check=True)
    link = f"{dest}/torch/_cfg_rs.cpython-312-x86_64-linux-gnu.so"
    os.unlink(link)
    os.symlink(f"{dest}/_cfg_rs.so", link)
print(dest)
