"""overlay.py DEST: a tree of the torch package of the g build that stands on its own: every Python file a copy
(so it can be changed without changing the build tree), everything else a link to the build tree."""
import os, shutil, sys
SRC = "/home/dev/local/g/pytorch/torch"
INST = "/home/dev/local/g/pytorch/build/install/cp312-cp312-linux_x86_64/torch"
DEST = os.path.join(sys.argv[1], "torch")
LINK_DIRS = {"lib", "include", "bin", "share", "csrc", "test"}
COPY = (".py", ".pyi", ".yaml", ".yml", ".json", ".txt", ".toml", ".typed", ".md", ".cpp", ".h", ".jinja", ".j2")
copied = linked = 0
for root in (SRC, INST):  # the install tree second: its files win, as they do for the build's own finder
    for d, subdirs, files in os.walk(root):
        rel = os.path.relpath(d, root)
        if rel == ".":
            for name in list(subdirs):
                if name in LINK_DIRS:
                    subdirs.remove(name)
                    link = os.path.join(DEST, name)
                    if root == INST or not os.path.lexists(link):
                        if os.path.lexists(link):
                            os.unlink(link)
                        os.makedirs(DEST, exist_ok=True)
                        os.symlink(os.path.join(root, name), link)
                        linked += 1
        subdirs[:] = [s for s in subdirs if s != "__pycache__"]
        out = os.path.normpath(os.path.join(DEST, rel))
        os.makedirs(out, exist_ok=True)
        for name in files:
            a, b = os.path.join(d, name), os.path.join(out, name)
            if os.path.lexists(b):
                os.unlink(b)
            if name.endswith(COPY) and os.path.getsize(a) < 4 << 20:
                shutil.copy2(a, b); copied += 1
            else:
                os.symlink(a, b); linked += 1
print(f"{DEST}: {copied} files copied, {linked} linked")
