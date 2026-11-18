"""With CHUNK_COUNTS=FILE in the environment: counts the chunks of the interpreter's frame stack that the dashboard
runner's timed loop asks for, without adding a frame to any stack. The runner's progress bar marks the start of that
loop: there the arena allocator gets a counting wrapper (arena_cache.c, built as CHUNK_LIB), on top of whatever torch.rewrite put
there, so that a request counts whether or not a kept chunk answers it. CHUNK_KEEP=1 has the wrapper keep the chunks
that are freed, which is what torch.rewrite's runtime does for itself; for a run of torch.compile it takes the same
accident out of the timed loop. One line is appended to FILE when the process ends."""
import atexit
import os
import sys

if os.environ.get("CHUNK_COUNTS"):
    import ctypes

    import tqdm.auto as _auto

    _inner = _auto.trange
    _state = {}

    def trange(*args, **kwargs):
        bar = _inner(*args, **kwargs)
        if kwargs.get("desc") == "running benchmark" and not _state:
            lib = ctypes.PyDLL(os.environ["CHUNK_LIB"])  # arena_cache.so, or arena_who.so
            for name in ("arena_small_allocs", "arena_small_frees", "arena_reused"):
                getattr(lib, name).restype = ctypes.c_long
            lib.arena_install(int(os.environ.get("CHUNK_KEEP", "0")))
            _state["lib"] = lib

            def marked():
                # A generator's frame is on the stack only while it hands out the next item, not during the loop's
                # body. The first repetition is counted apart: a compile in it asks for thousands of chunks once.
                first = True
                for item in bar:
                    if not first and "first" not in _state:
                        _state["first"] = lib.arena_small_allocs()
                    first = False
                    yield item
                _state["loop"] = lib.arena_small_allocs()

            return marked()
        return bar

    _auto.trange = trange

    def _told():
        lib = _state.get("lib")
        if lib is None:
            return
        if hasattr(lib, "arena_report"):  # arena_who.so: who asked, with how much left, and the frames of the thread
            lib.arena_report()
        argv = " ".join(a for a in sys.argv if a.startswith("--") or not a.startswith("/"))
        with open(os.environ["CHUNK_COUNTS"], "a") as f:
            loop, first = _state.get("loop", -1), _state.get("first", 0)
            f.write(f"{lib.arena_small_allocs()} asked, {lib.arena_reused()} reused, keep {os.environ.get('CHUNK_KEEP', '0')}, {loop} in the loop, {loop - first} after its first repetition | "
                    f"{os.path.basename(sys.argv[0])} {argv}\n")

    atexit.register(_told)
