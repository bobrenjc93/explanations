# Tools

| file | what it does |
|---|---|
| `chunk_cliff.py` | plain Python, no PyTorch: 2000 calls of a small function from each of 200 depths of the stack; per depth the page faults and the time of a call. `python3.12 chunk_cliff.py` |
| `depth.py` | the dashboard runner's training step of one model, timed at growing depths of the caller (`PYTORCH=<checkout> python depth.py timm_models --only tf_efficientnet_b0`; `COMPILE=1` for `torch.compile`, `STEP`, `MOST`) |
| `chain.py` | the Python frames above the first kernel launch of the forward graph, with their sizes, under either compiler |
| `arena_cache.c` | a wrapper around CPython's arena allocator (`PyObject_SetArenaAllocator`) that counts the blocks of at most 256 KB asked of it (the chunks of the frame stack) and, with `arena_install(1)`, keeps the freed ones for reuse. `gcc -O2 -shared -fPIC -I<python include> -o arena_cache.so arena_cache.c` |
| `arena_who.c` | the same wrapper, counting only, that also records for each 16 KB request which function was running, on which line, how many bytes its chunk had left, and once the frames of the thread as the interpreter has them. Needs CPython 3.12's internal headers |
| `sitecustomize.py` | put its folder on `PYTHONPATH` with `CHUNK_COUNTS=<file>` and `CHUNK_LIB=<one of the two .so>`: installs the wrapper when the runner's timed loop starts, without adding a frame to any stack, and appends one line per process to the file (`CHUNK_KEEP=1` keeps freed chunks) |

`chunks_training.log` and `chunks_inference.log` in `../../results/` are what `sitecustomize.py` wrote during the paired runs of the
dashboard's suites: per run, the chunks asked for from the start of the timed loop, how many of those a kept chunk answered, in the
loop, and after the loop's first repetition, followed by the runner's command line.
