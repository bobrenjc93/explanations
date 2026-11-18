// A wrapper around CPython's arena allocator (PyObject_SetArenaAllocator, public API) that counts the small blocks
// asked of it -- the chunks of a thread's stack of frames are 16 KB and up, obmalloc's arenas are 1 MB -- and, when
// told to, keeps the ones that are freed for the next request of the same size instead of unmapping them.
#include <Python.h>

static PyObjectArenaAllocator below;
static long small_allocs = 0, small_frees = 0, reused = 0;
static int keep = 0;
#define SMALL (256 * 1024)
#define SLOTS 8
static void* held[SLOTS];
static size_t held_size[SLOTS];

static void* arena_alloc(void* ctx, size_t size) {
  if (size <= SMALL) {
    small_allocs++;
    for (int i = 0; i < SLOTS; i++) {
      if (held[i] != NULL && held_size[i] == size) {
        void* found = held[i];
        held[i] = NULL;
        reused++;
        return found;
      }
    }
  }
  return below.alloc(below.ctx, size);
}

static void arena_free(void* ctx, void* ptr, size_t size) {
  if (size <= SMALL) {
    small_frees++;
    if (keep) {
      for (int i = 0; i < SLOTS; i++) {
        if (held[i] == NULL) {
          held[i] = ptr;
          held_size[i] = size;
          return;
        }
      }
    }
  }
  below.free(below.ctx, ptr, size);
}

void arena_install(int keep_freed) {
  PyObjectArenaAllocator mine = {NULL, arena_alloc, arena_free};
  keep = keep_freed;
  PyObject_GetArenaAllocator(&below);
  PyObject_SetArenaAllocator(&mine);
}

long arena_small_allocs(void) { return small_allocs; }
long arena_small_frees(void) { return small_frees; }
long arena_reused(void) { return reused; }
