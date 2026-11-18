// As arena_cache.c (counting only), and for every 16 KB chunk that is asked for, whose frame is the current one:
// the function whose callee did not fit. Printed by arena_report().
#define Py_BUILD_CORE 1
#include <Python.h>
#include <string.h>
#include "internal/pycore_frame.h"

static PyObjectArenaAllocator below;
#define SLOTS 4096
static PyCodeObject* who[SLOTS];
static int line_of[SLOTS];
static long count[SLOTS];
static long left_min[SLOTS], left_max[SLOTS], frame_bytes[SLOTS];
static long total = 0;
// The frames of the thread when a frame of a function named `call` was the first to have a callee that did not fit.
static char chain[16384];
static int chain_done = 0;

static void dump_chain(_PyInterpreterFrame* frame) {
  int at = 0;
  for (_PyInterpreterFrame* f = frame; f != NULL && at < (int)sizeof(chain) - 256; f = f->previous) {
    if (f->owner == FRAME_OWNED_BY_CSTACK) continue;  // an entry frame: on the C stack, not in a chunk
    PyCodeObject* code = f->f_code;
    const char* file = PyUnicode_AsUTF8(code->co_filename);
    const char* base = strrchr(file, '/');
    at += snprintf(chain + at, sizeof(chain) - at, "    %6d  %s (%s)\n", code->co_framesize * 8, PyUnicode_AsUTF8(code->co_qualname), base != NULL ? base + 1 : file);
  }
}

static void* arena_alloc(void* ctx, size_t size) {
  if (size == 16 * 1024) {
    total++;
    PyThreadState* tstate = PyThreadState_GET();
    _PyInterpreterFrame* frame = tstate->cframe->current_frame;
    PyCodeObject* code = frame != NULL ? frame->f_code : NULL;
    int line = frame != NULL && code != NULL ? PyCode_Addr2Line(code, (int)((char*)frame->prev_instr - (char*)_PyCode_CODE(code))) : 0;
    // What the current chunk has left when a frame did not fit in it: the fields of the thread are still the old chunk's.
    long left = (long)((char*)tstate->datastack_limit - (char*)tstate->datastack_top);
    if (!chain_done && code != NULL && strcmp(PyUnicode_AsUTF8(code->co_name), "call") == 0) {
      chain_done = 1;
      dump_chain(frame);
    }
    for (int i = 0; i < SLOTS; i++) {
      if (who[i] == code && line_of[i] == line) {
        count[i]++;
        if (left < left_min[i]) left_min[i] = left;
        if (left > left_max[i]) left_max[i] = left;
        break;
      }
      if (who[i] == NULL && count[i] == 0) {
        who[i] = code; Py_XINCREF(code); line_of[i] = line; count[i] = 1; left_min[i] = left_max[i] = left;
        frame_bytes[i] = code != NULL ? (long)code->co_framesize * 8 : 0;
        break;
      }
    }
  }
  return below.alloc(below.ctx, size);
}

static void arena_free(void* ctx, void* ptr, size_t size) { below.free(below.ctx, ptr, size); }

void arena_install(int unused) {
  PyObjectArenaAllocator mine = {NULL, arena_alloc, arena_free};
  PyObject_GetArenaAllocator(&below);
  PyObject_SetArenaAllocator(&mine);
}

long arena_small_allocs(void) { return total; }
long arena_small_frees(void) { return 0; }
long arena_reused(void) { return 0; }

void arena_report(void) {
  fprintf(stderr, "frames of the thread at the first such request (innermost first, bytes):\n%s", chain);
  fprintf(stderr, "chunks asked for: %ld; by the function that was running:\n", total);
  for (int i = 0; i < SLOTS; i++) {
    if (count[i] > 0) {
      const char* name = who[i] != NULL ? PyUnicode_AsUTF8(who[i]->co_qualname) : "(no frame)";
      const char* file = who[i] != NULL ? PyUnicode_AsUTF8(who[i]->co_filename) : "";
      fprintf(stderr, "  %8ld  %s (%s:%d)  frame %ld bytes; the chunk had %ld to %ld bytes left\n", count[i], name, file, line_of[i], frame_bytes[i], left_min[i], left_max[i]);
    }
  }
}
