"""torch.vm with its default passes, two passes of the user's over the graph of a region, graphs handed to
Inductor and run as CUDA graphs. The passes: max pools of one tensor as a cascade (exact), and convolution,
batch norm, LeakyReLU and residual add as one cuDNN kernel where that is faster.

The kernel: cuDNN's backend (graph) API fuses a convolution with the pointwise operations behind it at run
time, where the convolution PyTorch calls writes its result and a second kernel reads all of it again to add
the bias and apply the activation. It is called through ctypes on the cuDNN library PyTorch has loaded."""
import ctypes, operator, os, threading, time
from typing import List, Optional

import torch

# Values of cuDNN 9's cudnn_graph.h.
_E = {
    "CUDNN_ATTR_CONVOLUTION_COMP_TYPE": 100,
    "CUDNN_ATTR_CONVOLUTION_CONV_MODE": 101,
    "CUDNN_ATTR_CONVOLUTION_DILATIONS": 102,
    "CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES": 103,
    "CUDNN_ATTR_CONVOLUTION_POST_PADDINGS": 104,
    "CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS": 105,
    "CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS": 106,
    "CUDNN_ATTR_ENGINEHEUR_MODE": 200,
    "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH": 201,
    "CUDNN_ATTR_ENGINEHEUR_RESULTS": 202,
    "CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG": 401,
    "CUDNN_ATTR_EXECUTION_PLAN_HANDLE": 400,
    "CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE": 402,
    "CUDNN_ATTR_OPERATIONGRAPH_HANDLE": 800,
    "CUDNN_ATTR_OPERATIONGRAPH_OPS": 801,
    "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA": 700,
    "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA": 701,
    "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC": 702,
    "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W": 703,
    "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X": 704,
    "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y": 705,
    "CUDNN_ATTR_OPERATION_POINTWISE_BDESC": 752,
    "CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR": 750,
    "CUDNN_ATTR_OPERATION_POINTWISE_XDESC": 751,
    "CUDNN_ATTR_OPERATION_POINTWISE_YDESC": 753,
    "CUDNN_ATTR_POINTWISE_MATH_PREC": 1,
    "CUDNN_ATTR_POINTWISE_MODE": 0,
    "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP": 3,
    "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE": 5,
    "CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT": 900,
    "CUDNN_ATTR_TENSOR_DATA_TYPE": 901,
    "CUDNN_ATTR_TENSOR_DIMENSIONS": 902,
    "CUDNN_ATTR_TENSOR_IS_VIRTUAL": 907,
    "CUDNN_ATTR_TENSOR_STRIDES": 903,
    "CUDNN_ATTR_TENSOR_UNIQUE_ID": 906,
    "CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS": 1001,
    "CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS": 1000,
    "CUDNN_ATTR_VARIANT_PACK_WORKSPACE": 1003,
    "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR": 1,
    "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR": 3,
    "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR": 4,
    "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR": 5,
    "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR": 15,
    "CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR": 10,
    "CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR": 13,
    "CUDNN_BACKEND_POINTWISE_DESCRIPTOR": 0,
    "CUDNN_BACKEND_TENSOR_DESCRIPTOR": 17,
    "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR": 16,
    "CUDNN_CROSS_CORRELATION": 1,
    "CUDNN_DATA_BFLOAT16": 9,
    "CUDNN_DATA_FLOAT": 0,
    "CUDNN_DATA_HALF": 2,
    "CUDNN_HEUR_MODE_A": 3,
    "CUDNN_HEUR_MODE_FALLBACK": 2,
    "CUDNN_POINTWISE_ADD": 0,
    "CUDNN_POINTWISE_RELU_FWD": 100,
    "CUDNN_TYPE_BACKEND_DESCRIPTOR": 15,
    "CUDNN_TYPE_BOOLEAN": 2,
    "CUDNN_TYPE_CONVOLUTION_MODE": 7,
    "CUDNN_TYPE_DATA_TYPE": 1,
    "CUDNN_TYPE_DOUBLE": 5,
    "CUDNN_TYPE_FLOAT": 4,
    "CUDNN_TYPE_HANDLE": 0,
    "CUDNN_TYPE_HEUR_MODE": 8,
    "CUDNN_TYPE_INT64": 3,
    "CUDNN_TYPE_POINTWISE_MODE": 14,
    "CUDNN_TYPE_VOID_PTR": 6,
}
_lib = None
_handles = {}
_lock = threading.Lock()
c_void_p, c_int64, c_int, c_double = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_double


class CudnnError(RuntimeError):
    pass


def _load():
    global _lib
    if _lib is None:
        torch.backends.cudnn.version()
        path = next((l.split()[-1] for l in open("/proc/self/maps") if "libcudnn.so" in l), "libcudnn.so.9")
        lib = ctypes.CDLL(path)
        lib.cudnnCreate.argtypes = [ctypes.POINTER(c_void_p)]
        lib.cudnnSetStream.argtypes = [c_void_p, c_void_p]
        lib.cudnnBackendCreateDescriptor.argtypes = [c_int, ctypes.POINTER(c_void_p)]
        lib.cudnnBackendDestroyDescriptor.argtypes = [c_void_p]
        lib.cudnnBackendSetAttribute.argtypes = [c_void_p, c_int, c_int, c_int64, c_void_p]
        lib.cudnnBackendGetAttribute.argtypes = [c_void_p, c_int, c_int, c_int64, ctypes.POINTER(c_int64), c_void_p]
        lib.cudnnBackendFinalize.argtypes = [c_void_p]
        lib.cudnnBackendExecute.argtypes = [c_void_p, c_void_p, c_void_p]
        lib.cudnnGetErrorString.restype = ctypes.c_char_p
        lib.cudnnGetErrorString.argtypes = [c_int]
        _lib = lib
    return _lib


def _check(status, what):
    if status != 0:
        raise CudnnError(f"{what}: {_lib.cudnnGetErrorString(status).decode()} ({status})")


def _handle(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _handles:
        lib = _load()
        handle = c_void_p()
        with torch.cuda.device(index):
            _check(lib.cudnnCreate(ctypes.byref(handle)), "cudnnCreate")
        _handles[index] = handle
    return _handles[index]


class _Desc:
    def __init__(self, kind):
        self.ptr = c_void_p()
        _check(_load().cudnnBackendCreateDescriptor(_E[kind], ctypes.byref(self.ptr)), kind)

    def set(self, attr, kind, values):
        ctype = {"INT64": c_int64, "DOUBLE": c_double, "FLOAT": ctypes.c_float, "BOOLEAN": ctypes.c_bool, "VOID_PTR": c_void_p,
                 "BACKEND_DESCRIPTOR": c_void_p, "HANDLE": c_void_p}.get(kind, c_int)
        values = [v.ptr if isinstance(v, _Desc) else v for v in values]
        array = (ctype * len(values))(*values)
        _check(_lib.cudnnBackendSetAttribute(self.ptr, _E[attr], _E["CUDNN_TYPE_" + kind], len(values), ctypes.cast(array, c_void_p)), attr)
        return self

    def finalize(self):
        _check(_lib.cudnnBackendFinalize(self.ptr), "finalize")
        return self

    def __del__(self):
        if self.ptr and _lib is not None:
            _lib.cudnnBackendDestroyDescriptor(self.ptr)


_DTYPES = {torch.bfloat16: "CUDNN_DATA_BFLOAT16", torch.float16: "CUDNN_DATA_HALF", torch.float32: "CUDNN_DATA_FLOAT"}


def _tensor(uid, dtype, dims, strides, virtual=False):
    desc = _Desc("CUDNN_BACKEND_TENSOR_DESCRIPTOR")
    desc.set("CUDNN_ATTR_TENSOR_DATA_TYPE", "DATA_TYPE", [_E[_DTYPES[dtype]]])
    desc.set("CUDNN_ATTR_TENSOR_DIMENSIONS", "INT64", list(dims))
    desc.set("CUDNN_ATTR_TENSOR_STRIDES", "INT64", list(strides))
    desc.set("CUDNN_ATTR_TENSOR_UNIQUE_ID", "INT64", [uid])
    desc.set("CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT", "INT64", [16])
    if virtual:
        desc.set("CUDNN_ATTR_TENSOR_IS_VIRTUAL", "BOOLEAN", [True])
    return desc.finalize()


def _nhwc(dims):
    n, c, h, w = dims
    return (h * w * c, 1, w * c, c)


class Plan:
    """y = leaky(conv(x, w) + b) [+ z], for one set of shapes, on NHWC memory."""

    def __init__(self, device, dtype, x_dims, w_dims, stride, padding, slope, residual):
        self.handle = _handle(device)
        n, _, h, w = x_dims
        k, _, r, s = w_dims
        y_dims = (n, k, (h + 2 * padding[0] - r) // stride[0] + 1, (w + 2 * padding[1] - s) // stride[1] + 1)
        self.y_dims = y_dims
        X = _tensor(1, dtype, x_dims, _nhwc(x_dims))
        W = _tensor(2, dtype, w_dims, _nhwc(w_dims))
        B = _tensor(3, dtype, (1, k, 1, 1), _nhwc((1, k, 1, 1)))
        Y = _tensor(4, dtype, y_dims, _nhwc(y_dims))
        t1 = _tensor(11, dtype, y_dims, _nhwc(y_dims), True)
        t2 = _tensor(12, dtype, y_dims, _nhwc(y_dims), True)
        conv = _Desc("CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR")
        conv.set("CUDNN_ATTR_CONVOLUTION_COMP_TYPE", "DATA_TYPE", [_E["CUDNN_DATA_FLOAT"]])
        conv.set("CUDNN_ATTR_CONVOLUTION_CONV_MODE", "CONVOLUTION_MODE", [_E["CUDNN_CROSS_CORRELATION"]])
        conv.set("CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS", "INT64", [2])
        conv.set("CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS", "INT64", list(padding))
        conv.set("CUDNN_ATTR_CONVOLUTION_POST_PADDINGS", "INT64", list(padding))
        conv.set("CUDNN_ATTR_CONVOLUTION_DILATIONS", "INT64", [1, 1])
        conv.set("CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES", "INT64", list(stride))
        conv.finalize()
        op_conv = _Desc("CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR")
        op_conv.set("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC", "BACKEND_DESCRIPTOR", [conv])
        op_conv.set("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X", "BACKEND_DESCRIPTOR", [X])
        op_conv.set("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W", "BACKEND_DESCRIPTOR", [W])
        op_conv.set("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y", "BACKEND_DESCRIPTOR", [t1])
        op_conv.set("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA", "DOUBLE", [1.0])
        op_conv.set("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA", "DOUBLE", [0.0])
        op_conv.finalize()

        def pointwise(mode, x, b, y, **attrs):
            pw = _Desc("CUDNN_BACKEND_POINTWISE_DESCRIPTOR")
            pw.set("CUDNN_ATTR_POINTWISE_MODE", "POINTWISE_MODE", [_E[mode]])
            pw.set("CUDNN_ATTR_POINTWISE_MATH_PREC", "DATA_TYPE", [_E["CUDNN_DATA_FLOAT"]])
            for name, value in attrs.items():
                pw.set(name, "DOUBLE", [value])
            pw.finalize()
            op = _Desc("CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR")
            op.set("CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR", "BACKEND_DESCRIPTOR", [pw])
            op.set("CUDNN_ATTR_OPERATION_POINTWISE_XDESC", "BACKEND_DESCRIPTOR", [x])
            if b is not None:
                op.set("CUDNN_ATTR_OPERATION_POINTWISE_BDESC", "BACKEND_DESCRIPTOR", [b])
            op.set("CUDNN_ATTR_OPERATION_POINTWISE_YDESC", "BACKEND_DESCRIPTOR", [y])
            op.finalize()
            return op, pw

        keep = [X, W, B, Y, t1, t2, conv]
        ops = [op_conv]
        self.ids = [1, 2, 3, 4]
        plain = slope == 1.0
        op, pw = pointwise("CUDNN_POINTWISE_ADD", t1, B, Y if plain and not residual else t2)
        ops.append(op); keep.append(pw)
        last = t2
        if not plain:
            if residual:
                t3 = _tensor(13, dtype, y_dims, _nhwc(y_dims), True)
                keep.append(t3)
            op, pw = pointwise("CUDNN_POINTWISE_RELU_FWD", t2, None, t3 if residual else Y,
                               CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP=0.0, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE=slope)
            ops.append(op); keep.append(pw)
            last = t3 if residual else None
        if residual:
            Z = _tensor(5, dtype, y_dims, _nhwc(y_dims))
            keep.append(Z)
            op, pw = pointwise("CUDNN_POINTWISE_ADD", last, Z, Y)
            ops.append(op); keep.append(pw)
            self.ids.append(5)
        graph = _Desc("CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR")
        graph.set("CUDNN_ATTR_OPERATIONGRAPH_HANDLE", "HANDLE", [self.handle])
        graph.set("CUDNN_ATTR_OPERATIONGRAPH_OPS", "BACKEND_DESCRIPTOR", ops)
        graph.finalize()
        self.keep = keep + ops + [graph]
        found = []
        errors = []
        for mode in ("CUDNN_HEUR_MODE_A", "CUDNN_HEUR_MODE_FALLBACK"):
            try:
                heur = _Desc("CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR")
                heur.set("CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH", "BACKEND_DESCRIPTOR", [graph])
                heur.set("CUDNN_ATTR_ENGINEHEUR_MODE", "HEUR_MODE", [_E[mode]])
                heur.finalize()
                count = c_int64(0)
                _check(_lib.cudnnBackendGetAttribute(heur.ptr, _E["CUDNN_ATTR_ENGINEHEUR_RESULTS"], _E["CUDNN_TYPE_BACKEND_DESCRIPTOR"], 0, ctypes.byref(count), None), "count of engine configs")
                configs = [_Desc("CUDNN_BACKEND_ENGINECFG_DESCRIPTOR") for _ in range(count.value)]
                array = (c_void_p * len(configs))(*[c.ptr for c in configs])
                _check(_lib.cudnnBackendGetAttribute(heur.ptr, _E["CUDNN_ATTR_ENGINEHEUR_RESULTS"], _E["CUDNN_TYPE_BACKEND_DESCRIPTOR"], len(configs), ctypes.byref(count), ctypes.cast(array, c_void_p)), "engine configs")
            except CudnnError as e:
                errors.append(str(e))
                continue
            for config in configs[: count.value][:_TRIED]:
                try:
                    plan = _Desc("CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR")
                    plan.set("CUDNN_ATTR_EXECUTION_PLAN_HANDLE", "HANDLE", [self.handle])
                    plan.set("CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG", "BACKEND_DESCRIPTOR", [config])
                    plan.finalize()
                    size, count1 = c_int64(0), c_int64(0)
                    _check(_lib.cudnnBackendGetAttribute(plan.ptr, _E["CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE"], _E["CUDNN_TYPE_INT64"], 1, ctypes.byref(count1), ctypes.cast(ctypes.byref(size), c_void_p)), "workspace size")
                except CudnnError as e:
                    errors.append(str(e))
                    continue
                found.append((plan, config, max(size.value, 16)))
            if found:
                break
        if not found:
            raise CudnnError("no engine for the fused convolution: " + "; ".join(errors[:3]))
        self.plan, self.config, size = found[0]
        self.workspace = torch.empty(size, dtype=torch.uint8, device=device)
        if len(found) > 1 and not torch.cuda.is_current_stream_capturing():
            # The heuristic's first engine is often not the fastest for a layer: the first few are
            # timed on tensors of the layer's shapes, once, and the fastest is kept (as cudnn.benchmark does).
            x = torch.randn(x_dims, device=device).to(dtype).contiguous(memory_format=torch.channels_last)
            wt = (torch.randn(w_dims, device=device) * 0.05).to(dtype).contiguous(memory_format=torch.channels_last)
            bias = torch.randn((1, k, 1, 1), device=device).to(dtype)
            y = torch.empty(y_dims, dtype=dtype, device=device, memory_format=torch.channels_last)
            z = torch.randn(y_dims, device=device).to(dtype).contiguous(memory_format=torch.channels_last) if residual else None
            best = None
            for plan, config, size in found:
                self.plan, self.config = plan, config
                self.workspace = torch.empty(size, dtype=torch.uint8, device=device)
                try:
                    taken = _gpu_time(lambda: self.run(x, wt, bias, y, z))
                except Exception:
                    continue
                if best is None or taken < best[0]:
                    best = (taken, plan, config, self.workspace)
            if best is None:
                raise CudnnError("no engine for the fused convolution ran")
            _, self.plan, self.config, self.workspace = best

    def run(self, x, w, b, y, z=None):
        """All of them NHWC memory of the plan's shapes; launches on the current stream."""
        pointers = [_address(t) for t in (x, w, b, y)] + ([_address(z)] if z is not None else [])
        pack = _Desc("CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR")
        pack.set("CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS", "INT64", self.ids)
        pack.set("CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS", "VOID_PTR", pointers)
        pack.set("CUDNN_ATTR_VARIANT_PACK_WORKSPACE", "VOID_PTR", [_address(self.workspace)])
        pack.finalize()
        with _lock:
            _check(_lib.cudnnSetStream(self.handle, c_void_p(torch.cuda.current_stream(x.device).cuda_stream)), "cudnnSetStream")
            _check(_lib.cudnnBackendExecute(self.handle, self.plan.ptr, pack.ptr), "execute")


# ----------------------------------------------------------------------------------------------------
# The kernel as an operator, and the pass over the graph of a region that puts it in
# ----------------------------------------------------------------------------------------------------

_plans = {}
_CL = torch.channels_last
# How many of the engines cuDNN offers for a layer are timed.
_TRIED = 10


def _out_dims(x_shape, w_shape, stride, padding):
    n, _, h, w = x_shape
    k, _, r, s = w_shape
    return (n, k, (h + 2 * padding[0] - r) // stride[0] + 1, (w + 2 * padding[1] - s) // stride[1] + 1)


def _address(t):
    """Where a tensor's first element is. Not data_ptr() where the runtime has another way: that is a
    pointer one could write through, and asking for it counts as a write to a weight torch.vm watches."""
    base = getattr(torch._C, "_data_address", None)
    return t.data_ptr() if base is None else base(t) + t.storage_offset() * t.element_size()


def _nhwc_memory(t):
    if not t.is_contiguous(memory_format=_CL) or _address(t) % 16:
        t = t.contiguous(memory_format=_CL)
        if _address(t) % 16:
            t = t.clone(memory_format=_CL)
    return t


@torch.library.custom_op("hillvm::conv_bias_leaky", mutates_args=())
def conv_bias_leaky(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, z: Optional[torch.Tensor], stride: List[int], padding: List[int], slope: float) -> torch.Tensor:
    """leaky_relu(conv2d(x, w) + b, slope) [+ z] as one cuDNN kernel; b is (1, K, 1, 1)."""
    key = (x.device, x.dtype, tuple(x.shape), tuple(w.shape), tuple(stride), tuple(padding), float(slope), z is not None)
    plan = _plans.get(key)
    if plan is None:
        plan = _plans[key] = Plan(x.device, x.dtype, key[2], key[3], key[4], key[5], key[6], key[7])
    y = torch.empty(plan.y_dims, dtype=x.dtype, device=x.device, memory_format=_CL)
    plan.run(_nhwc_memory(x), _nhwc_memory(w), _nhwc_memory(b), y, None if z is None else _nhwc_memory(z))
    return y


@conv_bias_leaky.register_fake
def _(x, w, b, z, stride, padding, slope):
    return torch.empty(_out_dims(x.shape, w.shape, stride, padding), dtype=x.dtype, device=x.device, memory_format=_CL)


def _gpu_time(body, reps=16):
    """The time the GPU takes for body(), without what the CPU takes to launch it: recorded, replayed."""
    body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(torch.cuda.Stream()):
        graph.capture_begin()
        try:
            kept = [body() for _ in range(reps)]
        finally:
            graph.capture_end()
    best = float("inf")
    for _ in range(12):
        torch.cuda.synchronize()
        start = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    del kept
    return best / reps


_choices = {}


def fused_is_faster(device, dtype, x_shape, w_shape, stride, padding, slope, residual):
    """Whether the one kernel takes less of the GPU than the convolution and the pass behind it that
    adds the bias and applies the activation (timed once per kind of layer, like cudnn.benchmark)."""
    key = (device, dtype, tuple(x_shape), tuple(w_shape), tuple(stride), tuple(padding), float(slope), residual)
    if key not in _choices:
        try:
            with torch.no_grad():
                x = torch.randn(x_shape, device=device).to(dtype).contiguous(memory_format=_CL)
                w = (torch.randn(w_shape, device=device) * 0.05).to(dtype).contiguous(memory_format=_CL)
                b = torch.randn((1, w_shape[0], 1, 1), device=device).to(dtype)
                z = torch.randn(_out_dims(x_shape, w_shape, stride, padding), device=device).to(dtype).contiguous(memory_format=_CL) if residual else None
                fused = _gpu_time(lambda: conv_bias_leaky(x, w, b, z, list(stride), list(padding), slope))

                def separate():
                    c = torch.nn.functional.conv2d(x, w, None, stride, padding)
                    # One pass over the result, as the kernel Inductor makes of bias, activation and add is.
                    return torch.add(c, z) if residual else torch.nn.functional.leaky_relu(c, slope)

                apart = _gpu_time(separate)
                # Left apart, a layer between fused ones also has its tensors copied from one layout to the
                # other (the kernel is on NHWC memory), so it has to be clearly faster to be left.
                _choices[key] = (fused < 1.10 * apart, f"fused {fused * 1e6:.1f} us, apart {apart * 1e6:.1f} us")
        except Exception as e:
            _choices[key] = (False, f"{type(e).__name__}: {e}"[:200])
    return _choices[key][0]


def fuse_convolutions(gm, input_types):
    """conv2d -> batch norm (inference) -> leaky_relu [-> + shortcut] as one cuDNN kernel, where that is
    faster. The batch norm goes into the weights and the bias (computed in float32 from the module's
    tensors, in the graph: Inductor folds it where they are built in)."""
    aten = torch.ops.aten
    nodes = list(gm.graph.nodes)
    vals = [n.meta.get("val") for n in nodes if isinstance(n.meta.get("val"), torch.Tensor)]
    fake_mode = getattr(vals[0], "fake_mode", None) if vals else None
    if fake_mode is None:
        return gm

    def static(node):
        val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
        return isinstance(val, torch.Tensor) and all(type(s) is int for s in val.shape)

    def emit(target, *args, **kwargs):
        node = gm.graph.call_function(target, args, kwargs)
        unwrap = lambda a: a.meta["val"] if isinstance(a, torch.fx.Node) else a
        with fake_mode:
            node.meta["val"] = target(*[unwrap(a) for a in args], **kwargs)
        return node

    def nhwc(node):
        # The kernel is on NHWC memory. What another fused kernel gives is; anything else is asked for
        # in that layout here, in the graph, so that Inductor writes it so where it computes it,
        # where the operator would have to copy it.
        if node.op == "call_function" and node.target is conv_bias_leaky._opoverload:
            return node
        return emit(aten.clone.default, node, memory_format=_CL)

    done = 0
    for conv in nodes:
        if conv.op != "call_function" or conv.target is not aten.convolution.default or len(conv.args) != 9:
            continue
        x, w, conv_bias, stride, padding, dilation, transposed, output_padding, groups = conv.args
        if transposed or groups != 1 or list(dilation) != [1, 1] or len(stride) != 2 or len(padding) != 2:
            continue
        users = list(conv.users)
        if len(users) != 1 or users[0].target is not aten.native_batch_norm.default or users[0].args[0] is not conv:
            # A convolution with a bias and no batch norm behind it: the bias in the kernel (slope 1).
            if not isinstance(conv_bias, torch.fx.Node) or not (static(x) and static(w) and static(conv)):
                continue
            xv, wv = x.meta["val"], w.meta["val"]
            if xv.dim() != 4 or xv.device.type != "cuda" or xv.dtype not in _DTYPES:
                continue
            # cuDNN's tensor-core kernels are for channels in multiples of 8: other counts (255 outputs
            # of a detection head) take a kernel several times slower. Zero filters are added up to
            # the next multiple, and what they give is cut off again.
            filters = wv.shape[0]
            more = -filters % 8
            shape = (filters + more, *wv.shape[1:])
            if not fused_is_faster(xv.device, xv.dtype, tuple(xv.shape), shape, tuple(stride), tuple(padding), 1.0, False):
                continue
            with gm.graph.inserting_before(conv):
                weight, bias = w, conv_bias
                if more:
                    weight = emit(aten.constant_pad_nd.default, weight, [0, 0, 0, 0, 0, 0, 0, more], 0.0)
                    bias = emit(aten.constant_pad_nd.default, bias, [0, more], 0.0)
                weight = emit(aten.clone.default, weight, memory_format=_CL)
                bias = emit(aten.reshape.default, bias, [1, -1, 1, 1])
                fused = emit(conv_bias_leaky._opoverload, nhwc(x), weight, bias, None, list(stride), list(padding), 1.0)
                if more:
                    fused = emit(aten.slice.Tensor, fused, 1, 0, filters)
            conv.replace_all_uses_with(fused)
            gm.graph.erase_node(conv)
            done += 1
            continue
        bn = users[0]
        if len(bn.args) != 8 or bn.args[5] is not False or any(not isinstance(a, torch.fx.Node) for a in bn.args[1:5]):
            continue
        items = list(bn.users)
        first = [n for n in items if n.target is operator.getitem and n.args[1] == 0]
        if len(first) != 1 or any(n.users for n in items if n is not first[0]):
            continue
        acts = list(first[0].users)
        if len(acts) != 1 or acts[0].target not in (aten.leaky_relu_.default, aten.leaky_relu.default):
            continue
        act = acts[0]
        slope = float(act.args[1]) if len(act.args) > 1 else 0.01
        if not (static(x) and static(w) and static(act)) or x.meta["val"].dim() != 4:
            continue
        if x.meta["val"].device.type != "cuda" or x.meta["val"].dtype not in _DTYPES:
            continue
        last, shortcut = act, None
        adds = list(act.users)
        if len(adds) == 1 and adds[0].target is aten.add.Tensor and len(adds[0].args) == 2 and not adds[0].kwargs:
            other = adds[0].args[1] if adds[0].args[0] is act else adds[0].args[0]
            if other is not act and static(other) and other.meta["val"].shape == act.meta["val"].shape and other.meta["val"].dtype == act.meta["val"].dtype:
                last, shortcut = adds[0], other
        xv, wv = x.meta["val"], w.meta["val"]
        # Input channels in multiples of 4, as cuDNN's kernels want them: an image's 3 channels get a
        # channel of zeros, and the filters zeros for it (a third faster for the first layer).
        wider = -xv.shape[1] % 4
        x_shape = (xv.shape[0], xv.shape[1] + wider, *xv.shape[2:])
        w_shape = (wv.shape[0], wv.shape[1] + wider, *wv.shape[2:])
        if not fused_is_faster(xv.device, xv.dtype, x_shape, w_shape, tuple(stride), tuple(padding), slope, shortcut is not None):
            if shortcut is None or not fused_is_faster(xv.device, xv.dtype, x_shape, w_shape, tuple(stride), tuple(padding), slope, False):
                continue
            last, shortcut = act, None
        _, gamma, beta, mean, var, _, _, eps = bn.args

        f32 = lambda t: emit(aten._to_copy.default, t, dtype=torch.float32)
        with gm.graph.inserting_before(last):
            scale = emit(aten.mul.Tensor, f32(gamma), emit(aten.rsqrt.default, emit(aten.add.Tensor, f32(var), float(eps))))
            weight = emit(aten.mul.Tensor, f32(w), emit(aten.reshape.default, scale, [-1, 1, 1, 1]))
            source = x
            if wider:
                weight = emit(aten.constant_pad_nd.default, weight, [0, 0, 0, 0, 0, wider], 0.0)
                source = emit(aten.constant_pad_nd.default, x, [0, 0, 0, 0, 0, wider], 0.0)
            weight = emit(aten.clone.default, emit(aten._to_copy.default, weight, dtype=wv.dtype), memory_format=_CL)
            centre = f32(mean) if conv_bias is None else emit(aten.sub.Tensor, f32(mean), f32(conv_bias))
            bias = emit(aten.sub.Tensor, f32(beta), emit(aten.mul.Tensor, centre, scale))
            bias = emit(aten._to_copy.default, emit(aten.reshape.default, bias, [1, -1, 1, 1]), dtype=wv.dtype)
            fused = emit(conv_bias_leaky._opoverload, nhwc(source), weight, bias, shortcut and nhwc(shortcut), list(stride), list(padding), slope)
        last.replace_all_uses_with(fused)
        for dead in ([last] if last is not act else []) + [act, first[0]] + [n for n in items if n is not first[0]] + [bn, conv]:
            gm.graph.erase_node(dead)
        done += 1
    if done:
        gm.graph.lint()
    if os.environ.get("HILL_DEBUG"):
        print(f"fuse_convolutions: {done} fused", flush=True)
        for key, (choice, note) in _choices.items():
            print("  ", key[2], key[3], key[4], "residual" if key[7] else "", choice, note if isinstance(note, str) else f"{note * 1e6:.1f} us", flush=True)
    return gm


@torch.library.custom_op("hillvm::concat", mutates_args=())
def concat(tensors: List[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.cat(tensors, dim)


@concat.register_fake
def _(tensors, dim):
    return torch.cat(tensors, dim)


def results_apart(gm, input_types):
    """A concatenation that is a result of the graph, as a kernel of its own. Inductor computes a
    concatenation inside the kernel that computes its parts, and every element of the result then goes
    through the arithmetic of every part (for three detection heads with their slice updates: 126 loads,
    39 sigmoids and 18 exponentials per element). Apart, each part is computed for its own elements only,
    and the concatenation is a copy."""
    results = set()
    for node in gm.graph.nodes:
        if node.op == "output":
            results = {n for n in node.all_input_nodes}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.cat.default and node in results:
            tensors = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
            if len(tensors) >= 2 and all(isinstance(t, torch.fx.Node) for t in tensors) and isinstance(dim, int):
                node.target = concat._opoverload
                node.args, node.kwargs = (list(tensors), dim), {}
    return gm


def cascade_pools(gm, input_types):
    """Max pools of one tensor with stride 1 and windows of 5, 9 and 13 (a spatial pyramid) as a cascade:
    with stride 1 and padding of half the window, a window of a + b - 1 over x is a window of b over the
    result of a window of a over x. The maximum is exact, so the results are the same bits; the largest
    window then reads 25 values per output where it read 169."""
    pool = torch.ops.aten.max_pool2d_with_indices.default
    order = {node: i for i, node in enumerate(gm.graph.nodes)}

    def window(node):
        # (k, values) for a pool with stride 1, padding k // 2, no dilation, whose indices nobody reads.
        args = list(node.args) + [node.kwargs.get(name) for name in ("dilation", "ceil_mode")][len(node.args) - 4 :]
        if len(node.args) < 4 or len(args) != 6:
            return None
        _, kernel, stride, padding, dilation, ceil_mode = args
        k = kernel[0] if isinstance(kernel, (list, tuple)) else kernel
        if not isinstance(k, int) or k % 2 != 1 or list(kernel) != [k, k] or list(stride) != [1, 1]:
            return None
        if list(padding) != [k // 2, k // 2] or dilation not in (None, 1, [1, 1], (1, 1)) or ceil_mode not in (None, False):
            return None
        values = None
        for user in node.users:
            if user.op != "call_function" or user.target.__name__ != "getitem":
                return None
            if user.args[1] == 0:
                values = user
            elif user.users:
                return None
        return (k, values) if values is not None else None

    by_input = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is pool and isinstance(node.args[0], torch.fx.Node):
            found = window(node)
            if found is not None:
                by_input.setdefault(node.args[0], []).append((found[0], node, found[1]))
    for pools in by_input.values():
        pools.sort(key=lambda entry: entry[0])
        k0, _, values0 = pools[0]
        for k, node, values in pools[1:]:
            if k > k0 and order[values0] < order[node]:
                step = k - k0 + 1
                node.args = (values0, [step, step], [1, 1], [step // 2, step // 2], *node.args[4:])
            k0, values0 = k, values
        # And each window as two: down the columns, then along the rows (the maximum over a rectangle
        # is the maximum of the maxima of its rows), 2k values per output where it was k * k.
        for _, node, values in pools:
            step = node.args[1][0]
            if step > 1:
                with gm.graph.inserting_before(node):
                    first = gm.graph.call_function(pool, (node.args[0], [step, 1], [1, 1], [step // 2, 0]))
                    first.meta = dict(node.meta)
                    rows = gm.graph.call_function(operator.getitem, (first, 0))
                    rows.meta = dict(values.meta)
                node.args = (rows, [1, step], [1, 1], [0, step // 2], *node.args[4:])
    return gm


def build(model, example_inputs):
    def call(*args, **kwargs):
        return model(*args, **kwargs)

    graphs = torch.vm.passes.Graphs("inductor", cuda_graphs=True)
    passes = (*torch.vm.DEFAULT_PASSES, cascade_pools, fuse_convolutions, results_apart, graphs)
    return torch.vm(call, passes=passes)
