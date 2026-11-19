"""What is timed for the torch.compile side: build(model, example_inputs) returns the callable.

Dashboard configuration plus:
- CUDA graphs ("triton.cudagraphs", what mode="reduce-overhead" sets): in the default mode the host was
  the bottleneck (3.8 ms of Python/dispatch per call against 2.5 ms of GPU kernels).
- Inductor freezing: parameters and buffers become constants of the graph, so batch norms fold into the
  convolutions and the channels-last copies of the weights are made once instead of on every call.
  Gives up following an in-place weight update (the weights are baked in at compile time).
- Pointwise cat only for two inputs: a torch.cat of three or more inputs becomes a ConcatKernel
  (producers write straight into slices of the output) instead of one kernel that re-evaluates every
  input under a mask for each output element; the yolo head's cat of three masked sigmoid/exp
  expressions was the slowest non-convolution kernel. Two-input cats (the upsample routes, the tiny
  grid stacks) still fuse.
- inplace_buffers=False: the bias + leaky_relu after each convolution writes a new buffer instead of
  overwriting the convolution's output; the 100 MB first layer ran at 91 us in place, 67 us out of place.
- realize_opcount_threshold=20 (default 30): an expression is written to memory once it has more than
  20 operations instead of being inlined into its users. The yolo head's chain of four in-place slice
  assignments doubles the inlined expression at every step (each element re-evaluates sigmoid/exp
  under different masks); cutting it earlier costs one small kernel and saves the duplicates.
- max_autotune=True: with the weights frozen, Inductor times its own Triton templates against
  cuDNN/cuBLAS for every convolution and matrix product and keeps the faster one (with the bias and
  leaky_relu that follow fused in as an epilogue where a template wins). Costs about 50 s of compile.
- triton.cudagraph_trees=False: the older one-graph-per-compiled-function CUDA graph path; it does
  35 us less bookkeeping per call than the tree manager (host time 0.160 -> 0.125 ms). Gives up the
  tree manager's check on stale outputs: the returned tensors are static buffers that the next call
  overwrites silently (with trees, touching an output of an earlier call raises an error instead).
- guard_filter_fn=torch.compiler.skip_guard_on_all_nn_modules_unsafe: Dynamo's ~4700 guards on the
  attributes of the 225 nn.Modules cost 0.15 ms on every call (6% of it). Gives up noticing a change to
  an nn.Module attribute made after compilation (model.train(), a replaced layer, an edited
  hyperparameter); guards on the input tensor (shape, dtype, device) and on globals are kept, so
  another batch size still recompiles.
"""
import torch
import torch._inductor.config as inductor_config


def build(model, example_inputs):
    inductor_config.freezing = True
    inductor_config.max_pointwise_cat_inputs = 2
    inductor_config.max_complex_pointwise_cat_inputs = 2
    inductor_config.inplace_buffers = False
    inductor_config.realize_opcount_threshold = 20
    inductor_config.max_autotune = True

    def call(*args, **kwargs):
        return model(*args, **kwargs)

    return torch.compile(
        call,
        options={
            "triton.cudagraphs": True,
            "triton.cudagraph_trees": False,
            "guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe,
        },
    )
