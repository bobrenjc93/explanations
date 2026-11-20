"""What is timed for the torch.compile side: build(model, example_inputs) returns the callable.

Second half, the frozen configuration made sound. Dashboard configuration plus:
- CUDA graphs WITHOUT cudagraph trees ("triton.cudagraph_trees": False): the older one-graph-per-function
  path, less bookkeeping per call. GIVEN UP: the trees' check on stale outputs; the returned tensors are
  static buffers that the next call overwrites silently.
- Inductor freezing: parameters and buffers become constants of the graph (batch norms fold into the
  convolutions, channels-last weights are made once). With the source change of the second half, Dynamo
  guards the version counter of every tensor that freezing baked in, so an in-place update of a weight
  recompiles and re-freezes instead of being ignored.
- guard_filter_fn=keep_weight_and_mode_guards (below): of Dynamo's ~5000 guards on nn.Module attributes
  (0.2 ms per call) it keeps those on tensors (identity, version counter, metadata) and on `.training`,
  and drops the rest. So in-place weight updates, replaced parameters, model.train()/eval(), other batch
  and image sizes are followed. GIVEN UP: noticing an edit of a non-tensor attribute of a module after
  compilation (a layer's stride or padding, a hook, a replaced forward).
- recompile_limit=32 (default 8): each weight update costs one recompilation; with the default the
  ninth one would send the function back to eager for good.
- automatic_dynamic_shapes=False: every input shape gets its own static graph. With the default, once
  the function has seen several shapes, the next recompilation (a weight update, a replaced parameter)
  becomes a dynamic-shape graph, which ran at 1476 img/s instead of ~3640.
- max_autotune: Triton templates against cuDNN/cuBLAS per convolution, bias + leaky_relu fused in.
- Pointwise cat only for two inputs; inplace_buffers=False (see the journal).
- realize_opcount_threshold=20 (fine without trees; with trees it makes a re-recording at an unchanged
  shape fail in the trees' pool check).
Still not followed: a write through `.data` (it has its own version counter, as for autograd).
work/config_best_3973.py is the fastest variant (all module guards dropped, no trees; gives up more).
"""
import torch
import torch._inductor.config as inductor_config


def keep_weight_and_mode_guards(entries):
    keep = []
    for entry in entries:
        if not entry.orig_guard.source.is_unspecialized_nn_module():
            keep.append(True)
        else:
            keep.append(
                entry.guard_type in ("ID_MATCH", "FROZEN_TENSOR_VERSION", "TENSOR_MATCH")
                or entry.name.endswith(".training")
            )
    return keep


def build(model, example_inputs):
    torch._dynamo.config.recompile_limit = 32
    torch._dynamo.config.automatic_dynamic_shapes = False
    inductor_config.freezing = True
    inductor_config.max_pointwise_cat_inputs = 2
    inductor_config.max_complex_pointwise_cat_inputs = 2
    inductor_config.inplace_buffers = False
    inductor_config.max_autotune = True
    inductor_config.realize_opcount_threshold = 20

    def call(*args, **kwargs):
        return model(*args, **kwargs)

    return torch.compile(
        call,
        options={
            "triton.cudagraphs": True,
            "triton.cudagraph_trees": False,
            "guard_filter_fn": keep_weight_and_mode_guards,
        },
    )
