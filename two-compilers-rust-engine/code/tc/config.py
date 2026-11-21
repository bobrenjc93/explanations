"""torch.compile side: the inductor backend with CUDA graphs.

- CUDA graphs ("triton.cudagraphs", what mode="reduce-overhead" sets): one replay instead of ~330 kernel launches.
- Outputs of a replay live in the graph's memory pool and are overwritten by the next call, so the callable
  hands back copies: a caller may keep, compare or modify results of earlier calls as with eager.
- max_complex_pointwise_cat_inputs=0: a torch.cat whose inputs are long pointwise expressions is not fused into one
  kernel with masked loads (the YOLO head became one kernel evaluating every branch for every element); each
  input writes its slice of the result instead.
- use_recursive_dict_tags_for_guards: Dynamo skips the guards under an nn.Module whose dicts are unchanged.
- skip_nnmodule_hook_guards=False: hooks registered on a module after compilation are followed, as eager does
  (off by default in torch.compile; with the dict tags above the extra guards cost ~25 us a call).
- automatic_dynamic_shapes=False (what dynamic=False sets, without its per-call config patch): an input of
  another shape gets its own static graph. With the default, the recompile for a new
  batch or image size is a dynamic-shape graph that afterwards also serves the timed shape, at half the speed."""
import torch

OPTIONS = {
    "triton.cudagraphs": True,
    "max_complex_pointwise_cat_inputs": 0,
}


def _own(out):
    if isinstance(out, torch.Tensor):
        return out.clone()
    if isinstance(out, (list, tuple)):
        return type(out)(_own(o) for o in out)
    if isinstance(out, dict):
        return {k: _own(v) for k, v in out.items()}
    return out


def build(model, example_inputs):
    torch._dynamo.config.use_recursive_dict_tags_for_guards = True
    torch._dynamo.config.skip_nnmodule_hook_guards = False
    torch._dynamo.config.automatic_dynamic_shapes = False

    def call(*args, **kwargs):
        return model(*args, **kwargs)

    compiled = torch.compile(call, options=OPTIONS)

    def run(*args, **kwargs):
        return _own(compiled(*args, **kwargs))

    return run
