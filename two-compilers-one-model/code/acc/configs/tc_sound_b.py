"""torch.compile with the final source of the tc side and every option of its final configuration that keeps
following what eager does: no freezing, no guard filter, CUDA graphs with their default bookkeeping."""
import torch
import torch._inductor.config as inductor_config


def build(model, example_inputs):
    inductor_config.max_pointwise_cat_inputs = 2
    inductor_config.max_complex_pointwise_cat_inputs = 2
    inductor_config.inplace_buffers = False
    inductor_config.realize_opcount_threshold = 20

    def call(*args, **kwargs):
        return model(*args, **kwargs)

    return torch.compile(call, options={"triton.cudagraphs": True})
