import torch


def build(model, example_inputs):
    import torch._inductor.config as inductor_config

    # Inductor tries more launch configurations for its pointwise kernels (the batch norm + leaky relu kernels here).
    inductor_config.max_autotune_pointwise = True

    def call(*args, **kwargs):
        return model(*args, **kwargs)

    P = torch.vm.passes
    # The safe configuration: results are copied out, weights laid out on every call; every write is followed.
    behind = P.Inductor(cuda_graphs=True)
    passes = (*torch.vm.DEFAULT_PASSES, P.Graphs(), P.SplitPools(), behind)
    return torch.vm(call, passes=passes)
