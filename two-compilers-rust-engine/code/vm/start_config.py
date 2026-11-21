"""What is timed for the torch.vm side: build(model, example_inputs) returns the callable. Starts as the
performance dashboard's configuration: the default passes, then the graphs the engine finds, compiled by Inductor."""
import torch


def build(model, example_inputs):
    def call(*args, **kwargs):
        return model(*args, **kwargs)

    passes = (*torch.vm.DEFAULT_PASSES, torch.vm.passes.Graphs(), torch.vm.passes.Inductor())
    return torch.vm(call, passes=passes)
