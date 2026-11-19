"""What is timed for the torch.vm side: build(model, example_inputs) returns the callable. Starts as the
performance dashboard's configuration (the default passes, then graphs handed to Inductor)."""
import torch


def build(model, example_inputs):
    def call(*args, **kwargs):
        return model(*args, **kwargs)

    passes = (*torch.vm.DEFAULT_PASSES, torch.vm.passes.Graphs("inductor"))
    return torch.vm(call, passes=passes)
