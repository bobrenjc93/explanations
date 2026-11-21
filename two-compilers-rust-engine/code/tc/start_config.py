"""What is timed for the torch.compile side: build(model, example_inputs) returns the callable. Starts as the
performance dashboard's configuration: its runner compiles a function that calls the model, with the
inductor backend in its default mode and without CUDA graphs."""
import torch


def build(model, example_inputs):
    def call(*args, **kwargs):
        return model(*args, **kwargs)

    return torch.compile(call)
