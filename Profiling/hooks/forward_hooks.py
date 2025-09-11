import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def forward_hook_fn(module, input, output):
    print(f"Hook Called on Module: {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}, Output shape: {output[0].shape}")

model = MyModel()

hook_handle = model.linear2.register_forward_hook(forward_hook_fn)

dummy_input = torch.randn(1, 10)
output = model(dummy_input)

hook_handle.remove()