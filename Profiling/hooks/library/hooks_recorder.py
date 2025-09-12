import os
from datetime import datetime


import torch

class hooks_recorder:
    def __init__(self, rank: int, model: torch.nn.Module):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.forward_tensor_directory =  f"/home/vpolamre/nemo_scractch_space/tensors/forward/rank{rank}_{timestamp}/"
        self.backward_tensor_directory =  f"/home/vpolamre/nemo_scractch_space/tensors/backward/rank{rank}_{timestamp}/"
        self.layer_names_to_hook = []
        self.forward_hook_handles = []
        self.backward_hook_handles = []

        os.makedirs(self.forward_tensor_directory, exist_ok=True)
        os.makedirs(self.backward_tensor_directory, exist_ok=True)
        self._select_layers(model)
        self._add_hooks(model)

    def cleanup(self):
        for handle in self.forward_hook_handles:
            handle.remove()
            
        for handle in self.backward_hook_handles:
            handle.remove()


    def _save_forward_hook(self, layer_name: str):
        def hook(module, input, output):
            output_filepath = os.path.join(self.forward_tensor_directory, f"{layer_name}.output.pt")
            if isinstance(output, torch.Tensor):
                torch.save(output.detach().cpu(), output_filepath)

            if isinstance(input, (tuple, list)):
                for idx, tensor in enumerate(input):
                    if isinstance(tensor, torch.Tensor):
                        input_filepath = os.path.join(self.forward_tensor_directory, f"{layer_name}.input{idx}.pt")
                        torch.save(tensor.detach().cpu(), input_filepath)
            elif isinstance(input, torch.Tensor):
                input_filepath = os.path.join(self.forward_tensor_directory, f"{layer_name}.input.pt")
                torch.save(input.detach().cpu(), input_filepath)
        
        return hook
    
    def _save_backward_hook(self, layer_name: str):
        def hook(module, grad_input, grad_output):
            for idx, grad in enumerate(grad_input):
                if isinstance(grad, torch.Tensor):
                    grad_input_filepath = os.path.join(self.backward_tensor_directory, f"{layer_name}.grad_input{idx}.pt")
                    torch.save(grad.detach().cpu(), grad_input_filepath)

            for idx, grad in enumerate(grad_output):
                if isinstance(grad, torch.Tensor):
                    grad_output_filepath = os.path.join(self.backward_tensor_directory, f"{layer_name}.grad_output{idx}.pt")
                    torch.save(grad.detach().cpu(), grad_output_filepath)

        return hook


    def _select_layers(self, model):
        for layer_name, module in model.named_modules():
                self.layer_names_to_hook.append(layer_name)

    def _add_hooks(self, model):
        for layer_name, module in model.named_modules():
            if layer_name in self.layer_names_to_hook:
                forward_handle = module.register_forward_hook(self._save_forward_hook(layer_name))
                backward_handle = module.register_full_backward_hook(self._save_backward_hook(layer_name))
                self.forward_hook_handles.append(forward_handle)
                self.backward_hook_handles.append(backward_handle)

if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer
    model_name = "/model"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    hookRec = hooks_recorder(rank=0, model=model)

    inputs = tokenizer("Hello, this is Alice in the wonderland!!", return_tensors="pt")

    print(f"Runnign model...", flush=True)
    with torch.no_grad():
        model(**inputs)

    hookRec.cleanup()