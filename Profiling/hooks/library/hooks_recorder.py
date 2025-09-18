import os
from datetime import datetime


import torch

class hooks_recorder:
    def __init__(self, rank: int, model: torch.nn.Module, layer_names=[], repro_mode=False, save_activations=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.repro_save_directory =  f"/home/vpolamre/nemo_scractch_space/tensors/repro/rank{rank}_{timestamp}/"
        self.forward_tensor_directory =  f"/home/vpolamre/nemo_scractch_space/tensors/forward/rank{rank}_{timestamp}/"
        self.backward_tensor_directory =  f"/home/vpolamre/nemo_scractch_space/tensors/backward/rank{rank}_{timestamp}/"
        self.layer_names_to_hook = []
        self.forward_hook_handles = []
        self.backward_hook_handles = []
        self.activations = {}
        self.model = model
        self.repro_mode= repro_mode
        self.save_activations = save_activations

        os.makedirs(self.forward_tensor_directory, exist_ok=True)
        os.makedirs(self.backward_tensor_directory, exist_ok=True)
        self._select_layers(model, layer_names)
        self._add_hooks(model)

    def check_model_for_nan(self, model):
        for layer_name, module in model.named_modules():
            
            # Check for weight
            if hasattr(module, "weight") and module.weight is not None and module.weight.grad is not None:
                if torch.isnan(module.weight.grad).any():
                    print(f"  ⚠️ NaNs in weight gradients of {layer_name}")

            # Check for bias
            if hasattr(module, "bias") and module.bias is not None and module.bias.grad is not None:
                if torch.isnan(module.bias.grad).any():
                    print(f"  ⚠️ NaNs in bias gradients of {layer_name}")

            # Check for activation gradients
            # if layer_name in self.activations and hasattr(self.activations[layer_name], "grad") and self.activations[layer_name].grad is not None:
            #     if torch.isnan(self.activations[layer_name].grad.detach().cpu()).any():
            #         print("  ⚠️ NaNs in activation gradients!")

    def check_grad_for_nan(self, output, layer_name) -> bool:
        (grads,) = torch.autograd.grad((output,), hook_rec.model.get_submodule(layer_name).parameters())
        if torch.isnan(grads).any():
            return False

        return True
    
    def save_loss_for_repro(self, output):
        loss_save_repro_path = os.path.join(self.repro_save_directory, f"loss.pt")
        torch.save(output.detach().cpu(), loss_save_repro_path)

    def cleanup(self):
        for handle in self.forward_hook_handles:
            handle.remove()
            
        for handle in self.backward_hook_handles:
            handle.remove()

    def _chek_tensor_for_nan(self, data_Tensor, layer_name="") -> bool:
        if isinstance(data_Tensor, (tuple, list)):
            for data in data_Tensor:
                try:
                    if isinstance(data, torch.Tensor) and data.dtype.is_floating_point:
                        # Move to CPU before checking for NaNs to avoid CUDA faults
                        if torch.isnan(data.detach().cpu()).any():
                            return True
                except Exception as e:
                    print(f"⚠️ Error while checking tensor: {e}")
        else:
            if isinstance(data_Tensor, torch.Tensor) and data_Tensor.dtype.is_floating_point:
                if torch.isnan(data_Tensor.detach().cpu()).any():
                    return True
        
        return False


    def _save_forward_hook(self, layer_name: str):
        def hook(module, input, output):

            if not self._chek_tensor_for_nan(input, layer_name):
                print(f"  ⚠️ NaNs in Input for {layer_name}")
                return

            if not self._chek_tensor_for_nan(output, layer_name):
                print(f"  ⚠️ NaNs in Output for {layer_name}")
                return

            if self.repro_mode:
                dir_save_layer_repro_dir = os.path.join(self.repro_save_directory, f"{layer_name}")
                os.makedirs(, exist_ok=True)
                dir_save_layer_state_path = os.path.join(dir_save_layer_repro_dir, f"state.pt")
                
                layer = hook_rec.model.get_submodule(layer_name)
                torch.save({
                    "layer_state_dict": layer.state_dict(),
                    "input_tensor": input[0].detach().clone()
                }, dir_save_layer_state_path)

                
                dir_save_layer_config_path = os.path.join(dir_save_layer_repro_dir, f"config.pt")
                layer_config = {
                    "type": type(layer).__name__,
                    "in_features": getattr(layer, "in_features", None),
                    "out_features": getattr(layer, "out_features", None),
                    "bias": layer.bias is not None if hasattr(layer, "bias") else None
                }
                # Save configuration
                torch.save(layer_config, dir_save_layer_config_path)


            if self.save_activations:
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
            return
            for idx, grad in enumerate(grad_input):
                if isinstance(grad, torch.Tensor):
                    grad_input_filepath = os.path.join(self.backward_tensor_directory, f"{layer_name}.grad_input{idx}.pt")
                    torch.save(grad.detach().cpu(), grad_input_filepath)

            for idx, grad in enumerate(grad_output):
                if isinstance(grad, torch.Tensor):
                    grad_output_filepath = os.path.join(self.backward_tensor_directory, f"{layer_name}.grad_output{idx}.pt")
                    torch.save(grad.detach().cpu(), grad_output_filepath)

        return hook


    def _select_layers(self, model, layer_names=[]):
        for layer_name, module in model.named_modules():
            if len(layer_names) == 0 or layer_name in layer_names:
                self.layer_names_to_hook.append(layer_name)

    def _add_hooks(self, model):
        for layer_name, module in model.named_modules():
            if layer_name in self.layer_names_to_hook:
                forward_handle = module.register_forward_hook(self._save_forward_hook(layer_name))
                # backward_handle = module.register_full_backward_hook(self._save_backward_hook(layer_name))
                self.forward_hook_handles.append(forward_handle)
                # self.backward_hook_handles.append(backward_handle)

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