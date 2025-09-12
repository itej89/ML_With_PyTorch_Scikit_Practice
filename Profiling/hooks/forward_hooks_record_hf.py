from transformers import AutoModel, AutoTokenizer
import torch

import os
import collections

activations= collections.defaultdict(list)
tensor_directory = None

def save_activation_hook(name:str):

    def hook(module, input, output):
        activations[name].append(output.detach().cpu())
        
        output_filepath = os.path.join(tensor_directory, f"{name}.output.pt")
        torch.save(output.detach().cpu(), output_filepath)

        for idx, tensor in enumerate(input):
            input_filepath = os.path.join(tensor_directory, f"{name}.input{idx}.pt")
            torch.save(tensor.detach().cpu(), input_filepath)
    
    return hook



model_name = "/model"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

layer_names_to_hook = []
for name, module in model.named_modules():
    if len(name.split(".")) == 4:
        layer_names_to_hook.append(name)

===================================  Record Data
hook_handles = []
for name, module in model.named_modules():
    if len(name.split(".")) == 4:
        print(f"Registering hook for: {name}")
        handle = module.register_forward_hook(save_activation_hook(name))
        hook_handles.append(handle)



inputs = tokenizer("Hello, this is Alice in the wonderland!!", return_tensors="pt")

with torch.no_grad():
    tensor_directory = "/home/vpolamre/ML_With_PyTorch_Scikit_Practice/Profiling/hooks/run/tensor_model_iter_1"
    os.makedirs(tensor_directory, exist_ok=True)
    model(**inputs)

for name in layer_names_to_hook:
    tensor_list = activations[name]
    if tensor_list:
        print(f"Activations for '{name}':")
        print(f"Shape: {tensor_list[0].shape}\n")
    else:
        print(f"No activation captured for '{name}'")


for handle in hook_handles:
    handle.remove()
=====================================================


#===================================  Replay Data
tensor_directory = "/home/vpolamre/ML_With_PyTorch_Scikit_Practice/Profiling/hooks/run/tensor_model_iter_1"

def get_inputs_outputs(layer_name: str):
    output_file = os.path.join(tensor_directory, f"{layer_name}.output.pt")
    input_file  = os.path.join(tensor_directory, f"{layer_name}.input0.pt")
    if (not os.path.exists(output_file)) and (not os.path.exists(input_file)):
        return None 

    tensor_out = torch.load(output_file)
    tensor_in = (torch.load(input_file))
    for i in range(1,100):
        in_file  = (os.path.join(tensor_directory, f"{layer_name}.input{i}.pt"))
        if os.path.exists(in_file):
            tensor_in.append(torch.load(in_file))

    return (tensor_in, tensor_out)


for layer_name in layer_names_to_hook:
    activations[layer_name] = get_inputs_outputs(layer_name)
    if activations[layer_name] != None:
        print(f"{layer_name:<30}: IN_SIZE: {len(activations[layer_name][0]):<10}; IN: {str(activations[layer_name][0][0].shape):<30}; OUT: {str(activations[layer_name][1].shape):<30}")

for layer_name, layer in model.named_modules():
    if len(layer_name.split(".")) == 4 and activations[layer_name] != None:
        output = layer(activations[layer_name][0].requires_grad_())
        loss = output.sum()  # or any dummy loss
        loss.backward()

        
        if torch.allclose(output, activations[layer_name][1]) and not torch.isnan(output).any():
            print(f"{layer_name: <30}: OK")
        else:
            print(f"{layer_name: <30}: FAIL")


print("Completed!!")