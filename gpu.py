import torch
print(torch.cuda.is_available())
print("CUDA Available: ", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # Prints the name of the GPU
print(torch.cuda.get_device_name(0))  # Prints the name of the GPU
