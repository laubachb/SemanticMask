import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.version.cuda)         # Prints the version of CUDA PyTorch is compiled with
