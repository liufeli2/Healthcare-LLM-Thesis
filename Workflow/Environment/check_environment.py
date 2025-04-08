# python /scratch/k/khalvati/liufeli2/LLMs/llama32vision/check_pytorch.py

# Script to check environment setup
print("Checking environment setup...\n")

# Check CUDA availability in PyTorch
try:
    import torch
    print("PyTorch imported successfully.")
    print("CUDA available:", torch.cuda.is_available())
except ImportError:
    print("PyTorch is not installed.")

# Check if requests is available
try:
    import requests
    print("Requests imported successfully.")
except ImportError:
    print("Requests is not installed.")

# Check if PIL (Pillow) is available
try:
    from PIL import Image
    print("Pillow (PIL) imported successfully.")
except ImportError:
    print("Pillow (PIL) is not installed.")

# Check Hugging Face Transformers availability
try:
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    print("Hugging Face Transformers imported successfully.")
except ImportError:
    print("Hugging Face Transformers is not installed.")


# Check if additional libraries are available
try:
    import random
    print("Random imported successfully.")
except ImportError:
    print("Random is not installed.")

try:
    import os
    print("OS imported successfully.")
except ImportError:
    print("OS is not installed.")

try:
    import time
    print("Time imported successfully.")
except ImportError:
    print("Time is not installed.")

# Check computational libraries
try:
    import numpy as np
    print("NumPy imported successfully.")
except ImportError:
    print("NumPy is not installed.")

try:
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    print("PyTorch modules (nn, optim, DataLoader, TensorDataset) imported successfully.")
except ImportError:
    print("One or more PyTorch modules (nn, optim, DataLoader, TensorDataset) are not installed.")

print("\nEnvironment check complete.")




'''
Checking environment setup...

PyTorch imported successfully.
CUDA available: True
Requests imported successfully.
Pillow (PIL) imported successfully.
/home/k/khalvati/liufeli2/.conda/envs/llms/lib/python3.9/site-packages/torchvision/datapoints/__init__.py:12: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)
/home/k/khalvati/liufeli2/.conda/envs/llms/lib/python3.9/site-packages/torchvision/transforms/v2/__init__.py:54: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)
Hugging Face Transformers imported successfully.
Random imported successfully.
OS imported successfully.
Time imported successfully.
NumPy imported successfully.
PyTorch modules (nn, optim, DataLoader, TensorDataset) imported successfully.

Environment check complete.
'''