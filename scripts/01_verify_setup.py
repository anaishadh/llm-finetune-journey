import torch
import transformers
import peft
import trl

print("=" * 40)
print("Environment Check")
print("=" * 40)
print(f"PyTorch:        {torch.__version__}")
print(f"Transformers:   {transformers.__version__}")
print(f"PEFT:           {peft.__version__}")
print(f"TRL:            {trl.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM:           {vram:.1f} GB")
    print("=" * 40)
    print("All good! Ready to fine-tune.")
else:
    print("WARNING: No GPU detected!")
print("=" * 40)
