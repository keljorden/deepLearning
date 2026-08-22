import torch

print("-" * 45)
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version    : {torch.version.cuda}")
    print(f"Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"Device Count    : {torch.cuda.device_count()}")
    device = torch.device("cuda")
else:
    print("Device Name     : CPU (No CUDA detected)")
    device = torch.device("cpu")

print("-" * 45)


x = torch.randn(3, 3, device=device)
y = torch.randn(3, 3, device=device)
z = torch.matmul(x, y)

print(f"Tensor computation successful on [{device}]:")
print(z)
print("-" * 45)
print("Verification complete!")