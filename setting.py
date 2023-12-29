import torch

# 设备处理
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

begin_token = 100300
end_token = 100301
padding_token = 100302