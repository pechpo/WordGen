import torch

# 设备处理
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

begin_token = 100300
end_token = 100301
padding_token = 100302

batch_size = 16

# 定义模型和优化器
input_dim = 100400  # 输入词典大小
output_dim = 100400  # 输出词典大小
hidden_dim = 512  # 隐藏层大小
num_layers = 2  # Transformer层数
num_heads = 8  # 注意力头数
dropout = 0.1  # Dropout概率
epoch = 10
max_len = 200