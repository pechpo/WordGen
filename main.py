from model import *
from dataset import *
from tqdm import tqdm
from setting import *
import torch.optim as optim

# 设备处理
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 数据集处理
batch_size = 10
train_dataset = ROCStories_dataset(
    "../story_generation_dataset/ROCStories_train.csv")
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
)
val_dataset = ROCStories_dataset(
    "../story_generation_dataset/ROCStories_val.csv")
val_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
)

# 示例数据
# src_data = torch.randint(0, 100, (10, 32))  # 10个句子，每个句子32个词
# tgt_data = torch.randint(0, 100, (10, 30))  # 10个目标句子，每个句子30个词

# 定义模型和优化器
input_dim = 100400  # 输入词典大小
output_dim = 100400  # 输出词典大小
hidden_dim = 512  # 隐藏层大小
num_layers = 6  # Transformer层数
num_heads = 8  # 注意力头数
dropout = 0.1  # Dropout概率
epoch = 10

encoder = TransformerEncoder(
    input_dim, hidden_dim, num_layers, num_heads, dropout)
decoder = TransformerDecoder(
    output_dim, hidden_dim, num_layers, num_heads, dropout)
model = TransformerSeq2Seq(encoder, decoder)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 损失函数
criterion = nn.CrossEntropyLoss()

loss_min = 1e9

for epoch_id in range(0, epoch):
    # 训练
    pbar = tqdm(total=len(train_dataloader), desc=f"Training Epoch {epoch_id}")
    for seq_in, seq_out, mask_in, mask_out in train_dataloader:
        # print(seq_in.shape, seq_out.shape, sep="\n")
        seq_in = seq_in.to(device)
        seq_out = seq_out.to(device)
        mask_in = mask_in.to(device)
        mask_out = mask_out.to(device)
        # print(seq_in, seq_out, sep="\n")
        optimizer.zero_grad()
        output = model(seq_in, seq_out[:, :-1], mask_in, mask_out)  # 去掉目标句子的最后一个词，作为decoder输入
        loss = criterion(
            output.reshape(-1, output.shape[-1]), seq_out[:, 1:].reshape(-1)
        )  # 计算交叉熵损失，比较模型输出和去掉目标句子的第一个词的序列
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})
        pbar.update(1)
    pbar.close()
    # 验证
    pbar = tqdm(total=len(val_dataloader), desc=f"Validating Epoch {epoch_id}")
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for seq_in, seq_out, mask_in, mask_out in val_dataloader:
            seq_in = seq_in.to(device)
            seq_out = seq_out.to(device)
            # 验证的阶段依旧使用teacher forcing，只有测试的时候不知道正确答案，需要自己生成
            output = model(seq_in, seq_out[:, :-1], mask_in, mask_out)
            loss = criterion( output.reshape(-1, output.shape[-1]), seq_out[:, 1:].reshape(-1))
            total_loss += loss
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
        pbar.close()
    total_loss /= len(val_dataloader)
    print(f"Total Loss = {total_loss}")
    if total_loss < loss_min:
        loss_min = total_loss
        print(f"Temporarily Best Result in Epoch {epoch_id}, Saving Result.")
        torch.save(model.state_dict(), "model.pth")
