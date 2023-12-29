from model import *
from dataset import *
from tqdm import tqdm
from setting import *
import torch.optim as optim

# 数据集处理
batch_size = 10
test_dataset = ROCStories_dataset(
    "../story_generation_dataset/ROCStories_test.csv")
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
)

# 定义和加载模型
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
model.load_state_dict(torch.load("model.pth"))

model.eval()
res = [""] * len(test_dataset)
with torch.no_grad():
    for seq_in, seq_out, mask_in, mask_out in tqdm(test_dataloader):
        seq_in = seq_in.to(device)
        seq_out = seq_out.to(device)
        mask_in = mask_in.to(device)
        #mask_out = mask_out.to(device)  # 由于输出序列长度可能不一致，这里不能用mask_out
        mask_out = None
        # 测试的阶段使用串行的token生成
        decoder_input = torch.tensor(seq_out).to(device)
        for i in range(1, seq_out.shape(1)):
            output = model(seq_in, decoder_input[:, :-1], mask_in, mask_out)
            decoder_input[:, i] = torch.argmax(output)