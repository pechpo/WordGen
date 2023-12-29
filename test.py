from model import *
from dataset import ROCStories_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# 设备处理
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 数据集处理
test_dataset = ROCStories_dataset("../story_generation_dataset/ROCStories_test.csv")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 示例数据
#src_data = torch.randint(0, 100, (10, 32))  # 10个句子，每个句子32个词
#tgt_data = torch.randint(0, 100, (10, 30))  # 10个目标句子，每个句子30个词

# 定义模型和优化器
input_dim = 100400  # 输入词典大小
output_dim = 100400  # 输出词典大小
hidden_dim = 512  # 隐藏层大小
num_layers = 6     # Transformer层数
num_heads = 8      # 注意力头数
dropout = 0.1      # Dropout概率

encoder = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
decoder = TransformerDecoder(output_dim, hidden_dim, num_layers, num_heads, dropout)
model = TransformerSeq2Seq(encoder, decoder)
model = model.to(device)
model.load_state_dict(torch.load("model.pth"))

#测试
model.eval()
with torch.no_grad():
    for seq_in, seq_out in tqdm(test_dataloader):
        seq_in = seq_in.to(device)
        seq_out = seq_out.to(device)
        output = model(seq_in, seq_out[:, :-1])
        decoder_input = torch.tensor(seq_out)
            #for i in range(1, seq_out.shape(1)):
        #print(output.shape)
        token = output.argmax(2)
        token_list = token.tolist()[0]
        print(token)
        words = enc.decode(token_list)
        print(words)

