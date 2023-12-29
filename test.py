from model import *
from dataset import *
from tqdm import tqdm
from setting import *
import pandas as pd

# 数据集处理
batch_size = 16
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
max_len = 200  # 最大输出长度

encoder = TransformerEncoder(
    input_dim, hidden_dim, num_layers, num_heads, dropout)
decoder = TransformerDecoder(
    output_dim, hidden_dim, num_layers, num_heads, dropout)
model = TransformerSeq2Seq(encoder, decoder)
model = model.to(device)
model.load_state_dict(torch.load("model 1.pth"))

model.eval()
res = [[]] * len(test_dataset)
with torch.no_grad():
    for batch_id, (seq_in, seq_out, mask_in, mask_out) in enumerate(tqdm(test_dataloader)):
        seq_in = seq_in.to(device)
        #seq_out = seq_out.to(device)  # 不用
        mask_in = mask_in.to(device)
        #mask_out = mask_out.to(device)  # 由于输出序列长度可能不一致，这里不能用mask_out
        mask_out = None
        # 测试的阶段使用串行的token生成
        decoder_input = torch.zeros(batch_size, max_len, dtype=torch.int64).to(device)
        decoder_input[:, 0] = begin_token
        token_pos = 1
        end = [False] * batch_size  # 表示该条故事是否还在生成
        while not all(end):
            output = model(seq_in, decoder_input, mask_in, mask_out)
            decoder_input[:, token_pos] = torch.argmax(output, dim=2)[:, token_pos-1]
            for i in range(0, batch_size):
                if end[i] == True:
                    continue
                id = batch_id * batch_size + i
                res[id].append(decoder_input[i][token_pos])
                if decoder_input[i][token_pos] == end_token or token_pos == max_len-1:
                    end[i] = True
            token_pos += 1
        for i in range(0, batch_size):
            id = batch_id * batch_size + i
            print(res[id])

enc = tiktoken.get_encoding("cl100k_base")
id_list = []
text_result = []
for story_id, seq in enumerate(res):
    text = enc.decode(seq)
    id_list.append(story_id)
    text_result.append(text)
table = pd.DataFrame({"story id": id_list, "text": text_result})
table.to_csv("result.csv")