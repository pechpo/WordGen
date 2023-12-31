from model import *
from dataset import *
from tqdm import tqdm
from setting import *
import pandas as pd

def beam_search(model, seq_in, mask_in, max_len, beam_width=5):
    batch_size = seq_in.size(0)
    end_token = torch.tensor([end_token]).to(seq_in.device)

    # 初始化beam search
    beams = [{'tokens': [begin_token], 'log_prob': 0.0} for _ in range(batch_size)]

    for token_pos in range(1, max_len):
        next_beams = []

        for beam in beams:
            if beam['tokens'][-1] == end_token or token_pos == max_len - 1:
                next_beams.append(beam)
                continue

            decoder_input = torch.tensor(beam['tokens'], dtype=torch.int64).unsqueeze(0).to(seq_in.device)
            output = model(seq_in, decoder_input, mask_in, mask_out=None)

            prob, pos = output[:, -1, :].topk(beam_width)
            for i in range(beam_width):
                new_token = pos[0, i].item()
                log_prob = beam['log_prob'] + torch.log(prob[0, i]).item()

                new_beam = {
                    'tokens': beam['tokens'] + [new_token],
                    'log_prob': log_prob,
                }

                next_beams.append(new_beam)

        next_beams = sorted(next_beams, key=lambda x: x['log_prob'], reverse=True)[:beam_width]
        beams = next_beams

    return [beam['tokens'] for beam in beams]

# 数据集处理
test_dataset = ROCStories_dataset("../story_generation_dataset/ROCStories_test.csv")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

model = Transformer(input_dim, hidden_dim, num_layers, num_heads, dropout)
model = model.to(device)
model.load_state_dict(torch.load("model 4.pth"))

enc = tiktoken.get_encoding("cl100k_base")
model.eval()
res = [[] for _ in range(len(test_dataset))]

with torch.no_grad():
    for batch_id, (seq_in, _, mask_in, _) in enumerate(tqdm(test_dataloader)):
        seq_in = seq_in.to(device)
        mask_in = mask_in.to(device)

        beam_search_tokens = beam_search(model, seq_in, mask_in, max_len, beam_width=10)

        for i in range(batch_size):
            id = batch_id * batch_size + i
            res[id] = beam_search_tokens[i]

# 生成结果
id_list = []
text_result = []
for story_id, seq in enumerate(res):
    text = enc.decode(seq)
    id_list.append(story_id)
    text_result.append(text)

# 保存到CSV
table = pd.DataFrame({"story id": id_list, "text": text_result})
table.to_csv("result.csv", index=False)
