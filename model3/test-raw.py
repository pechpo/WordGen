from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import pandas as pd

#测试模型

path = "../story_generation_dataset/ROCStories_test.csv"
data = pd.read_csv(path)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model_name = "gpt2-finetune-raw"
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to("cuda")
model.eval()

for idx in range(len(data)):
    prompt = data["storytitle"][idx] + "\n" + data["sentence1"][idx] + " "
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask
    outputs = model.generate(input_ids=inputs.to("cuda"), max_new_tokens=100, do_sample=True, top_k=10, top_p=0.95,
        attention_mask=attention_mask.to("cuda"), pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(text)