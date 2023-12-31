from dataset import data_handle
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import math
from transformers import pipeline

#数据处理
lm_datasets, data_collator = data_handle()

#获取模型
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model_name = "gpt2-finetune-raw"

#训练
training_args = TrainingArguments(
    output_dir=model_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    num_train_epochs=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["val"],
    data_collator=data_collator,
)

trainer.train()

#评估模型
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

model.save_pretrained(model_name)