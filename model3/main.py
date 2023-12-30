from dataset import *
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import math
from transformers import pipeline

#获取模型
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

#训练
training_args = TrainingArguments(
    output_dir="my_awesome_eli5_clm-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
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

#测试模型

prompt = "Bad Dream\nTommy was very close to his dad and loved him greatly. His was a cop and was shot and killed on duty. "
generator = pipeline("text-generation", model="my-gpt2-model")
generator(prompt)