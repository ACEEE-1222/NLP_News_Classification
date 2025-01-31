import os

os.chdir(r'D:\Desktop\MachineLearning\NLP\NLP新闻分类赛\Bert')
import pandas as pd
from datasets import load_dataset
from datasets import Dataset

train_df = pd.read_csv('./data/train_set.csv', sep='\t')
test_df = pd.read_csv('./data/test_a.csv', sep='\t')
df = pd.concat((train_df, test_df))
df.columns

#data_collator是一个函数，负责获取样本并将它们批处理成张量
#在data_collator中可以确保每次以新的方式完成随机掩蔽。
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

#初始化bert模型，参数参考讨论区代码
from transformers import BertConfig

config = BertConfig(
    vocab_size=vocab_size,
    hidden_size=512,
    intermediate_size=4 * 512,
    max_position_embeddings=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    type_vocab_size=2
)
from transformers import BertForMaskedLM

model = BertForMaskedLM(config=config)
from transformers import BertTokenizer

tokenizer = BertTokenizer(vocab_file=vocab_file_path)


def group_texts(examples,chunk_size=256):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result
#%%
lm_datasets = tokenized_dataset.map(group_texts, batched=True, num_proc=16)
lm_datasets
#%%
#加载和保存拼接后的文本，掉线的时候这么做
lm_datasets.save_to_disk('./preTrain_data')
# import pandas as pd
# from datasets import load_from_disk
# lm_datasets=load_from_disk('./preTrain_data')

#%%
# #解码分词器预处理的lm_datasets数据，里面有标点符号
# la=tokenizer.decode(lm_datasets[0]['input_ids'])
# la
#%%
# num_samples = len(lm_datasets)
# print(f"数据集的样本数为: {num_samples}")
#%%
#使用GPU训练，运行这段代码
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from transformers import Trainer, TrainingArguments



training_args = TrainingArguments(
    output_dir="preTrain_log",
    logging_strategy="steps",
    logging_steps=2000,
    save_strategy="steps",
    save_steps=6000,
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    fp16=False,  # 必须关闭
    bf16=True,   # 启用BF16
    weight_decay=0.01)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    data_collator=data_collator)

#训练并保存模型
trainer.train()

trainer.save_model("./model")