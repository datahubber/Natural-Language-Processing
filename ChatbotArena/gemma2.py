import os
import copy
from dataclasses import dataclass
os.environ['NCCL_IB_DISABLE']='1'
os.environ['NCCL_P2P_DISABLE']='1'
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score


@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "model/lmsys/lmsys-r-gemma-s1-mix110-merged-headless" #"rbiswasfc/lmsys-r-gemma-s1-mix110-merged-headless"  # 4-bit quantized gemma-2-9b-instruct
    max_length: int = 4096
    n_splits: int = 5
    fold_idx: int = 0
    optim_type: str = "adamw_8bit"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # global batch size is 8
    per_device_eval_batch_size: int = 4
    n_epochs: int = 3
    freeze_layers: int = 0  # there're 42 layers in total, we don't add adapters to the first 16 layers
    lr: float = 2e-4
    warmup_steps: int = 20
    lora_r: int = 64
    lora_alpha: float = 4
    lora_dropout: float = 0.05
    lora_bias: str = "none"

config = Config()


training_args = TrainingArguments(
    output_dir="output/gemma_model_checkpoints",
    overwrite_output_dir=True,
    report_to="none",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=200,
    optim=config.optim_type,
    fp16=True,
    learning_rate=config.lr,
    warmup_steps=config.warmup_steps,
)

lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    # only target self-attention
    target_modules=["q_proj", "k_proj", "v_proj","down_proj","up_proj","o_proj","gate_proj"],
    layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.SEQ_CLS,
)

tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
# tokenizer.add_eos_token = True  # We'll add <eos> at the end
# tokenizer.padding_side = "right"
# Save the tokenizer
tokenizer.save_pretrained("content/path_to_save_gemma2_model")

bnb_config =  BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False,
)
model = Gemma2ForSequenceClassification.from_pretrained(
    config.checkpoint,
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.save_pretrained("content/path_to_save_gemma2_model")

ds = Dataset.from_parquet("data/train.parquet")
ds = ds.select(torch.arange(100))
print(ds[0])
import json
from transformers import PreTrainedTokenizerBase

class CustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
        response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]]
        texts = [self.prepare_text(p, r_a, r_b) for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)

        # Map the winner column to labels
        label_map = {"model_a": 0, "model_b": 1}
        labels = [label_map[winner] for winner in batch["winner"]]

        # Return tokenized inputs with labels
        return {**tokenized, "labels": labels}

    @staticmethod
    def process_text(text: str) -> str:
        try:
            # Safely parse JSON strings
            parsed_text = json.loads(text)
            if isinstance(parsed_text, list):
                return " ".join(str(item) for item in parsed_text)
            elif isinstance(parsed_text, dict):
                return " ".join(str(value) for value in parsed_text.values())
            else:
                return str(parsed_text)
        except json.JSONDecodeError:
            # Handle non-JSON text gracefully
            return text

    def prepare_text(self, prompt, response_a, response_b):
        """Prepares text by truncating from the beginning if it exceeds the maximum token length."""
        conversation = (
            f"<start_of_turn>prompt\n{prompt}<end_of_turn>\n"
            + f"<start_of_turn>response_a\n{response_a}<end_of_turn>\n"
            + f"<start_of_turn>response_b\n{response_b}<end_of_turn>"
        )

        # Tokenize the full text
        input_ids = self.tokenizer(conversation)["input_ids"]

        # If the tokenized length exceeds the max_length, truncate from the beginning
        if len(input_ids) > self.max_length:
            truncated_input_ids = input_ids[-self.max_length:]
            conversation = self.tokenizer.decode(truncated_input_ids, skip_special_tokens=True)

        return conversation

import json
encode = CustomTokenizer(tokenizer, max_length=config.max_length)
ds = ds.map(encode, batched=True)

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}

folds = [
    (
        [i for i in range(len(ds)) if i % config.n_splits != fold_idx],
        [i for i in range(len(ds)) if i % config.n_splits == fold_idx]
    )
    for fold_idx in range(config.n_splits)
]

train_idx, eval_idx = folds[config.fold_idx]

trainer = Trainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds.select(train_idx),
    eval_dataset=ds.select(eval_idx),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
#trainer.train()
trainer.train(resume_from_checkpoint="checkpoint-4844")

import shutil
import os

# Define the source folder and the destination in Google Drive
source_folder = 'content/path_to_save_gemma2_model'
destination_folder = 'content/gemma2_model'

# Check if the destination folder exists
if os.path.exists(destination_folder):
    # If it exists, remove it
    shutil.rmtree(destination_folder)

# Copy the folder to the destination
shutil.copytree(source_folder, destination_folder)

print(f"Folder copied to {destination_folder} successfully.")










