import json
import os
import random

import hydra
# import kagglehub
import numpy as np
import pandas as pd
import torch
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from ranker_listwise.ranker_dataset import RankerDataset
from ranker_listwise.ranker_loader import RankerCollator, show_batch
from ranker_listwise.ranker_model import EediRanker, get_base_model
from ranker_listwise.ranker_optimizer import get_optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.train_utils import AverageMeter, add_fs_examples_listwise, add_example_wsdm,eedi_process_df, get_custom_cosine_schedule_with_warmup, get_lr, setup_training_run, train_valid_split
from datasets import Dataset
logger = get_logger(__name__)
torch._dynamo.config.optimize_ddp = False


def format_example(row, id2name, query2content):
    cid = int(query2content[row["query_id"]])
    misconception_name = id2name[cid]
    example = f"Question: {row['QuestionText']}\nAnswer:{row['CorrectAnswerText']}\nIncorrect Answer: {row['InCorrectAnswerText']}\nMisconception: {misconception_name}"
    return example


def to_list(t):
    return t.float().cpu().tolist()

def process_conversations(df):
    """
    处理对话数据，确保user和assistant交替出现。
    如果system后出现的是assistant，则合并到system；如果system后出现的是user，则保留system和user。
    
    参数:
        df (pd.DataFrame): 包含对话数据的DataFrame，其中包含'from'和'value'两列。
    
    返回:
        pd.DataFrame: 处理后的DataFrame。
    """
    processed_data = []  # 用于存储处理后的数据
    i = 0  # 遍历索引

    while i < len(df):
        row = df.iloc[i]
        if row['from'] == 'system':
            # 如果当前行是system，检查下一行
            if i + 1 < len(df):
                next_row = df.iloc[i + 1]
                if next_row['from'] == 'assistant':
                    # 如果system后是assistant，合并到system
                    merged_value = row['value'] + ' ' + next_row['value']
                    processed_data.append({'from': 'system', 'value': merged_value})
                    i += 2  # 跳过已经处理的assistant行
                elif next_row['from'] == 'user':
                    # 如果system后是user，直接保留system和user
                    processed_data.append(row)
                    processed_data.append(next_row)
                    i += 2  # 跳过已经处理的user行
                else:
                    # 如果system后既不是assistant也不是user，直接保留system
                    processed_data.append(row)
                    i += 1
            else:
                # 如果system是最后一行，直接保留
                processed_data.append(row)
                i += 1
        else:
            # 检查user和assistant的交替顺序
            if i + 1 < len(df) and df.iloc[i + 1]['from'] == 'assistant':
                processed_data.append(row)
                processed_data.append(df.iloc[i + 1])
                i += 2
            else:
                # 如果顺序不符合，跳过当前行
                i += 1

    # 将处理后的数据转换为DataFrame
    processed_df = pd.DataFrame(processed_data)
    return processed_df
def sort_by_scores(pred_ids, scores):
    keep_idxs = np.argsort(-np.array(scores)).tolist()
    ret_ids = [pred_ids[idx] for idx in keep_idxs]
    ret_scores = [scores[idx] for idx in keep_idxs]
    return {"sorted_ids": ret_ids, "sorted_scores": ret_scores}
def check_conversation_order(conversation):
    # 检查第一个角色是否是 'system'
    if conversation and conversation[0]['from'] == 'system':
        system_message = conversation[0]
        # 检查 system 后面的第一个角色是否是 'assistant'
        if len(conversation) > 1 and conversation[1]['from'] == 'assistant':
            assistant_message = conversation.pop(1)  # 移除第一个 assistant 消息
            system_message['value'] += "\n\n" + assistant_message['value']  # 合并内容
    roles = [msg['from'] for msg in conversation]
    # 移除 'system' 角色
    roles = [role for role in roles if role != 'system']
    # 检查角色是否交替出现
    for i in range(len(roles) - 1):
        if roles[i] == roles[i + 1]:
            return False
    return True
def process_conversation(messages):
    # Step 1: 合并system后的所有连续assistant消息
    merged_messages = []
    i = 0
    while i < len(messages):
        current_msg = messages[i]
        if current_msg['from'] == 'system':
            # 合并后续连续的assistant消息
            merged_value = current_msg['value']
            j = i + 1
            while j < len(messages) and messages[j]['from'] == 'assistant':
                merged_value += ' ' + messages[j]['value']
                j += 1
            merged_msg = {'from': 'system', 'value': merged_value.strip()}
            merged_messages.append(merged_msg)
            i = j  # 跳过已合并的消息
        else:
            merged_messages.append(current_msg)
            i += 1

    # Step 2: 检查对话顺序是否有效
    if not merged_messages:
        return None  # 空对话无效
    
    valid_roles = {'user', 'assistant', 'system'}
    prev_role = None
    for idx, msg in enumerate(merged_messages):
        current_role = msg['from']
        if current_role not in valid_roles:
            return None  # 存在无效角色
        
        if idx == 0:
            # 检查起始角色
            if current_role not in ['system', 'user']:
                return None
            if current_role == 'system':
                # system后必须接user
                if len(merged_messages) < 2 or merged_messages[1]['from'] != 'user':
                    return None
            prev_role = current_role
        else:
            if prev_role == 'system':
                # system之后必须是user
                if current_role != 'user':
                    return None
                prev_role = current_role
            else:
                # 检查交替顺序
                if current_role == prev_role:
                    return None
                if prev_role == 'user' and current_role != 'assistant':
                    return None
                if prev_role == 'assistant' and current_role != 'user':
                    return None
                prev_role = current_role
    
    return merged_messages
@hydra.main(version_base=None, config_path="conf/ranker_listwise", config_name="conf_listwise")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    accelerator = setup_training_run(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg.local_rank = accelerator.process_index
    rng = random.Random(cfg.seed)

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit * 50 + suffix)

    print_line()
    accelerator.print(json.dumps(cfg_dict, indent=4))

    # ------- load data -----------------------------------------------------------------#
    print_line()


#     ds = Dataset.from_parquet(cfg.dataset.input_dataset)
#     # ds=  ds.select(torch.arange(100))

#     folds = [
#     (
#         [i for i in range(len(ds)) if i % 5 != fold_idx],
#         [i for i in range(len(ds)) if i % 5 == fold_idx]
#     )
#     for fold_idx in range(5)
# ]

#     train_idx, eval_idx = folds[0]

#     train_ds = ds.select(train_idx)
#     valid_ds = ds.select(eval_idx)
#     train_df = train_ds.to_pandas()
#     valid_df = valid_ds.to_pandas()

    ds = Dataset.from_parquet(cfg.dataset.input_dataset)

    train_df = ds.to_pandas()
    valid_df =  ds.select(torch.arange(100)).to_pandas()
    
    print("len(train_df)",len(train_df))
    # 应用函数并过滤 DataFrame




    # 应用处理函数到DataFrame
    train_df['processed_conversation'] =  train_df['conversations'].apply(process_conversation)
    
    # 删除无效数据
    train_df = train_df.dropna(subset=['processed_conversation']).reset_index(drop=True)
    
    # 更新原始conversation列（可选）
    train_df['conversation'] = train_df['processed_conversation']
    train_df = train_df.drop(columns=['processed_conversation'])
    print("len(train_df)",len(train_df))

    valid_df['processed_conversation'] =  valid_df['conversations'].apply(process_conversation)
    
    # 删除无效数据
    valid_df = valid_df.dropna(subset=['processed_conversation']).reset_index(drop=True)
    
    # 更新原始conversation列（可选）
    valid_df['conversation'] = valid_df['processed_conversation']
    valid_df = valid_df.drop(columns=['processed_conversation'])
    if cfg.debug:
        n = min(2, len(train_df))
        n = min(n, len(valid_df))
        train_df = train_df.head(n).reset_index(drop=True)
        valid_df = valid_df.head(n).reset_index(drop=True)
    

    dataset_creator = RankerDataset(cfg)
    tokenizer = dataset_creator.tokenizer
    if cfg.model.add_fs:
        train_df=add_example_wsdm(train_df,tokenizer, max_length=cfg.model.max_length//4, k_shot=cfg.model.k_shot)#up to k_shot examples
        valid_df=add_example_wsdm(valid_df,tokenizer, max_length=cfg.model.max_length//4, k_shot=cfg.model.k_shot) #up to 1/3 tokens to example

    accelerator.print(f"shape of train data: {train_df.shape}")
    accelerator.print(f"shape of validation data: {valid_df.shape}")
    
    
    print_line()
    train_ds = dataset_creator.get_dataset(train_df, is_train=True, rng=rng)
    valid_ds = dataset_creator.get_dataset(valid_df, is_train=False, rng=None)


    
    # train_ds=train_ds.sort("length", reverse=True)
    # valid_ds = valid_ds.sort("length", reverse=True)

    data_collator = RankerCollator(tokenizer=tokenizer, pad_to_multiple_of=16)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.per_device_train_batch_size,
        shuffle=True ,
        collate_fn=data_collator,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    accelerator.print("data preparation done...")
    print_line()

    # --- show batch -------------------------------------------------------------------#
    print_line()
    for idx, b in enumerate(train_dl):
        accelerator.print("TRAINING BATCH 1")
        show_batch(b, tokenizer, task="training", print_fn=accelerator.print)
        if idx > 1:
            break

    print_line()
    accelerator.print("VALIDATION BATCH 1")
    for b in valid_dl:
        break
    show_batch(b, tokenizer, task="validation", print_fn=accelerator.print)

    # --- model -------------------------------------------------------------------------#
    print_line()
    accelerator.print("Loading model....")
    base_model = get_base_model(cfg)
    model = EediRanker(cfg, base_model, tokenizer)

    if cfg.model.use_gradient_checkpointing:
        accelerator.print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    accelerator.wait_for_everyone()

    if cfg.model.compile_model:
        accelerator.print("Compiling model...")
        model = torch.compile(model)

    # --- optimizer ---------------------------------------------------------------------#
    print_line()
    optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)

    # ------- Prepare -------------------------------------------------------------------#

    model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_train_epochs
    grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl) // grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct * num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_custom_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # ------- training setup ---------------------------------------------------------------#
    current_iteration = 0

    # ------- training  --------------------------------------------------------------------#
    accelerator.wait_for_everyone()
    progress_bar = None

    for epoch in range(num_epochs):
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
        loss_meter = AverageMeter()

        # Training ------
        model.train()

        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):  # gives sync vs no sync context manager
                outputs = model(**batch, use_distillation=cfg.use_distillation, temperature=cfg.temperature,multi_task=cfg.multi_task)
                loss = outputs.loss
                # print("loss",loss)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loss_meter.update(loss.item())  # tracks loss in each batch, no accumulation

            if accelerator.sync_gradients:
                progress_bar.set_description(f"STEP: {current_iteration+1:5}/{num_training_steps:5}. " f"LR: {get_lr(optimizer):.4f}. " f"Loss: {loss_meter.avg:.4f}. ")

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)  # only on main process
                    accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)
                    accelerator.log({"total_grad_l2_norm": round(grad_norm.item(), 5)}, step=current_iteration)

                    try:
                        accelerator.log({"distillation_loss": round(outputs.distillation_loss.item(), 5)}, step=current_iteration)
                        accelerator.log({"ce_loss": round(outputs.ce_loss.item(), 5)}, step=current_iteration)
                        accelerator.log({"causalLM_loss": round(outputs.causalLMLoss.item(), 5)}, step=current_iteration)
                    except Exception:
                        pass

            # ------- evaluation  -------------------------------------------------------#
            if (accelerator.sync_gradients) & (current_iteration % cfg.train_params.eval_frequency == 0):
                model.eval()

                # saving -----
                accelerator.wait_for_everyone()

                if cfg.save_model:
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save(cfg.outputs.model_dir)
                        tokenizer.save_pretrained(cfg.outputs.model_dir)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()

    # --- end training
    if cfg.save_model:
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save(cfg.outputs.model_dir)
            tokenizer.save_pretrained(cfg.outputs.model_dir)

    # --- end training
    accelerator.end_training()


if __name__ == "__main__":
    run_training()
