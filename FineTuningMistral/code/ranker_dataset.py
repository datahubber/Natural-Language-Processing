from copy import deepcopy

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm 
IGNORE_INDEX = -100

def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.backbone_path,
        use_fast=False,#cfg.model.tokenizer.use_fast,
        add_eos_token=False,
        truncation_side=cfg.model.tokenizer.truncation_side,
    )
    tokenizer.add_bos_token = True
    tokenizer.padding_side = "left"  # use left padding

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eod_id is not None:
            tokenizer.pad_token = tokenizer.eod
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token = tokenizer.im_start
            tokenizer.bos_token_id = tokenizer.im_start_id
            tokenizer.eos_token = tokenizer.im_end
            tokenizer.eos_token_id = tokenizer.im_end_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.bos_token =  "<|im_start|>"
    # tokenizer.bos_token_id = 151644
    print( tokenizer.bos_token)
    print(tokenizer.eos_token)
    print( tokenizer.bos_token_id)
    print(tokenizer.eos_token_id)
    return tokenizer
def find_token_instruction_masking(input_ids, token_pattern):
    """Find the last occurrence of a token_pattern in a list."""
    ret = 0
    token_pattern_len = len(token_pattern)
    # for ex_input_ids in input_ids:
    search_end = len(input_ids)
    found = False
    # print("token_pattern_len",token_pattern_len)
    # print(input_ids)
    for j in range(search_end - token_pattern_len, -1, -1):
        if input_ids[j : j + token_pattern_len] == token_pattern:
            ret=j + token_pattern_len
            found = True
            break
    if not found:
        print("not found!")
        ret=0  # If not found, return 0 # assuming left truncation
    return ret


class RankerDataset:
    """
    Dataset class for EEDI - Misconception Detection
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)
        self.token_pattern = self.tokenizer.encode("Based on your role and its description, give short answer:\n", add_special_tokens=False)
        

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(
            examples["text"]+"Answer:\n",
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            return_length=True,
            add_special_tokens=True,
        )
        expl_model_inputs =self.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            return_length=True,
            add_special_tokens=True,
            # return_token_type_ids=False,
        )
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
        
        labels = deepcopy(expl_model_inputs["input_ids"])
        # assistant_start_idxs = find_token_instruction_masking(expl_model_inputs["input_ids"], self.token_pattern)
        # print("assistant_start_idxs",assistant_start_idxs)

        # labels[:assistant_start_idxs] = [IGNORE_INDEX] * assistant_start_idxs

        # rationale_output_encodings = self.tokenizer(cot, max_length=self.cfg.model.max_length, truncation=True,padding=False, return_length=True)
        model_inputs['aux_labels'] = labels
        return model_inputs
    
    def process_conversations(self,data):
        """
        处理对话数据，将第一个user之前的所有内容（包括system和assistant）合并到第一个user的内容中。
        如果第一个role是user，则保持原样。
        
        参数:
            data (list): 包含对话数据的列表，每个元素是一个字典，包含'role'和'content'键。
        
        返回:
            list: 处理后的对话数据。
        """
        if not data:
            return []
    
        # 检查第一个role是否是user
        if data[0]['role'] == 'user':
            return data  # 如果第一个role是user，直接返回原数据
    
        # 如果第一个role不是user，找到第一个user的位置
        first_user_index = None
        for i, item in enumerate(data):
            if item['role'] == 'user':
                first_user_index = i
                break
    
        if first_user_index is None:
            return data  # 如果没有找到user，直接返回原数据
    
        # 将第一个user之前的所有内容合并到第一个user的内容中
        merged_content = ' '.join([item['content'] for item in data[:first_user_index]])
        data[first_user_index]['content'] = merged_content + ' ' + data[first_user_index]['content']
    
        # 返回处理后的数据，去掉第一个user之前的内容
        return data[first_user_index:]
    def preprocess_messages(self,messages):
        
        processed_messages = []
        system_message = ""
    
        for message in messages:
            if message['role'] == 'system':
                # 如果是 system 角色，保存其内容
                system_message += message['content'] + "\n\n"
            else:
                # 如果是 assistant 或 user 角色，将 system 消息添加到当前消息中
                if system_message:
                    message['content'] = system_message + message['content']
                    system_message = ""  # 清空 system 消息
                processed_messages.append(message)
    
        return processed_messages
    def preprocess_function(self, df, is_train=False, rng=None):
        
        formatted_texts = []
        

        for _, row in df.iterrows():
            # few_shot_examples = row["examples"]

            # user_message = ""
            # system=f"You are {row['role']}, your description is: {row['description']}. Now please answer some questions to accurately show your personality traits! Your speaking style should fully imitate the personality role assigned to you! Please do not expose that you are an artificial intelligence model or a language model, you must always remember that you are only assigned one personality role. Don't be verbose or too formal or polite when speaking."


            # user_message += f"{row['question']}"

            # conversation = [
            #     {"role": "system", "content": system},
            #     {"role": "user", "content": user_message},
            # ]
            data = row['conversations']
            conversation  = [
                {"role": item['from'], "content": item['value']} for item in data
            ]
            # conversation = self.preprocess_messages(conversation) #.replace("system","user")
            conversation = self.process_conversations(conversation)
            # print( conversation[:100])
            # return 
            try:
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            except:
                print( conversation)
                return 
            formatted_texts.append(text)

        df["text"] = formatted_texts
        return df

    def get_dataset(self, df, is_train=False, rng=None):
        """use this function to get the dataset

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            Dataset: HF Dataset object with tokenized inputs and labels
        """

        df = deepcopy(df)
        df = self.preprocess_function(df, is_train, rng)
        # df = self.preprocess_function_truncation(df, is_train,ad_fs=self.cfg.model.add_fs)
        # text_column_df = df[['text']]
    
        # # 将 DataFrame 保存为 CSV 文件
        # text_column_df.to_csv("output/all_calib_data.csv", index=False)
        # text_column_df.to_parquet("output/all_calib_data.parquet", index=False)
        task_dataset = Dataset.from_pandas(df)
        # print(df.columns)
        remove_columns = [col for col in df.columns if col not in ["query_id", "content_ids", "combined_id", "winner", "teacher_logits","generated"]]

        task_dataset = task_dataset.map(self.tokenize_function, batched=False, num_proc=self.cfg.model.num_proc, remove_columns=remove_columns)
        # task_dataset = task_dataset.map(lambda x: self.tokenize_function(x, ad_fs= self.cfg.add_fs), batched=False, num_proc=self.cfg.model.num_proc, remove_columns=remove_columns)

        return task_dataset
