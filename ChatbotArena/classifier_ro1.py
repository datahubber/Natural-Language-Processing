# Credit: Mark Tenenholtz
import argparse
import gc
import os
import random
import re
import kagglehub
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams  # noqa

TEMPLATE =  """I have a question and two responses. I will tell you which response the user prefers. You will analyze the question and two responses to identify the specific reasoning that led to user's preferring.
Your goal is to explain precisely how the response caused user to select it.

Here are the question and its two responses:
{question_responses}

**The user perfers to the {winner}.**

First, examine all components of the question carefully:
1. The background and statement of the question.
2. The form of the question to be answered.
3. The user demands reflected by this question.

Next, examine all components of two responses carefully:
1. The user's preferring response.
2. The accuracy, relevance, completeness, conciseness, organization, clarity, objectivity, practicality, language expression, and innovation of two responses.

Then, reconstruct the user's likely thought process:
- Identify the exact point where user's reasoning leading to the preferring response.
- Note which specific components of response the response that users don't like fail to meet the demands of users.

Write your analysis in <evaluation> tags, following this structure:
- Explain the specificadvantages of the perferring response.
- Demonstrate the defects of the less-preferred.
- Keep your explanation to 5-6 clear, non-repetitive sentences.
- Don't mention which responser the user prefers in the explanation
- Just objectively and sequentially analyze the characteristics of Response_A and Response_B respectively.
- Just explan like this: "Response_A ...... Response_B ......"

Guidelines for writing your explanation:
- Do not restate the question and responses.
- Be precise about the connection between the question and responses.
- Show exactly how the preferring response led to the better anwser.
- Note the difference between the two responses.
- Avoid repetition

Now, start with your analysis in English:
"""


def get_tokenizer(backbone_path):
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, add_eos_token=True)

    if tokenizer.eos_token == "":
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        tokenizer.eos_token = "</s>"

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.bos_token = "<|im_start|>"
    tokenizer.padding_side = "left"
    return tokenizer


def main(args):
    seed = random.randint(100, 10000)
    print(f"setting seed to: {seed}")
    random.seed(seed)
    ds = Dataset.from_parquet(args.input_dataset)
    # ds =ds[:32]
    print(f"Number of examples: {len(ds)}")

    # ds = ds.shuffle(seed=seed)
    # ds = ds.select(range(args.num_examples))
    query_ids = ds["id"]
    print(f"# of query_ids: {len(query_ids)}")

    print("==" * 50)

    # tokenizer = get_tokenizer(args.model)
    df = pd.DataFrame(ds)
    print(df.head(5))
    prompts = []
    for _,example in df.iterrows():
        question = example["prompt"]
        response_a = example["response_a"]
        response_b = example["response_b"]
        winner = example["winner"].replace("model","Response").replace("a","A").replace("b","B")
        user_message = f"""<Question>
{question[:3000]}
</Question>
{'---'*10}
<Response_A>
{response_a[:4000]}
</Response_A>
{'---'*10}
<Response_B>
{response_b[:4000]}
</Response_B>
{'---'*10}"""

        text = TEMPLATE.format(question_responses=user_message,winner=winner)
        prompts.append(text)

    # print a few prompts
    for p in prompts[:5]:
        print(p)
        print("-" * 100)
        print(f"Generating for model: {args.model}")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=16384,
        gpu_memory_utilization=0.9,
        max_num_seqs= 128,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(temperature=0, top_k=1, repetition_penalty=1.0, max_tokens=384)
    responses = llm.generate(
        prompts,
        SamplingParams(
            n=1, top_k=1, max_tokens=384, temperature=0, skip_special_tokens=False,
            # logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=["A","B"])]
        ),
        use_tqdm=True
    )
    cot=[]
    for generated_text in responses:
        generated_text = re.search(r'<evaluation>(.*?)</evaluation>', generated_text.outputs[0].text, re.DOTALL)
        if generated_text:
            generated_text = generated_text.group(1)
        else:
            print("No <evaluation> text found.")
        cot.append(generated_text)
    # print(len(cot))
    # print(df.loc[:(nums-1),"cot_text"])
    df["cot_text"] =cot

    try:
        intermediate_path = os.path.join(args.save_dir, f"swap_72b_cot_{seed}.parquet")
        excel_path = os.path.join(args.save_dir, f"cot_{seed}_.xlsx")
        df.to_parquet(intermediate_path)
        # df.to_excel(excel_path,index=False)
        print(f"Saved intermediate results to {intermediate_path}")
    except Exception as e:
        print(f"Error saving intermediate results: {e}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dataset", type=str, default="data/full_swap_trainfold0.parquet")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--save_dir", type=str, default="data/vllm_cot_generated")
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)



# export MKL_THREADING_LAYER=GNU
# export MKL_SERVICE_FORCE_INTEL=1
