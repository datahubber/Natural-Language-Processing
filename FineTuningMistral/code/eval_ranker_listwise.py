import json
import os
import random
from itertools import chain

import hydra
import kagglehub
import numpy as np
import pandas as pd
import torch
import vllm  # noqa: F401, E402
from omegaconf import OmegaConf

# local imports ---
from ranker_listwise.ranker_dataset import RankerDataset
from sklearn import metrics
from utils.metric_utils import compute_retrieval_metrics, mapk
from utils.train_utils import eedi_process_df, train_valid_split

torch._dynamo.config.optimize_ddp = False


def add_fs_examples(df, content2query, query2example, rng):
    cache = {}

    def _add_examples(row):
        cids = row["content_ids"]

        selected_qids = []
        for cid in cids:
            cid = int(cid)
            qids = content2query[cid]
            qids = [qid for qid in qids if qid != row["query_id"]]
            if len(qids) > 0:
                if cid not in cache:
                    selected = rng.choice(qids)
                    cache[cid] = selected
                    selected_qids.append(cache[cid])
                else:
                    selected_qids.append(cache[cid])

        if len(selected_qids) == 0:
            return ""

        # keep max of N (=4) examples
        selected_qids = rng.sample(selected_qids, k=min(4, len(selected_qids)))
        selected_qids = sorted(selected_qids)  # for prefix cache
        examples = [query2example[qid] for qid in selected_qids]
        fs = "\n--\n".join(examples)
        return fs

    df["examples"] = df.apply(_add_examples, axis=1)
    return df


def format_example(row, id2name, query2content):
    cid = int(query2content[row["query_id"]])
    misconception_name = id2name[cid]
    example = f"Question: {row['QuestionText']}\nAnswer:{row['CorrectAnswerText']}\nIncorrect Answer: {row['InCorrectAnswerText']}\nMisconception: {misconception_name}"
    return example


def sort_by_scores(pred_ids, scores):
    keep_idxs = np.argsort(-np.array(scores)).tolist()
    ret_ids = [pred_ids[idx] for idx in keep_idxs]
    ret_scores = [scores[idx] for idx in keep_idxs]
    return {"sorted_ids": ret_ids, "sorted_scores": ret_scores}


def run_evaluation(result_df, qa2mc):
    agg_df = result_df.copy()

    agg_df["truth"] = agg_df["QuestionId_Answer"].map(qa2mc)
    agg_df["truth"] = agg_df["truth"].apply(lambda x: [str(x)])
    agg_df["y"] = agg_df.apply(lambda x: [1 if y in x["truth"] else 0 for y in x["MisconceptionId"]], axis=1)

    truths = list(chain(*agg_df["y"].values))
    preds = list(chain(*agg_df["score"].values))
    fpr, tpr, thresholds = metrics.roc_curve(truths, preds)
    ranker_auc = metrics.auc(fpr, tpr)

    # sort by score and keep top 25, remember score is a list
    agg_df["topk_info"] = agg_df.apply(lambda x: sort_by_scores(x["MisconceptionId"], x["score"]), axis=1)
    agg_df["MisconceptionId"] = agg_df["topk_info"].apply(lambda x: x["sorted_ids"])
    agg_df["score"] = agg_df["topk_info"].apply(lambda x: x["sorted_scores"])

    lb = mapk(agg_df["truth"].values, agg_df["MisconceptionId"].values, k=25)

    # get recall scores ---
    recall_dict = dict()
    cutoffs = [1, 2, 3, 4, 5]
    for cutoff in cutoffs:
        cdf = agg_df.copy()
        cdf["MisconceptionId"] = cdf["MisconceptionId"].apply(lambda x: x[:cutoff])
        m = compute_retrieval_metrics(cdf["truth"].values, cdf["MisconceptionId"].values)
        recall_dict[f"recall@{cutoff}"] = m["recall_score"]

    print("--------------------------------")
    for pt in cutoffs:
        print(f">>> Current Recall@{pt} = {round(recall_dict[f'recall@{pt}'], 4)}")
    print("--------------------------------")

    # compute oof dataframe ---
    oof_df = agg_df.copy()
    oof_df = oof_df[["QuestionId_Answer", "MisconceptionId", "score"]].copy()
    oof_df = oof_df.rename(columns={"score": "pred_scores"})

    to_return = {
        "lb": lb,
        "auc": ranker_auc,
        "result_df": result_df,
        "oof_df": oof_df,
        "recall_dict": recall_dict,
    }

    return to_return


@hydra.main(version_base=None, config_path="../conf/llm_tutor", config_name="conf_r_tutor_infer")
def main(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    rng = random.Random(cfg.seed)

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        print(prefix + unit * 50 + suffix)

    print_line()
    print(json.dumps(cfg_dict, indent=4))

    # ------- load data -----------------------------------------------------------------#
    print_line()

    # download data in main process
    input_dir = kagglehub.dataset_download(cfg.dataset.input_dataset)
    try:
        comp_dir = kagglehub.competition_download(cfg.dataset.comp_dataset)
    except Exception:
        comp_dir = kagglehub.dataset_download(cfg.dataset.comp_dataset)

    if cfg.infer_on_train:
        print("Warning: using train.parquet as valid.parquet for inference")
        valid_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))

    else:
        valid_df = pd.read_parquet(os.path.join(input_dir, "valid.parquet"))

    if cfg.use_tta:
        valid_df_tta = valid_df.copy()
        valid_df_tta["content_ids"] = valid_df_tta["content_ids"].apply(lambda x: x[::-1])
        valid_df_tta["MisconceptionNameList"] = valid_df_tta["MisconceptionNameList"].apply(lambda x: x[::-1])

        valid_df = pd.concat([valid_df, valid_df_tta]).reset_index(drop=True)

    # For few-shot examples
    comp_df = pd.read_csv(os.path.join(comp_dir, "train.csv")).rename(columns={"QuestionId": "query_id"})
    content_df = pd.read_csv(os.path.join(comp_dir, "misconception_mapping.csv"))
    id2name = dict(zip(content_df["MisconceptionId"], content_df["MisconceptionName"]))

    fs_df, _ = train_valid_split(cfg, comp_df)
    query_df, fs_corr_df, content2query = eedi_process_df(fs_df, content_df)
    fs_query2content = dict(zip(fs_corr_df["query_id"], fs_corr_df["content_id"]))

    label_dir = kagglehub.dataset_download(cfg.dataset.label_dataset)
    label_df = pd.read_csv(os.path.join(label_dir, "train.csv"))
    _, corr_df_all, _ = eedi_process_df(label_df)

    query_df["demo"] = query_df.apply(lambda x: format_example(x, id2name, fs_query2content), axis=1)
    query2example = dict(zip(query_df["query_id"], query_df["demo"]))

    valid_df = add_fs_examples(valid_df, content2query, query2example, rng)  # .head(16)

    qa2mc = dict(zip(corr_df_all["query_id"], corr_df_all["content_id"]))  # label map (qa_id -> mc_id)

    print(f"shape of validation data: {valid_df.shape}")
    print_line()

    dataset_creator = RankerDataset(cfg)

    # valid_df["cot_32b"] = ""
    valid_ds = dataset_creator.get_dataset(valid_df)

    tokenizer = dataset_creator.tokenizer
    valid_ds = valid_ds.sort("length", reverse=True)

    a_tok_id = tokenizer("A", add_special_tokens=False)["input_ids"][-1]
    b_tok_id = tokenizer("B", add_special_tokens=False)["input_ids"][-1]
    c_tok_id = tokenizer("C", add_special_tokens=False)["input_ids"][-1]
    d_tok_id = tokenizer("D", add_special_tokens=False)["input_ids"][-1]
    e_tok_id = tokenizer("E", add_special_tokens=False)["input_ids"][-1]

    print(f">> EediRanker: A token id: {a_tok_id}")
    print(f">> EediRanker: B token id: {b_tok_id}")
    print(f">> EediRanker: C token id: {c_tok_id}")
    print(f">> EediRanker: D token id: {d_tok_id}")
    print(f">> EediRanker: E token id: {e_tok_id}")

    valid_ds = valid_ds.map(lambda example: {"prompt": tokenizer.decode(example["input_ids"], skip_special_tokens=False)})
    # valid_ds = valid_ds.sort("prompt", reverse=True)

    valid_qa_ids = valid_ds["query_id"]
    valid_mc_ids = valid_ds["content_ids"]

    prompts = valid_ds["prompt"]

    print(f"# of requests: {len(prompts)}")
    print(f"Example:\n\n{prompts[0]}")
    print("data preparation done...")
    print_line()

    llm = vllm.LLM(
        cfg.model.backbone_path,
        quantization=cfg.model.quantization,
        tensor_parallel_size=cfg.num_gpus,
        gpu_memory_utilization=0.99,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        enable_prefix_caching=True,
        max_num_seqs=20,
    )

    # ------- Prepare -------------------------------------------------------------------#
    sampling_params = vllm.SamplingParams(n=1, top_p=0.8, logprobs=20, max_tokens=1, temperature=0.0, skip_special_tokens=False)
    responses = llm.generate(prompts, sampling_params, use_tqdm=True)

    print("inference done...")
    print_line()

    QuestionId_Answer = []
    MisconceptionId = []
    scores = []

    for qid, cids, response in zip(valid_qa_ids, valid_mc_ids, responses):
        logprob_dict = response.outputs[0].logprobs[0]

        top_tok_ids = set(list(logprob_dict.keys()))
        if len(top_tok_ids.intersection(set([a_tok_id, b_tok_id, c_tok_id, d_tok_id, e_tok_id]))) == 0:
            print(f"Bad Output for {qid}")
            continue

        a_logit, b_logit, c_logit, d_logit, e_logit = -10.0, -10.0, -10.0, -10.0, -10.0

        if a_tok_id in logprob_dict:
            a_logit = logprob_dict[a_tok_id].logprob

        if b_tok_id in logprob_dict:
            b_logit = logprob_dict[b_tok_id].logprob

        if c_tok_id in logprob_dict:
            c_logit = logprob_dict[c_tok_id].logprob

        if d_tok_id in logprob_dict:
            d_logit = logprob_dict[d_tok_id].logprob

        if e_tok_id in logprob_dict:
            e_logit = logprob_dict[e_tok_id].logprob

        logits = np.array([a_logit, b_logit, c_logit, d_logit, e_logit])
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        normalized_scores = exp_logits / np.sum(exp_logits)

        QuestionId_Answer.append(qid)
        MisconceptionId.append(cids)
        scores.append(normalized_scores)

    result_df = pd.DataFrame()
    result_df["QuestionId_Answer"] = QuestionId_Answer
    result_df["MisconceptionId"] = MisconceptionId
    result_df["MisconceptionId"] = result_df["MisconceptionId"].apply(lambda x: [str(y) for y in x])
    result_df["score"] = scores
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    result_df.to_parquet(os.path.join(cfg.outputs.model_dir, "tmp.parquet"), index=False)

    if cfg.use_tta:
        result_df = result_df.explode(["MisconceptionId", "score"]).reset_index(drop=True)
        result_df = result_df.groupby(["QuestionId_Answer", "MisconceptionId"]).agg({"score": "mean"}).reset_index()

        # regroup
        agg_df = result_df.groupby("QuestionId_Answer")["MisconceptionId"].agg(list).reset_index()
        score_agg_df = result_df.groupby("QuestionId_Answer")["score"].agg(list).reset_index()
        agg_df = pd.merge(agg_df, score_agg_df, on="QuestionId_Answer", how="left")
        result_df = agg_df.copy()

    eval_response = run_evaluation(result_df, qa2mc)

    lb = eval_response["lb"]
    ranker_auc = eval_response["auc"]

    result_df = eval_response["result_df"]
    oof_df = eval_response["oof_df"]

    os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    oof_df.to_csv(os.path.join(cfg.outputs.model_dir, "ext_oof_df_best.csv"), index=False)
    result_df.to_csv(os.path.join(cfg.outputs.model_dir, "ext_result_df_best.csv"), index=False)

    print(f">>> Current LB (MAP@25) = {round(lb, 4)}")
    print(f">>> AUC = {round(ranker_auc, 4)}")


if __name__ == "__main__":
    main()
