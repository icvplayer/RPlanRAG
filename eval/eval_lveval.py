import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass, field, asdict
from argparse import ArgumentParser
from tqdm import tqdm
from freestyleRAG.freestylerag import freestyleRAG
from freestyleRAG.llm import ollama_model_complete, ollama_embedding, openai_complete_if_cache
from freestyleRAG.utils import EmbeddingFunc, PlanningModelConfig, seed_everything
from freestyleRAG.baseClasses import QueryParameters
import time
import json
from accelerate import Accelerator
from transformers import AutoTokenizer
from functools import partial
import datasets
from collections import defaultdict
from dataset_process.data_utils import DefaultDataCollator, makedirs, scorer, DATASET2CATEGORY, acc_score
from torch.utils.data import DataLoader
from transformers.utils import logging
logger = logging.get_logger(__name__)

# hotpotwikiqa_mixup_16k
cls = "hotpotwikiqa_mixup_16k"

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "Qwen/Qwen2.5-7B-Instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("API_KEY"),
        base_url="your url",
        max_new_tokens=32,
        **kwargs,
    )

def process_lveval(data, indices, tokenizer, max_length=100000, truncate_from_middle=True):
    outputs = {'context': [], 'question': [], "dataset": [], "index": [], "length": []}

    for input, context, dataset, index in zip(data['input'], data['context'], data['dataset'], indices):

        question = input
        
        if max_length is not None:
            if truncate_from_middle:
                try:
                    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
                except:
                    tokenized_context = tokenizer.encode(context)
                if len(tokenized_context) > max_length:
                    half = int(max_length / 2)
                    context = tokenizer.decode(tokenized_context[:half]) + tokenizer.decode(tokenized_context[-half:])
            else:
                tokenized_context = tokenizer.encode(context)
                context = tokenizer.decode(tokenized_context[-max_length:])

        length = len(tokenizer.encode(context))

        outputs["context"].append(context)
        outputs["question"].append(question)
        outputs["dataset"].append(dataset)
        outputs["index"].append(index)
        outputs["length"].append(length)

    return outputs


def insert_text(rag, unique_contexts):

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.construct_graph(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")


def main(args):

    accelerator = Accelerator(cpu=False)

    cache_dir = os.path.join(args.working_dir, "tokenizer_cache")
    tokenizer_kwargs = {
            "cache_dir": cache_dir,
            "padding_side": "left",
            "trust_remote_code": True,
        }

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, **tokenizer_kwargs)

    task_name = "qa"
    # if cls not in ["hotpotqa", "2wikimqa", "musique"]:
    #     args.planning_model_path = None
        
    planning_model = None
    if args.use_planning and args.planning_model_path:
        planning_model = PlanningModelConfig(args.planning_model_path)
        
    summary_model = None
    if args.use_summary and args.summary_model_path:
        summary_model = PlanningModelConfig(args.summary_model_path, max_length=16384)


    with accelerator.main_process_first():
        process_fn = partial(
            process_lveval, 
            tokenizer=tokenizer,
            max_length=args.max_length,
            truncate_from_middle=args.truncate_from_middle
        )
        raw_dataset = datasets.load_dataset("json", data_files=args.eval_data, split="train")
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, with_indices=True, remove_columns=raw_dataset.column_names)
    groupby_dataset = dataset.to_pandas().groupby("dataset")

    metrics = {}
    if args.dataset_names is None:
        dataset_names = [key for key, _ in groupby_dataset]
    else:
        dataset_names = args.dataset_names  
    
    for i, dataset_name in enumerate(dataset_names):  
        if accelerator.process_index == 0:
            logger.info(f"Evaluating {dataset_name} ({i + 1} / {len(dataset_names)})...")
        result_path = os.path.join(args.working_dir, f"{dataset_name}.json")
        dataset = datasets.Dataset.from_pandas(groupby_dataset.get_group(dataset_name), preserve_index=False)  
        data_collator = DefaultDataCollator(padding_side="left")
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            collate_fn=data_collator,
            # only pin memory when no gpu
        )

        # NOTE: prepare dataloader so the data moves to GPU automatically
        dataloader = accelerator.prepare(dataloader)  
        indices = []
        preds = []

        num_of_commars = 0
        wrong_answer_num_of_commar = 0
        all_num_sub_q = 0

        for i, x in enumerate(tqdm(dataloader, desc="Generating")):
            x.pop("dataset")
            index = x.pop("index")[0]
            context=x["context"][0]
            query=x["question"][0]

            working_dir = os.path.join(args.working_dir, f"{i+1}")
            
            # rag = freestyleRAG(working_dir=working_dir, llm_model_func=ollama_model_complete, 
            #     llm_model_name=args.gen_llm_name, llm_model_max_async=args.llm_model_max_async,
            #     llm_model_max_token_size=args.llm_model_max_token_size, 
            #     llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 16384}, "timeout":args.time_out},
            #     embedding_func=EmbeddingFunc(
            #         embedding_dim=args.embedding_dim,
            #         max_token_size=args.max_token_size,
            #         func=lambda texts: ollama_embedding(texts, embed_model=args.embedding_model_name, host="http://localhost:11434"),                           
            #     ),
            #     one_client=args.one_client,
            #     client_size=args.client_size)

            rag = freestyleRAG(working_dir=working_dir, llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=args.embedding_dim,
                    max_token_size=args.max_token_size,
                    func=lambda texts: ollama_embedding(texts, embed_model="bge-m3", host="http://localhost:11434"),                           
                ),
                one_client=args.one_client,
                client_size=args.client_size)

            insert_text(rag, unique_contexts=context)

            q_param = QueryParameters(top_k=60, task_name=task_name)
            
            if args.use_planning:
                num_commars = None
                output, _ = rag.query_with_plan(query, q_param, planning_model, summary_model)
                if num_commars:
                    num_of_commars += 1
            else:
                output = rag.query_with_origional_keywords(query, q_param)

            if accelerator.num_processes > 1:
                # pad across device to the same length
                output = accelerator.gather_for_metrics(output)
                index = accelerator.gather_for_metrics(index)

            accelerator.print(output)

            index = index.tolist()

            if accelerator.process_index == 0:
                if isinstance(output, list):
                    preds.extend(output)
                preds.append(output)
                if isinstance(index, list):
                    indices.extend(index)
                else:
                    # single process
                    indices.append(index)

            if accelerator.process_index == 0:
                raw_dataset_subset = raw_dataset[indices]
                answers = raw_dataset_subset["answers"]  
                lengths = raw_dataset_subset["length"]
                all_classes = []
                # all_classes = raw_dataset_subset["all_classes"][0]
                score = scorer(dataset_name, preds, answers, all_classes)
                acc_score_ = acc_score(preds, answers) 

                if num_commars:
                    all_num_sub_q += 1

                    pre = output
                    if isinstance(answers, list):
                        ans = answers[i][0]
                    else:
                        ans = answers
                    if pre.lower() != ans.lower():
                        wrong_answer_num_of_commar += 1
                        print("#######################")
                        print(f"Wrong num: ", wrong_answer_num_of_commar)
                        print('all_num: ', all_num_sub_q)
                        print("#######################")
                
                acc_score_name = dataset_name + "_acc"
                logger.info(f"{dataset_name}: {score}")
                logger.info(f"{acc_score_name}: {acc_score_}")
                metrics[dataset_name] = score
                metrics[acc_score_name] = acc_score_
                
                with open(makedirs(result_path), "w", encoding="utf-8") as f:
                    f.write(json.dumps(score, ensure_ascii=False) + "\n")
                    f.write(json.dumps(acc_score_, ensure_ascii=False) + "\n")
                    for index, pred in zip(indices, preds):
                        sample = raw_dataset[index]
                        del sample["context"]
                        sample["pred"] = pred
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if accelerator.process_index == 0:
        # compute category score
        category_metrics = defaultdict(list)
        for dataset, metric in metrics.items():
            category = DATASET2CATEGORY[dataset]
            category_metrics[category].append(metric)
        for k, v in category_metrics.items():
            # when evaluating on longbench_e, each metric is a dict of float
            if isinstance(v[0], dict):
                category_metric = {}
                for kk in v[0].keys():
                    vv = [v[j][kk] for j in range(len(v))]
                    category_metric[kk] = round(sum(vv) / len(vv), 2)
                category_metrics[k] = category_metric
            else:
                category_metrics[k] = round(sum(v) / len(v), 2)
        
        # compute average score
        if isinstance(next(iter(metrics.values())), dict):
            avg = defaultdict(list)
            for k, v in metrics.items():
                for kk, vv in v.items():
                    avg[kk].append(vv)
            for k, v in avg.items():
                avg[k] = round(sum(v) / len(v), 2)
        else:
            avg = round(sum(metrics.values()) / len(metrics), 2) 
        metrics["avg"] = avg

        accelerator.print(metrics)
        with open(os.path.join(args.working_dir, "metrics.jsonl"), "a") as f:
            save_args = vars(args)
            save_args["metrics"] = metrics
            save_args["category_metrics"] = category_metrics
            f.write(json.dumps(save_args)+"\n")    


if __name__ == "__main__":

    seed_everything(66)

    parser = ArgumentParser()
    parser.add_argument("--working_dir", type=str, default=f"./lveval_results/{cls}_longbench", help="The path to storage results")
    parser.add_argument("--eval_data", type=str, default=f"../data/hotpotwikiqa_mixup_16k.jsonl", help="The path of the dataset")
    parser.add_argument("--tokenizer_path", type=str, default="../models/qwen2.5-7B", help="The tokenizer path")
    parser.add_argument("--max_length", type=str, default=None, help="Max input length")
    parser.add_argument("--truncate_from_middle", type=str, default=True, help="Truncate inputs from the middle")
    parser.add_argument("--dataset_names", type=str, nargs='+', default=[f"{cls}"], help="Which dataset to evaluate?")
    parser.add_argument("--gen_llm_name", type=str, default="qwen2.5-14b-i", help="The model name which is used to generate triples")
    parser.add_argument("--embedding_model_name", type=str, default="bge-m3", help="The embedding model name")
    parser.add_argument("--llm_model_max_async", type=int, default=4, help="Add restriction of maximum async calling times for a async func")
    parser.add_argument("--llm_model_max_token_size", type=int, default=32768, help="The maximum tokensize of the generating model")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="The embedding dim of the embedding model")
    parser.add_argument("--max_token_size", type=int, default=8192, help="The maximum tokensize of the embedding model")
    parser.add_argument("--time_out", type=int, default=250, help="The maximum request time of the ollama client")
    parser.add_argument("--one_client", type=bool, default=False, help="Using ollama client pool or not")
    parser.add_argument("--client_size", type=int, default=4, help="Number of ollama clients")
    parser.add_argument("--use_planning", type=bool, default=True, help="Wether to use question planning")
    parser.add_argument("--planning_model_path", type=str, default="../models/qwen2.5-7b-instruct-planv3", help="The path of the planning model")
    parser.add_argument("--use_summary", type=bool, default=True, help="Wether to use summary model")
    parser.add_argument("--summary_model_path", type=str, default="../models/qwen2.5-7b-summary-RL", help="The path of the summary model")

    args = parser.parse_args()

    main(args)