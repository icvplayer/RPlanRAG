import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass, field, asdict
from argparse import ArgumentParser
from tqdm import tqdm
from freestyleRAG.freestylerag import freestyleRAG, always_get_an_event_loop
from freestyleRAG.llm import ollama_model_complete, ollama_embedding, openai_complete_if_cache, ollama_model_if_cache
from freestyleRAG.utils import EmbeddingFunc, PlanningModelConfig
from freestyleRAG.baseClasses import QueryParameters
from freestyleRAG import prompt_template
import time
import json
from accelerate import Accelerator
from transformers import AutoTokenizer
from functools import partial
import datasets
from collections import defaultdict
from dataset_process.data_utils import DefaultDataCollator, makedirs, scorer, DATASET2CATEGORY
from torch.utils.data import DataLoader
from transformers.utils import logging
import asyncio
import re
logger = logging.get_logger(__name__)

cls = "hotpotqa"


def append_to_json(data, file_path):

    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        existing_data = []
    
   
    if not isinstance(existing_data, list):
        print("Existing data is not a list. Converting to list.")
        existing_data = []
    
    
    existing_data.append(data)
    
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)


def read_json_line_by_line(file_path):
    try:
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                print("The loaded data is not a list.")
                yield data 
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

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
        **kwargs,
    )

async def openai_output(prompt):
    output = await llm_model_func(prompt=prompt)
    return output



def process_longbench(data, indices, tokenizer, max_length=100000, truncate_from_middle=True):
    outputs = {'context': [], 'question': [], "dataset": [], "index": [], "length": []}

    for input, context, dataset, index in zip(data['input'], data['context'], data['dataset'], indices):
        if dataset.endswith("_e"):
            dataset = dataset[:-2]

        if dataset in ['nq_10', 'hotpotqa', '2wikimqa', 'musique']:
            question = input
        elif dataset == "gov_report":
            question = ""
        elif dataset == "multi_news":
            question = ""
        else:
            continue
        
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
    if args.use_planning:
        planning_model = PlanningModelConfig(args.planning_model_path)

    with accelerator.main_process_first():  
        process_fn = partial(  
            process_longbench, 
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
        dataset = datasets.Dataset.from_pandas(groupby_dataset.get_group(dataset_name), preserve_index=False)  # 获得数据集中该组的数据，之后将pandas数据转换为huggingface数据
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

        processed_index = 0
        open_once = True
        for i, x in enumerate(tqdm(dataloader, desc="Generating")):
            processed_num_path = f"training_summarize_data/{cls}_processed.json"
            if open_once:
                if os.path.exists(processed_num_path):
                    with open(processed_num_path, "r") as file:
                        processed_index = json.load(file)
                else:
                    processed_index = 0
                open_once = False
            
            if processed_index != 0:
                processed_index -= 1
                print(f"this step {i} is already processed, continue.")
                continue

            x.pop("dataset")
            index = x.pop("index")[0]
            context=x["context"][0]
            query=x["question"][0]

            working_dir = os.path.join(args.working_dir, f"{i+1}")
            
            # rag = freestyleRAG(working_dir=working_dir, llm_model_func=ollama_model_complete, 
            #     llm_model_name=args.gen_llm_name, llm_model_max_async=args.llm_model_max_async,
            #     llm_model_max_token_size=args.llm_model_max_token_size, 
            #     llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}, "timeout":args.time_out},
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

            q_param = QueryParameters(only_need_context=True, top_k=60, task_name="qa", no_keywords=True, need_training_summary_return=False)
            
            if args.use_planning:
                output = rag.query_with_plan(query, q_param, planning_model)
            else:
                if q_param.need_training_summary_return:
                    output1, output2 = rag.query_with_origional_keywords(query, q_param)
                output = rag.query_with_origional_keywords(query, q_param)

            if accelerator.num_processes > 1:
                # pad across device to the same length
                if q_param.need_training_summary_return:
                    output1 = accelerator.gather_for_metrics(output1)
                    output2 = accelerator.gather_for_metrics(output2)
                output = accelerator.gather_for_metrics(output)
                index = accelerator.gather_for_metrics(index)

            index = index.tolist()

            if accelerator.process_index == 0:
                # if isinstance(output, list):
                #     preds.extend(output)
                # preds.append(output)
                if isinstance(index, list):
                    indices.extend(index)
                else:
                    # single process
                    indices.append(index)

            if accelerator.process_index == 0:
                raw_dataset_subset = raw_dataset[indices]
                answers = raw_dataset_subset["answers"][-1]  
            if isinstance(answers, list):
               answers = ", ".join(map(str, answers))

            if "?" not in query:
                query += "?"

            loop = always_get_an_event_loop()
            if q_param.need_training_summary_return:
                train_promt1 = prompt_template.PROMPTS["training_summary_task1"]
                train_promt2 = prompt_template.PROMPTS["training_summary_task2"]
                format_prompt1 = train_promt1.format(input_question=query, input_data_table=output1, input_answer=answers)
                result1 = loop.run_until_complete(ollama_model_if_cache(model=args.gen_llm_name, prompt=format_prompt1, system_prompt=None, history_messages=[]))
                format_prompt2 = train_promt2.format(input_question=query, input_data_table=output1, input_answer=answers)
                result2 = loop.run_until_complete(ollama_model_if_cache(model=args.gen_llm_name, prompt=format_prompt2, system_prompt=None, history_messages=[]))
                result = result1 + result2

            else:
                train_promt = prompt_template.PROMPTS["training_summary"]
                format_prompt = train_promt.format(input_question=query, input_data_table=output, input_answer=answers)
                # result = loop.run_until_complete(openai_output(format_prompt))
                result = loop.run_until_complete(ollama_model_if_cache(model=args.gen_llm_name, prompt=format_prompt, system_prompt=None, history_messages=[]))

                pattern = r'(-----Relationships-----.*?)(?=<\|end\|>)'
                match = re.search(pattern, result, re.DOTALL)
                if match:
                    match = match.group(1).strip()

                    if "Relationships" in match and "Sources" in match:
                        save_format = {
                            "instruction":format_prompt,
                            "input":"",
                            "output":match
                        }
                        append_to_json(save_format, f"training_summarize_data/summary_training_data_{cls}.json")

                with open(processed_num_path, "w") as file:
                    json.dump(i+1, file)



if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("--working_dir", type=str, default=f"./mydata_results/{cls}", help="The path to storage results")
    parser.add_argument("--eval_data", type=str, default="./training_summarize_data/my_training_data.json", help="The path of the dataset")
    parser.add_argument("--tokenizer_path", type=str, default="../models/qwen2.5-7B", help="The tokenizer path")
    parser.add_argument("--max_length", type=str, default=None, help="Max input length")
    parser.add_argument("--truncate_from_middle", type=str, default=True, help="Truncate inputs from the middle")
    parser.add_argument("--dataset_names", type=str, nargs='+', default=[f"{cls}"], help="Which dataset to evaluate?")
    parser.add_argument("--gen_llm_name", type=str, default="qwen2.5:7b-instruct-m", help="The model name which is used to generate triples")
    parser.add_argument("--embedding_model_name", type=str, default="bge-m3", help="The embedding model name")
    parser.add_argument("--llm_model_max_async", type=int, default=4, help="Add restriction of maximum async calling times for a async func")
    parser.add_argument("--llm_model_max_token_size", type=int, default=32768, help="The maximum tokensize of the generating model")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="The embedding dim of the embedding model")
    parser.add_argument("--max_token_size", type=int, default=8192, help="The maximum tokensize of the embedding model")
    parser.add_argument("--time_out", type=int, default=250, help="The maximum request time of the ollama client")
    parser.add_argument("--one_client", type=bool, default=False, help="Using ollama client pool or not")
    parser.add_argument("--client_size", type=int, default=4, help="Number of ollama clients")
    parser.add_argument("--use_planning", type=bool, default=False, help="Wether to use question planning")
    parser.add_argument("--planning_model_path", type=str, default="../models/qwen2.5-7b-instruct-plan", help="The path of the planning model")

    args = parser.parse_args()

    main(args)