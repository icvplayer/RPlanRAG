import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_MODE"] = "offline"
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
PatchFastRL("GRPO", FastLanguageModel)
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import random
from collections import Counter
import string
from peft import LoraConfig, get_peft_model, PeftModel
from sentence_transformers import SentenceTransformer, util

model_1 = SentenceTransformer('./models/all-MiniLM-L6-v2')


max_seq_length = 8000
lora_rank = 32
# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format. Please note that there should be no symbols before "Because" and after "<|end|>".
Because ...
...
Therefore ...
...
Final answer: <|start|>...<|end|>"""


def extract_custom_answer(text: str) -> str:
    if "<|start|>" in text and "<|end|>" in text:
        start_idx = text.find("<|start|>") + len("<|start|>")
        end_idx = text.find("<|end|>")
        return text[start_idx:end_idx].strip()
    return ''

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_custom_questions(file_path: str) -> Dataset:
    data = load_dataset("json", data_files=file_path, split='train') # type: ignore
    data = data.shuffle(seed=42)
        # 定义一个函数来随机决定是否删除所有 "Passage i" 部分
    def remove_all_passages(text: str) -> str:
        
        your_turn_idx = text.find("#####Your turn: #####")
        if your_turn_idx == -1:
            return text
        
        
        before, after = text[:your_turn_idx], text[your_turn_idx:]
        
        
        passages = re.findall(r'Passage \d+:', after)
        if not passages:
            return text
        
     
        if random.random() < 0.5:  
            for passage in passages:
                after = after.replace(passage, '')
        
       
        return before + after
    

    def process_data(examples):

        prompts = []
        answers = []

        if isinstance(examples["instruction"], list):    
            for idx in range(len(examples["instruction"])):
                instruction = examples["instruction"][idx]
                output_text = examples["output"][idx]
                
                instruction = remove_all_passages(instruction)

                pattern = r"#####Here is an example: #####.*?#####Your turn: #####"
                instruction_ = re.sub(pattern, "#####Your turn: #####", instruction, flags=re.DOTALL)
                instruction = instruction_

                prompt = [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': instruction}
                ]
                

                answer = extract_custom_answer(output_text)
                
                prompts.append(prompt)
                answers.append(answer)
                return {'prompt': prompts, 'answer': answers}
        else:
            instruction = examples["instruction"]
            output_text = examples["output"]
            instruction = remove_all_passages(instruction)

            pattern = r"#####Here is an example: #####.*?#####Your turn: #####"
            instruction_ = re.sub(pattern, "#####Your turn: #####", instruction, flags=re.DOTALL)
            instruction = instruction_

            prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': instruction}
            ]       

            answer = extract_custom_answer(output_text)
            return {'prompt': prompt, 'answer': answer}
        
    def remove_instruction_key(example):
        if "instruction" in example:
            del example["instruction"]
        if "output" in example:
            del example["output"]
        if "input" in example:
            del example["input"]
        return example
    # 应用处理函数
    data = data.map(process_data)
    # data = data.map(remove_instruction_key)
    return data  # type: ignore

dataset = get_custom_questions("./data/Summary_training_data_5000_clean.json")

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)  

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):  
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)  
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 2)

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)  
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_custom_answer(r) for r in responses]
    print("=============")
    print(extracted_responses)
    print(answer)
    print("=============")
    # print('-'*20, f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [1.5 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# def int_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_custom_answer(r) for r in responses]
#     total_score = []
#     for (prediction, ground_truth) in zip(extracted_responses, answer):
#         score = 0.
#         score = max(score, qa_f1_score(prediction, ground_truth)) * 2.0
#         total_score.append(score)
#     return total_score 

def acc_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_custom_answer(r) for r in responses]
    # print('-'*20, f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [0.5 if (r in a or a in r) else 0.0 for r, a in zip(extracted_responses, answer)]

def semantic_similarity(a, b):
    embeddings = model_1.encode([a, b], convert_to_tensor=True)
    similar = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similar

def similar_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_custom_answer(r) for r in responses]
    rewards = []
    for r, a in zip(extracted_responses, answer):
        sim = round(semantic_similarity(r, a),2)
        if sim:
            if sim > 0.6:
                rewards.append(sim)
            else:
                rewards.append(0)
        else:
            rewards.append(0)
    return rewards
    
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^Because.*?Therefore.*?\nFinal answer: <|start|>.*?<|end|>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r ,re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"Because.*?Therefore.*?<|start|>.*?<|end|>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r ,re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("Because") >= 1:
        count += 0.125
    if text.count("Therefore") >= 1:
        count += 0.125
    if text.count("<|start|>") == 1:
        count += 0.125
    if text.count("<|end|>") == 1:
        count += 0.125  
    if text.count("<|start|>") >= 1:
        count -= 0.125
    if text.count("<|end|>") >= 1:
        count -= 0.125
    if len(text.split("<|end|>")[-1]) -1 > 0:
        count -= 0.125
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# the model that has been trained with SFT
model_name = "./models/qwen2.5-7b-instruct-summary"

output_dir="./models/qwen2.5-7b-summary-RLv3"
run_name="qwen2.5-7b-summary"

lora_config = LoraConfig(
    base_model_name_or_path=model_name,
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map=None
# ).to("cuda")

# model = get_peft_model(model, lora_config)


training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=3e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    optim = "paged_adamw_8bit",
    logging_steps=5,
    # bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=6,
    max_prompt_length=max_seq_length,
    max_completion_length=512,
    # num_train_epochs=1,
    max_steps = 400,
    save_steps=50,
    max_grad_norm=0.1,
    log_on_each_node=False,
    # use_vllm=True,
    # vllm_gpu_memory_utilization=.3,
    # vllm_device="cuda:1",
    report_to="wandb"
)


# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token


# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        similar_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
    #peft_config=peft_config
)
# trainer.train(resume_from_checkpoint=True)
trainer.train()


print(">>>>>>>>>>>>>>>model.save_lora:")
# trainer.save_model(output_dir)

print(">>>>>>>>>>>>>>>model.save_lora:")
model.save_pretrained_merged("./model_merged2", tokenizer, save_method = "merged_16bit",)