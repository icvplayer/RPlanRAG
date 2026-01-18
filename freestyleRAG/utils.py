import os
import re
import json
import html
import io
import csv
import logging
import tiktoken
import numpy as np
import asyncio
from typing import List, Any
from functools import wraps
from hashlib import md5
from dataclasses import dataclass
from itertools import cycle
import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random


ENCODER = None


logger = logging.getLogger("freestylerag")


def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def temp_write_json(json_obj, file_name):

    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            json.dump({}, f)
    existing_data = {}
    with open(file_name, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
    existing_data.update(json_obj)

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        # If the maximum number of calls is exceeded, a wait is required
        @wraps(func)  # The decorator simply does not lose the meta information of the decorated function
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_delimiters(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]

def split_string_when_redescription(content: str, markers: list[str]) -> list[str]:
    if not markers:
        return [content]
    
    if len(markers) != 2:
        return [content]
    
    parts_after_completion = content.split(markers[1])
    
    results = []
    for part in parts_after_completion:
        
        sub_parts = part.split(markers[0])
        results.extend([sub.strip() for sub in sub_parts if sub.strip()])
    
    return results


def clean_str(input) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input
        
    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]   # You can truncate it, because it's already sorted, and truncated shouldn't matter
    return list_data


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def list_of_list_to_csv(data: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


# ollama client pool
class OllamaClientPoolSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    async def initialize(self, host="localhost", timeout=15, pool_size=2):
        if not self._initialized:
            self.CLIENT_POOL = [
                ollama.AsyncClient(host=host, timeout=timeout) for _ in range(pool_size)
            ]
            self.CLIENT_CYCLE = cycle(self.CLIENT_POOL)
            self._initialized = True

    def get_next_client(self):
        if not self._initialized:
            raise ValueError("Client pool is not initialized. Call initialize() first.")
        return next(self.CLIENT_CYCLE)

client_pool_singleton = OllamaClientPoolSingleton()


class PlanningModelConfig:
    def __init__(self, model_path, max_length=8192):

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        self.path = model_path
        self.try_num = 0
        self.device = self.model.device
        self.max_length = max_length

    def timeout_handler(self, signum, frame):
        raise TimeoutError("Generation timed out")

    def process_question(self, prompt, system_prompt=None):

        if self.try_num == 5:
            self.try_num = 0
            return " \n"
        
        messages = []
        if system_prompt:
            system_prompt = system_prompt
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        input_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs_ids = self.tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
        inputs = {k: v.to(self.device) for k, v in inputs_ids.items()}
        import signal
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(250)
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs_ids, max_new_tokens=512, num_return_sequences=1, early_stopping=True)
                generated_text = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
                signal.alarm(0)
                return generated_text
        except TimeoutError:
            print("Generation timed out. Retrying...")
            self.try_num = self.try_num + 1
            return self.process_question(prompt=prompt)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def append_string_to_json(file_path, string_to_append):


    if os.path.exists(file_path):
        try:

            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                

            if not isinstance(data, list):

                data = []
        except json.JSONDecodeError:

            data = []
    else:

        data = []
    

    data.append(string_to_append)
    

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    