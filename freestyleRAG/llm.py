import numpy as np
import os
import ollama
import asyncio
import copy
import torch
from functools import lru_cache
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from freestyleRAG.baseClasses import BaseKVStorage
from freestyleRAG.utils import compute_args_hash, logger
from freestyleRAG.storageClasses import KVCacheProcess
from transformers import AutoTokenizer, AutoModelForCausalLM


log_lock = asyncio.Lock()
async def log_wrong_chunk(chunk_process_num):
    async with log_lock:
        logger.info('wrong_chunk_num: {}\n'.format(chunk_process_num))


async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    chunk_process_num=None,
    max_new_tokens=512,
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url, timeout=60)
    )
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # response = await openai_async_client.chat.completions.create(
    #     model=model, messages=messages, **kwargs
    # )
    attempt = 0
    MAX_RETRIES = 15
    while attempt < MAX_RETRIES:
        try:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages,**kwargs)
            if not response or not response.choices[0].message.content:
                raise ValueError("The value is None!")
            elif response.choices[0].message.content:
                return response.choices[0].message.content
        except Exception as e:
            print("llm resquest is failed, trying nummber: {}".format(attempt+1))
            attempt = attempt + 1
            if attempt < MAX_RETRIES:
                await asyncio.sleep(5)
                openai_async_client = (
                AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url, timeout=60)
                )
            else:
                response = ""
                print("Response time out, return None.")
                await log_wrong_chunk(chunk_process_num)
                return response

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


async def openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_model_if_cache(
    model, prompt, system_prompt=None, history_messages=[], use_ollama_client=None, chunk_process_num=0, **kwargs
) -> str:
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)
    host = kwargs.pop("host", "http://localhost:11434")
    timeout = kwargs.pop("timeout", 250)

    if not use_ollama_client:
        ollama_client = ollama.AsyncClient(host=host, timeout=timeout)
    else:
        ollama_client = use_ollama_client.get_next_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)  # get an unique hash id

        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
        
    attempt = 0
    MAX_RETRIES = 10
    result = ""
    while attempt < MAX_RETRIES:
        try:
            response = await ollama_client.chat(model=model, messages=messages, **kwargs)
            result = response["message"]["content"]
            if hashing_kv is not None:
                await hashing_kv.upsert({args_hash: {"return": result, "model": model}})

            return result
        except Exception as e:
            logger.info("ollama llm resquest is failed, trying nummber: {}".format(attempt+1))
            ollama_client = ollama.AsyncClient(host=host, timeout=timeout)
            attempt = attempt + 1
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2+attempt)
            else:
                print("Response time out, return None.")
                await log_wrong_chunk(chunk_process_num)
                result = ""
                return result
    return result

async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def ollama_embedding(texts: list[str], embed_model, embed_client = None, **kwargs) -> np.ndarray:
    embed_text = []

    if not embed_client:
        embed_client = ollama.Client(**kwargs)

    attempt = 0
    MAX_RETRIES = 5
    while attempt < MAX_RETRIES:
        try:
            for text in texts:
                data = embed_client.embeddings(model=embed_model, prompt=text)
                embed_text.append(data["embedding"])

            return embed_text
        except Exception as e:
            print("ollama embedding resquest is failed, trying nummber:{}".format(attempt))
            embed_client = ollama.Client(**kwargs)
            attempt = attempt + 1
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1)
            else:
                print("Response time out, return None.")
                embed_text = None
                return embed_text

    return embed_text


@lru_cache(maxsize=1)
def initialize_hf_model(model_name):
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    return hf_model, hf_tokenizer


async def hf_generate(llm_name, input_ids,
            max_new_tokens=512, early_stopping=True,
            position_ids=None,past_key_values=None, eos_tokenizerid=None):
    
    from transformers.generation.logits_process import TopPLogitsWarper
    logits_processor = TopPLogitsWarper(1.0)
    output_ids = input_ids.tolist()
    new_output_ids = list()
    position_offset = max(position_ids.tolist()[0]) + 1
    new_token_id = 0
    cache = None

    llm, tokenizer = initialize_hf_model(llm_name)
    device = llm.device

    with torch.no_grad():
        for i in range(max_new_tokens):

            if cache is None:

                output = llm(
                    input_ids,
                    position_ids=position_ids,
                    early_stopping=early_stopping,
                    past_key_values=past_key_values,
                    use_cache=True)
                
                logits = output.logits
                cache = output.past_key_values
                output = None
            else:
                token_ids = torch.tensor([[new_token_id]], device=device, dtype=torch.long)
                token_position_ids = torch.tensor([[position_offset + i]], device=device, dtype=torch.long)

                output = llm(
                    token_ids,
                    position_ids=token_position_ids,
                    early_stopping=early_stopping,
                    past_key_values=cache,
                    use_cache=True)
                
                logits = output.logits
                cache = output.past_key_values
                output = None


            tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
            probs = torch.softmax(last_token_logits, dim=-1)
            new_token_id = int(torch.multinomial(probs, num_samples=1))
            output_ids.append(new_token_id)
            new_output_ids.append(new_token_id)

            if new_token_id == eos_tokenizerid:
                break

    generated_text = tokenizer.decode(new_output_ids, skip_special_tokens=True)            
    return generated_text


async def hf_model_if_cache(
    model, prompt, system_prompt=None, history_messages=[], 
    prompt_cache_manage: KVCacheProcess=None, **kwargs
) -> str:

    if prompt_cache_manage != None:
        prompt_cache_length = prompt_cache_manage.prompt_length
        if prompt_cache_length > 0:
            prompt = prompt[prompt_cache_length:]

    model_name = model
    hf_model, hf_tokenizer = initialize_hf_model(model_name)  # loading tokenizer and model
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)  # computing hashing id
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    input_prompt = ""
    try:
        input_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]["role"] == "system":
                messages[1]["content"] = (
                    "<system>"
                    + messages[0]["content"]
                    + "</system>\n"
                    + messages[1]["content"]
                )
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for msgid in range(len_message):
                input_prompt = (
                    input_prompt
                    + "<"
                    + ori_message[msgid]["role"]
                    + ">"
                    + ori_message[msgid]["content"]
                    + "</"
                    + ori_message[msgid]["role"]
                    + ">\n"
                )

    input_ids = hf_tokenizer(
        input_prompt, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}

    if prompt_cache_manage:
        if prompt_cache_manage.kv_cache != None:

            new_position_ids = list(range(prompt_cache_manage.prompt_token_length, 
                                        prompt_cache_manage.prompt_token_length + input_ids['input_ids'].shape[1]))
            new_position_ids = torch.tensor([new_position_ids], device=hf_model.device, dtype=torch.long)

            # new_attention_mask=torch.ones(1, prompt_cache_manage.prompt_token_length + input_ids['input_ids'].shape[1]).to('cuda')

            response_text = await hf_generate(model_name, input_ids=input_ids['input_ids'], max_new_tokens=512, early_stopping= True, 
                        position_ids=new_position_ids, past_key_values=prompt_cache_manage.kv_cache,
                        eos_tokenizerid=hf_tokenizer.eos_token_id)
    else:
        output = hf_model.generate(
            **input_ids, max_new_tokens=32, num_return_sequences=1, early_stopping=True
        )

        response_text = hf_tokenizer.decode(
            output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": response_text, "model": model}})
    return response_text


async def hf_model_complete(
    prompt, system_prompt=None, history_messages=[], prompt_cache_manage: KVCacheProcess=None, **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await hf_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        prompt_cache_manage=prompt_cache_manage,
        **kwargs,
    )

async def hf_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

async def hf_model_kv_cache(prompt, system_prompt=None, **kwargs):
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    hf_model, hf_tokenizer = initialize_hf_model(model_name)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
