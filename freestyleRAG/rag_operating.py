import os
import json
import asyncio
import time
import re
import ast
from .baseClasses import (TextChunkSchema, BaseGraphStorage, BaseKVStorage, 
                          BaseVectorStorage, QueryParameters)
from .prompt_template import PROMPTS
from .utils import (encode_string_by_tiktoken, decode_tokens_by_tiktoken,
                    pack_user_ass_to_openai_messages, split_string_by_delimiters,
                    clean_str, logger, compute_mdhash_id, truncate_list_by_token_size,
                    is_float_regex, list_of_list_to_csv, split_string_when_redescription,
                    PlanningModelConfig)
from .storageClasses import KVCacheProcess
from collections import Counter, defaultdict
from .temporary_storage import TemporarySave
import copy
from dateutil import parser

class PromptManage:
    def __init__(self):

        self.entity_extract_prompt = PROMPTS["triples_extraction"]
        self.context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"], 
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"], 
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"])
        self.continue_prompt = PROMPTS["triples_continue_extraction"]
        self.regenerate_desc_prompt = PROMPTS["regenerate_description"]


GRAPH_FIELD_SEP = "<SEP>"

def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def handle_extracted_single_relationship(one_record: list, chunk_key: str):

    if len(one_record) < 5 or one_record[0] != '"relationship"':
        return None
    
    source = clean_str(one_record[1].upper())
    target = clean_str(one_record[2].upper())
    edge_description = clean_str(one_record[3])
    edge_source_id = chunk_key

    weight = (
        float(one_record[-1]) if is_float_regex(one_record[-1]) else 1.0
    )

    return dict(
        src_id = source,
        tgt_id = target,
        weight=weight,
        description = edge_description,
        source_id = edge_source_id)


# processing the llm output and return nodes and edges
async def process_llm_output(final_result:str, context_base:dict, chunk_key):

    records = split_string_by_delimiters(
        final_result,
        [context_base["record_delimiter"],  # "##"
        context_base["completion_delimiter"]])  # "<|COMPLETE|>"
    
    maybe_edges = defaultdict(list)
    
    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)  # get the content in "()"

        record_attributes = split_string_by_delimiters(
                record, [context_base["tuple_delimiter"]]  # "<|>"
            )
        if_relation = await handle_extracted_single_relationship(record_attributes, chunk_key=chunk_key)

        if if_relation is not None:
            maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)
    
    return maybe_edges


async def triples_extraction(
        chunks,
        knowledge_graph_inst: BaseGraphStorage,
        global_config:dict,
        relationships_vdb: BaseVectorStorage,
        prompt_cache_manage: KVCacheProcess,
        new_docs: dict,
        tem_save: TemporarySave
):
    
    use_llm_func: callable = global_config['llm_model_func']
    ordered_chunks = list(chunks.items())
    already_processed = 0
    processed_chunk_num = 0
    already_relations = 0
    maybe_edges = defaultdict(list)

    prompt_manage = PromptManage()
    entity_extract_prompt = prompt_manage.entity_extract_prompt
    context_base = prompt_manage.context_base
    continue_prompt = prompt_manage.continue_prompt  # For prompting some triples may be missed
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    async def process_single_content(
            chunk_key_dp: tuple[str, TextChunkSchema], 
            processed_chunk_num: int, 
            ollama_client=None, 
            prompt_cache_manage: KVCacheProcess = None):

        nonlocal already_processed, already_relations

        if already_processed == 0 and processed_chunk_num != 0:
            already_processed = processed_chunk_num
        
        chunk_key = chunk_key_dp[0]  # chunkid
        chunk_dp = chunk_key_dp[1]  # chunk value: include size, content, docid
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)

        start_time = time.perf_counter()

        if ollama_client != None:  # use llm to answer
            final_result = await use_llm_func(
                hint_prompt, 
                use_ollama_client=ollama_client)
        elif prompt_cache_manage != None:
            final_result = await use_llm_func(
                hint_prompt, 
                prompt_cache_manage=prompt_cache_manage)
        else:
            final_result = await use_llm_func(hint_prompt, 
                                              chunk_process_num=chunk_key)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)  # history message
        
        for now_glean_index in range(entity_extract_max_gleaning):

            if ollama_client != None:
                glean_result = await use_llm_func(
                    continue_prompt, 
                    history_messages=history, 
                    use_ollama_client=ollama_client)
            elif prompt_cache_manage != None:
                glean_result = await use_llm_func(
                    continue_prompt, 
                    history_messages=history, 
                    prompt_cache_manage=prompt_cache_manage)
            else:
                glean_result = await use_llm_func(
                    continue_prompt, 
                    history_messages=history,
                    chunk_process_num=chunk_key)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

        end_time = time.perf_counter()
        execution_time = round((end_time - start_time), 2)
        print(f"The time of extracting relationships: {execution_time} s")

        maybe_edges = await process_llm_output(
            final_result=final_result, 
            context_base=context_base, chunk_key=chunk_key)
        
        already_processed += 1
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )

        return dict(maybe_edges)
        

    async def process_chunks_in_batches(
            ordered_chunks, 
            ollama_client=None, 
            prompt_cache_manage: KVCacheProcess = None):

        nonlocal processed_chunk_num
        # set a fixed interval and save the number of processed chunks.
        processed_chunk_num_path:str = os.path.join(
            global_config["working_dir"]+"/processed_chunk_num.json")
        try:
            with open(processed_chunk_num_path, 'r') as file:
                processed_chunk = json.load(file)
                processed_chunk_num = processed_chunk["processed_num"]
        except Exception as e:
            processed_chunk_num = 0
        tasks = []
        for chunk in ordered_chunks:
            tasks.append(process_single_content(
                chunk, processed_chunk_num, ollama_client=ollama_client, 
                prompt_cache_manage=prompt_cache_manage))

        # save results every 100 steps
        all_results = []
        step = 100000
        for i in range(0, len(tasks), step):
            if i == 0 and processed_chunk_num != 0:
                i += processed_chunk_num
            batch_results = []
            batch_results = await asyncio.gather(*tasks[i:i+step])

            # wrong chunk process
            with open (global_config["working_dir"]+ f"/freestylerag.log") as fil:
                log_content = fil.read()
            numbers = re.findall(r'wrong_chunk_num:\s*(chunk-[^\s]+)', log_content)
            wrong_tasks = []
            if numbers:
                matching_chunks = [chunk for chunk in ordered_chunks if chunk[0] in numbers]
                if len(matching_chunks):
                    for chunk in matching_chunks:
                        wrong_tasks.append(process_single_content(
                        chunk, processed_chunk_num, ollama_client=ollama_client, 
                        prompt_cache_manage=prompt_cache_manage))

            if len(wrong_tasks):
                supple_results = await asyncio.gather(*wrong_tasks)
                batch_results.extend(supple_results)

            if len(tasks) <= step:
                return batch_results
            
            all_results.extend(batch_results)

            save_num = 0
            while True:
                try:
                    await tem_save.save_if_need(batch_results, new_docs, chunks, i+step)
                    if save_num == 3:
                        break
                    break
                except Exception as e:
                    save_num += 1
                    print("The current batch saving is abnormal!!")
                    print(f"Exception message: {e}")
                    await asyncio.sleep(0.1)

        return all_results
   
    if global_config['hf_model_promt_cache'] == True:  # Get the prompt cache which need to use at the next time
        
        hint_prompt_temp = entity_extract_prompt.format(**context_base, input_text="")
        prompt_cache_process = prompt_cache_manage
        prompt_cache_process.generate_with_cache(prompt=hint_prompt_temp)
        hint_prompt_temp = ''
        results = await process_chunks_in_batches(
            ordered_chunks, 
            prompt_cache_manage=prompt_cache_process)
    elif global_config['one_client'] == True:  # use one ollama client to get all the answers

        from .utils import client_pool_singleton
        await client_pool_singleton.initialize(
            host=global_config["llm_model_kwargs"]["host"],
            timeout=global_config["llm_model_kwargs"]["timeout"],
            pool_size=global_config['client_size'])
        results = await process_chunks_in_batches(
            ordered_chunks, 
            ollama_client=client_pool_singleton)
    else:
        results = await process_chunks_in_batches(ordered_chunks)

    for m_edges in results:
        for k, v in m_edges.items():
            key = tuple(k)
            unique_v = []
            seen = set()
            for v_index in v:
                identifier = (v_index["description"], v_index["source_id"])
                if identifier not in seen:
                    unique_v.append(v_index)
                    seen.add(identifier)
            if key not in maybe_edges:
                maybe_edges[key].extend(unique_v)
            else:
                for v_index in unique_v:
                    if not any(
                        existing["description"] == v_index["description"] and 
                        existing["source_id"] == v_index["source_id"] 
                        for existing in maybe_edges[key]
                    ):
                        maybe_edges[key].append(v_index)

    all_relationships_data1 = await asyncio.gather(
        *[
            merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )

    all_relationships_data = [result[0] for result in all_relationships_data1]

    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content":dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # If it is less than the maximum allowable summary, no further summary is required
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)  # Determine whether the node exists in the graph.
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_delimiters(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))  # Merge description
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(  # If the maximum summary limit is exceeded, the model needs to be invoked to summarize and refine
        entity_name, description, global_config
    )
    node_data = dict(
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(  # Adding node information to nx.Graph()
        entity_name,
        node_data=dict(
        description=description,
        source_id=source_id,
    ),
    )
    node_data["entity_name"] = entity_name

    one_node_data = {}
    temp_node_data = {}
    temp_node_data["entity_name"] = entity_name
    temp_node_data["description"] = description
    temp_node_data["source_id"] = source_id
    one_node_data[entity_name] = temp_node_data


    return node_data, one_node_data

async def merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_source_ids = []
    already_description = []
    already_weights = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_delimiters(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])

    # weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    weights = [dp["weight"] for dp in edges_data] + already_weights
    weight = round(sum(weights) / len(weights), 1) if weights else 0
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )

    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
    )

    one_edge_data = {}
    temp_edge_data = {}
    temp_edge_data["src_id"] = src_id
    temp_edge_data["tgt_id"] = tgt_id
    temp_edge_data["weight"] = weight
    temp_edge_data["description"] = description
    temp_edge_data["source_id"] = source_id
    tem_key = str((src_id,tgt_id))
    one_edge_data[tem_key] = temp_edge_data

    return edge_data, one_edge_data


async def find_related_text_unit_from_relationships(
        edge_datas: list[dict],
        query_param: QueryParameters,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
):
    text_units = [  # find all of the chunk id which is related to relationships
        split_string_by_delimiters(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    text_chunks_related = {}
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in text_chunks_related:
                text_chunks_related[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }
    if any([v is None for v in text_chunks_related.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")

    all_text_units = [
        {"id": k, **v} for k, v in text_chunks_related.items() if v is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units_with_id = all_text_units
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    return all_text_units, all_text_units_with_id

async def regenerate_description(
        edge_datas: list[dict],  # src_id, tgt_id, description
        use_text_units_with_id: list[dict],
        use_model_func):  #["id"] , ["data"]["content"]

    prompt_manage = PromptManage()
    context_base = prompt_manage.context_base
    regenerate_desc_prompt = prompt_manage.regenerate_desc_prompt

    all_matches = []
    for chunk in use_text_units_with_id:
        chunk_id = chunk["id"]
        entity_str_list = []
        edge_str_list = []
        for edge_data in edge_datas:
            if edge_data["source_id"] == chunk_id:
                src_id = edge_data["src_id"]
                tgt_id = edge_data["tgt_id"]
                tuple_delimiter = "{tuple_delimiter}"
                record_delimiter = "{record_delimiter}"

                if src_id == tgt_id:
                    entity_str = f'(\"entity\"{tuple_delimiter}{src_id}{tuple_delimiter}){record_delimiter}'
                    entity_str_list.append(entity_str)
                else:
                    edge_str = f'(\"relationship\"{tuple_delimiter}{src_id}{tuple_delimiter}{tgt_id}{tuple_delimiter}){record_delimiter}'
                    edge_str_list.append(edge_str)

        if len(entity_str_list):
            entities_str = "\n".join(entity_str_list)
        else:
            entities_str = 'None'
        edges_str = "\n".join(edge_str_list)

        # 调用LLM
        chunk_str = chunk["data"]["content"]

        if entities_str != 'None':
            entities_str = entities_str.format(**context_base)
        if edges_str:
            edges_str = edges_str.format(**context_base)

        hint_prompt = regenerate_desc_prompt.format(
        **context_base,
        input_text = chunk_str,
        entity_names=entities_str, 
        relationship_names=edges_str)

        answer = await use_model_func(hint_prompt)
        
        records = split_string_when_redescription(
            answer,
            [context_base["record_delimiter"],  # "##"
            context_base["completion_delimiter"]])  # "<|COMPLETE|>"
        all_matches.extend(records)
        # pattern = r'\((.*?)\)'
        # matches = re.findall(pattern, answer, re.DOTALL)
        # all_matches.extend(matches)
    
    new_edge_datas = []
    for match in all_matches:
        match = match.strip()
        
        if "(" in match and ")" in match:
            
            m = re.match(r'^[^(]*\((.*)\)[^)]*$', match, re.DOTALL)
            if m:
                record = m.group(1)
            else:
                continue
        else:
            continue

        parts = split_string_by_delimiters(
                record, [context_base["tuple_delimiter"]]  # "<|>"
            )

            
        if not parts or len(parts) < 3:
            continue

        if parts[0] != '"relationship"' and parts[0] != '"entity"':
            continue

        for edge_data in edge_datas:
            src_id = edge_data["src_id"]
            tgt_id = edge_data["tgt_id"]
            if parts[0] == '"relationship"':
                if len(parts) < 4:
                    continue
                if src_id.strip().lower() == parts[1].strip().lower() and tgt_id.strip().lower() == parts[2].strip().lower():
                    edge_data["description"] = parts[3]
                    new_edge_datas.append(edge_data)
            elif parts[0] == '"entity"':
                if src_id == tgt_id and src_id.strip().lower() == parts[1].strip().lower():
                    edge_data["description"] = parts[2]
                    new_edge_datas.append(edge_data)

    if len(new_edge_datas) != len(edge_datas):
        existing_pairs = set()
        for ed in new_edge_datas:
            
            pair = (ed["src_id"].strip().lower(), ed["tgt_id"].strip().lower())
            existing_pairs.add(pair)

        
        for edge_data in edge_datas:
            pair = (edge_data["src_id"].strip().lower(), edge_data["tgt_id"].strip().lower())
            if pair not in existing_pairs:
                new_edge_datas.append(edge_data)

    return new_edge_datas

            
async def get_query_context_with_keywords(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParameters,
    use_model_func,
    need_info_list:bool=False
):
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)
    if not len(results):
        return None
    
    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )
    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    # edge_datas = sorted(  # sort from largest to smallest
    #     edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    # )
    
    edge_datas = truncate_list_by_token_size(  # For descriptions, truncation is required beyond the maximum length
        edge_datas,
        key=lambda x: x['description'],
        max_token_size=query_param.max_token_for_global_context
    )

    use_text_units, use_text_units_with_id  = await find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db
    )

    logger.info(
        f"Query uses {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    # new_edge_datas = []
    # new_edge_datas = await regenerate_description(edge_datas, use_text_units_with_id, use_model_func)
    # if new_edge_datas:
    #     edge_datas = new_edge_datas

    relations_section_list = [
        ["id", "source entity", "relationship between the source entity and the target entity", "target entity", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["description"],
                e["tgt_id"],
                e["weight"],
                e["rank"],
            ]
        )

    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    info_str = f"""
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    if need_info_list:
        return info_str, relations_section_list, text_units_section_list
    else:
        return info_str


async def query_with_origional_keywords_(
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        q_param: QueryParameters,
        global_config: dict,
        sub_query_info: str=None):
    
    use_model_func = global_config["llm_model_func"]
    if q_param.no_keywords == False:
        kw_prompt_temp = PROMPTS["keywords_extraction"]

        kw_prompt = kw_prompt_temp.format(query=query)
        result = await use_model_func(kw_prompt)

        try:
            keywords_data = json.loads(result)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        except json.JSONDecodeError:
            try:
                result = (
                    result.replace(kw_prompt[:-1], "")  
                    .replace("user", "")  
                    .replace("model", "")  
                    .strip()
                )
                result = "{" + result.split("{")[1].split("}")[0] + "}"  
                keywords_data = json.loads(result)
                hl_keywords = keywords_data.get("high_level_keywords", [])  
                ll_keywords = keywords_data.get("low_level_keywords", [])  
                hl_keywords = ", ".join(hl_keywords)
                ll_keywords = ", ".join(ll_keywords)
            # Handle parsing error
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return PROMPTS["fail_response"]

        # get context through local keywards
        all_key_words = f'{hl_keywords}, {ll_keywords}'
    else:
        all_key_words = query
    context = await get_query_context_with_keywords(
        keywords=all_key_words,
        knowledge_graph_inst=knowledge_graph_inst,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=q_param,
        use_model_func=use_model_func
    )

    if q_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    if q_param.task_name == "qa":
        prompt_temp = PROMPTS['qa_response']
        if sub_query_info is not None:
            sys_prompt = prompt_temp.format(
                context=context+sub_query_info, input=query)
        else:
            sys_prompt = prompt_temp.format(
                context=context, input=query)
        response = await use_model_func(
            prompt=sys_prompt)
    else:
        sys_prompt_temp = PROMPTS["rag_response"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type=q_param.response_type
        )
        response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        )

    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response

class QA_Pipeline_v2:
    def __init__(self, original_q, knowledge_graph_inst, 
                 relationships_vdb, text_chunks_db, q_param, global_config, summary_model=None):
        
        self.original_q = original_q
        self.q1 = None
        self.ans1 = None
        self.q2 = None
        self.ans2 = None
        self.q3 = None
        self.ans3 = None
        self.final_ans = None
        self.strategy = False
        self.knowledge_graph_inst = knowledge_graph_inst
        self.relationships_vdb = relationships_vdb
        self.text_chunks_db = text_chunks_db
        self.final_summary_q_param = copy.deepcopy(q_param)
        self.q_param = copy.deepcopy(q_param)
        self.global_config = global_config
        self.info_relation_list = []
        self.info_text_list = []
        self.q1_summary: str = ''
        self.q2_summary: str = ''
        self.q3_summary: str = ''
        self.summary_model = summary_model
        self.num_commars = 0

    def str2set(self,s:str):
        s = s.strip()
        ans_list = s.lstrip('[').rstrip(']').split('#')
        if(isinstance(ans_list, str)):
            ans_list = [ans_list]
        for i in range(len(ans_list)):
            a = ans_list[i].strip()
            a = a.lstrip('[').rstrip(']')
            ans_list[i] = a
        # if(len(ans_list)>5):
        #     ans_list = ans_list[0:5]
        ans_set = set(ans_list)
        ans_set.discard('None')
        return ans_set

    async def merge_info(self):  
        order_relation_list = []
        order_text_list = []
        seen_elements = set()
        for sublist in self.info_relation_list:
            judge_str = sublist[1] + sublist[2] + sublist[3]
            if judge_str not in seen_elements:
                order_relation_list.append(sublist)
                seen_elements.add(judge_str)
        seen_elements = set()
        for sublist in self.info_text_list:
            judge_str = sublist[1]
            if judge_str not in seen_elements:
                order_text_list.append(sublist)
                seen_elements.add(judge_str)
        str_relation = list_of_list_to_csv(order_relation_list)
        str_text = list_of_list_to_csv(order_text_list)
        return f"""
-----Relationships-----
```csv
{str_relation}
```
-----Sources-----
```csv
{str_text}
```
"""

    async def getting_response_from_info(self, info, query, sub_query_info=None, who_call = None):
        original_info = info
        use_model_func = self.global_config["llm_model_func"]
        if info is None:
            return PROMPTS["fail_response"]

        
        if self.q_param.need_summary_return:
            summary_prompt = PROMPTS["summary_response"]
            summary_prompt = summary_prompt.format(input_question=query, input_data_table=info)
            temp_info = await use_model_func(prompt = summary_prompt)
            if "<|end|>" in temp_info:
                temp_info = re.sub(r'<\|end\|>', '', temp_info)
            if "Relationships" in temp_info and "Sources" in temp_info:
                info = temp_info
                if who_call == "q1":
                    self.q1_summary = info
                elif who_call == "q2":
                    self.q2_summary = info
                elif who_call == "q3":
                    self.q3_summary = info
            else:
                info = original_info
        elif self.q_param.final_answer_summary:
            summary_prompt = PROMPTS["summary_response"]
            summary_prompt = summary_prompt.format(input_question=query, input_data_table=original_info)
            temp_info = await use_model_func(prompt = summary_prompt)
            if "<|end|>" in temp_info:
                temp_info = re.sub(r'<\|end\|>', '', temp_info)
            if "Relationships" in temp_info and "Sources" in temp_info:
                info = temp_info
            else:
                info = original_info

        if self.q_param.task_name == "qa":
            prompt_temp = PROMPTS['qa_response']
            if sub_query_info is not None:
                sys_prompt = prompt_temp.format(
                    context=info+sub_query_info, input=query)
            else:
                sys_prompt = prompt_temp.format(
                    context=info, input=query)
            response = await use_model_func(
                prompt=sys_prompt)
        else:
            sys_prompt_temp = PROMPTS["qa_response_odqa"]
            sys_prompt = sys_prompt_temp.format(
                context=info, input=query
            )
            response = await use_model_func(
           prompt=sys_prompt,
            )

        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )
        return response

    async def Final_QA(self,final_call:str):
        answer = None
        if('Ans_1' in final_call):
            answer = self.ans1
            self.final_ans = answer[0]
        elif('Ans_2' in final_call):
            if(self.ans2):
                if len(self.ans2) == 1:
                    answer = self.ans2[0]
                else:
                    answer = set()
                    for s in self.ans2:
                        if(s):
                            answer = answer.union(s)
            self.final_ans = answer
        elif('Ans_3' in final_call):
            if(self.ans3):
                if len(self.ans3) == 1:
                    answer = self.ans3[0]
                else:
                    answer = set()
                    for s in self.ans3:
                        if(s):
                            answer = answer.union(s)  
            self.final_ans = answer

    def split_str(self, text):
        if text:
            
            sub_strings = text.split(',')
           
            sub_strings = [s.strip() for s in sub_strings]
            return sub_strings
        else:
            return ""

    async def question_answer_link(self, q, ans, only_question=False):
        sub_answer = ""

        if only_question:
            if isinstance(q, list):
                for i in range(len(q)):
                   if q[i]:
                    sub_answer += f"sub-question: {q[i]}\n"
            else:
                if q:
                    sub_answer = f"sub-question: {q}\n"
            return sub_answer

        if q and ans:
            if isinstance(q, list) and isinstance(ans, list):
                min_length = min(len(q), len(ans))
                for i in range(min_length):
                    if q[i] and ans[i]:
                        sub_answer += f"Question: {q[i]} Answer: {ans[i]}\n"
            elif isinstance(q, list):
                for q_single in q:
                    if q_single:
                        sub_answer += f"Question: {q_single} Answer: {ans}\n"
            elif isinstance(ans, list):
                for ans_single in ans:
                    if ans_single:
                        sub_answer += f"Question: {q} Answer: {ans_single}\n"
            else:
                sub_answer = f"Question: {q} Answer: {ans}\n"

        return sub_answer

    async def get_sub_query_info(self, only_question=False, only_original_question=False):

        if only_question:
            sub_q1 = await self.question_answer_link(q=self.q1, ans=self.ans1, only_question=only_question)
            sub_q2 = await self.question_answer_link(q=self.q2, ans=self.ans2, only_question=only_question)
            sub_q3 = await self.question_answer_link(q=self.q3, ans=self.ans3, only_question=only_question)
            sub_q = sub_q1 + sub_q2 + sub_q3
            if len(sub_q):
                sub_prefix = "\n(Here are some sub-questions related to the final question and they might help you to answer. Please note that you should not answer these sub-questions: \n"
                sub_query_information = sub_prefix + sub_q + ")\n"
            else:
                sub_query_information = None
            return sub_query_information
        
        if only_original_question:
            sub_query_information = f'\n### The prerequisite conditions for the current question:###{self.original_q}\n'
            return sub_query_information

        sub_answer1 = await self.question_answer_link(q=self.q1, ans=self.ans1)
        sub_answer2 = await self.question_answer_link(q=self.q2, ans=self.ans2)
        sub_answer3 = await self.question_answer_link(q=self.q3, ans=self.ans3)
        sub_answer = sub_answer1 + sub_answer2 + sub_answer3
        if len(sub_answer):
            sub_prefix = "\n### Other information:###("
            sub_query_information = sub_prefix + sub_answer + ")\n"
        else:
            sub_query_information = None
        return sub_query_information

    async def output_parse(self, llm_predict:str):
        
        def contains_date(text):
            try:
               
                result = parser.parse(text, fuzzy=True)
                return True
            except (ValueError, OverflowError):
                return False

        error_flag = False
        predict_lines = llm_predict.split('\n')
        if self.q_param.task_name=="qa":
            for line in predict_lines:  
                
                if self.ans1:
                    if isinstance(self.ans1, list):
                        ans1_index = False
                        for ans_i in self.ans1:
                            time_index = contains_date(ans_i)
                            if  "," in ans_i:
                                self.num_commars += 1
                                ans1_index = True
                            if time_index:
                                ans1_index = False
                        if ans1_index:
                            break
                    elif "," in self.ans1 and contains_date(self.ans1)==False:
                    # elif "<SEP>" in self.ans1:
                        self.num_commars += 1
                        break
                if self.ans2:
                    if isinstance(self.ans2, list):
                        ans2_index = False
                        for ans_i in self.ans2:
                            time_index = contains_date(ans_i)
                            if  "," in ans_i:
                                self.num_commars += 1
                                ans2_index = True
                            if time_index:
                                ans2_index = False
                        if ans2_index: 
                            break
                    elif "," in self.ans2 and contains_date(self.ans2)==False:
                    # elif "<SEP>" in self.ans2:
                        self.num_commars += 1
                        break
                if self.ans3:
                    if isinstance(self.ans3, list):
                        ans3_index = False
                        for ans_i in self.ans3:
                            time_index = contains_date(ans_i)
                            if  "," in ans_i:
                                self.num_commars += 1
                                ans3_index = True
                            if time_index:
                                ans3_index = False
                        if ans3_index:
                            break
                    elif "," in self.ans3 and contains_date(self.ans3)==False:
                    # elif "<SEP>" in self.ans3:
                        self.num_commars += 1
                        break

                if self.final_ans:
                    break
                if('Sub_Question_1: str = ' in line):
                    q1 = line.split('Sub_Question_1: str = ')[1].lstrip('f').strip("\"")  # 获取到子问题
                    if ("{Ans_1}" in q1):
                        break
                    print('Sub_Question_1:'+q1)
                    self.q1 = [q1]
                    
                    information, info_relation_list, info_text_list = \
                    await get_query_context_with_keywords(
                        keywords=q1,
                        knowledge_graph_inst=self.knowledge_graph_inst,
                        relationships_vdb=self.relationships_vdb,
                        text_chunks_db=self.text_chunks_db,
                        query_param=self.q_param,
                        use_model_func=self.global_config["llm_model_func"],
                        need_info_list=True
                        )
                    self.info_relation_list.extend(info_relation_list)
                    self.info_text_list.extend(info_text_list)
                    ans1 = await self.getting_response_from_info(information, q1, who_call="q1")
                    self.ans1 = [ans1]
                    # self.info1 = [information]
                    print('Ans_1:'+str(self.ans1))
                    if ans1:
                        error_flag = True
                    else:
                        break
                if('Sub_Question_2: str = ' in line):
                    q2 = line.split('Sub_Question_2: str = ')[1].lstrip('f').strip("\"")
                    print('Sub_Question_2:'+q2)
                    question2 = []
                    # info2 = []
                    answer2 = []
                    if("{Ans_1}" in q2):
                        if len(self.ans1) == 1:
                            ans_1 = self.ans1
                        else:
                            ans_1 = set()
                            for s in self.ans1:
                                if(s):
                                    ans_1 = ans_1.union(s)
                        for idx, a in enumerate(ans_1):
                            q2_1 = q2.replace("{Ans_1}",a)
                            print('Sub_Question_2_' + str(idx) + ':' +q2_1)
                            information, info_relation_list, info_text_list = \
                            await get_query_context_with_keywords(
                                keywords=q2_1,
                                knowledge_graph_inst=self.knowledge_graph_inst,
                                relationships_vdb=self.relationships_vdb,
                                text_chunks_db=self.text_chunks_db,
                                query_param=self.q_param,
                                use_model_func=self.global_config["llm_model_func"],
                                need_info_list=True
                                )
                            self.info_relation_list.extend(info_relation_list)
                            self.info_text_list.extend(info_text_list)
                            # sub_query_info = await self.get_sub_query_info()
                            ans2 = await self.getting_response_from_info(information, q2_1, who_call="q2")
                            answer2.append(ans2)
                            print('Ans_2_'+ str(idx) + ':' + str(ans2))
                            question2.append(q2_1)
                            # info2.append(information)
                    elif ("{Ans_2}" in q2):
                        break
                    else:
                        information, info_relation_list, info_text_list = \
                            await get_query_context_with_keywords(
                            keywords=q2,
                            knowledge_graph_inst=self.knowledge_graph_inst,
                            relationships_vdb=self.relationships_vdb,
                            text_chunks_db=self.text_chunks_db,
                            query_param=self.q_param,
                            use_model_func=self.global_config["llm_model_func"],
                            need_info_list=True
                            )
                        self.info_relation_list.extend(info_relation_list)
                        self.info_text_list.extend(info_text_list)
                        # sub_query_info = await self.get_sub_query_info()
                        ans2 = await self.getting_response_from_info(information, q2, who_call="q2")
                        answer2.append(ans2)
                        print('Ans_2:' + str(ans2))
                        question2.append(q2)
                        # info2.append(information)
                    self.q2 = question2
                    self.ans2 = answer2
                    # self.info2 = info2
                    print('Sub_Question2:' + q2)
                    print('All Ans_2:' + str(self.ans2))
                    if self.ans2:
                        error_flag = True
                    else:
                        break
                if('Sub_Question_3: str = ' in line):
                    q3 = line.split('Sub_Question_3: str = ')[1].lstrip('f').strip("\"")
                    print('Sub_Question_3:'+q3)
                    question3 = []
                    # info3 = []
                    answer3 = []
                    if len(self.ans1) == 1:
                        ans_1 = self.ans1
                    else:
                        ans_1 = set()
                        for s in self.ans1:
                            if(s):
                                ans_1 = ans_1.union(s)
                    if len(self.ans2) == 1:
                        ans_2 = self.ans2
                    else:
                        ans_2 = set()
                        for s in self.ans2:
                            if(s):
                                ans_2 = ans_2.union(s)
                    if ("{Ans_1}" in q3):
                        for idx1, a in enumerate(ans_1):
                            q3_1 = q3.replace("{Ans_1}",a)
                            if("{Ans_2}" in q3_1):
                                for idx2, b in enumerate(ans_2):
                                    q3_2 = q3_1.replace("{Ans_2}",b)
                                    print('Sub_Question_3_' + str(idx1) + '_' + str(idx2) + ':' +q3_2)
                                    information, info_relation_list, info_text_list = \
                                        await get_query_context_with_keywords(
                                        keywords=q3_2,
                                        knowledge_graph_inst=self.knowledge_graph_inst,
                                        relationships_vdb=self.relationships_vdb,
                                        text_chunks_db=self.text_chunks_db,
                                        query_param=self.q_param,
                                        use_model_func=self.global_config["llm_model_func"],
                                        need_info_list=True
                                        )
                                    self.info_relation_list.extend(info_relation_list)
                                    self.info_text_list.extend(info_text_list)
                                    # sub_query_info = await self.get_sub_query_info()
                                    ans3 = await self.getting_response_from_info(information, q3_2, who_call="q3")
                                    print('Ans_3_'+ str(idx1) + '_' + str(idx2) + ':' + str(ans3))
                                    question3.append(q3_2)
                                    answer3.append(ans3)
                                    # info3.append(information)
                            else:
                                print('Sub_Question_3_' + str(idx1) + ':' +q3_1)
                                information, info_relation_list, info_text_list = \
                                    await get_query_context_with_keywords(
                                    keywords=q3_1,
                                    knowledge_graph_inst=self.knowledge_graph_inst,
                                    relationships_vdb=self.relationships_vdb,
                                    text_chunks_db=self.text_chunks_db,
                                    query_param=self.q_param,
                                    use_model_func=self.global_config["llm_model_func"],
                                    need_info_list=True
                                    )
                                self.info_relation_list.extend(info_relation_list)
                                self.info_text_list.extend(info_text_list)
                                # sub_query_info = await self.get_sub_query_info()
                                ans3 = await self.getting_response_from_info(information, q3_1, who_call="q3")
                                answer3.append(ans3)
                                print('Ans_3_'+ str(idx1) + ':' + str(ans3))
                                question3.append(q3)
                                # info3.append(information)
                    elif ("{Ans_2}" in q3):
                        for idx,b in enumerate(ans_2):
                            q3_1 = q3.replace("{Ans_2}",b)
                            print('Sub_Question_3_' + str(idx) + ':' +q3_1)
                            information, info_relation_list, info_text_list = \
                                await get_query_context_with_keywords(
                                keywords=q3_1,
                                knowledge_graph_inst=self.knowledge_graph_inst,
                                relationships_vdb=self.relationships_vdb,
                                text_chunks_db=self.text_chunks_db,
                                query_param=self.q_param,
                                use_model_func=self.global_config["llm_model_func"],
                                need_info_list=True
                                )
                            self.info_relation_list.extend(info_relation_list)
                            self.info_text_list.extend(info_text_list)
                            # sub_query_info = await self.get_sub_query_info()
                            ans3 = await self.getting_response_from_info(information, q3_1, who_call="q3")
                            # info3.append(information)
                            question3.append(q3_1)
                            answer3.append(ans3)
                            print('Ans_3_'+ str(idx) + ':' + str(ans3))
                    elif ("{Ans_3}" in q3):
                        break
                    else:
                        information, info_relation_list, info_text_list = \
                            await get_query_context_with_keywords(
                            keywords=q3,
                            knowledge_graph_inst=self.knowledge_graph_inst,
                            relationships_vdb=self.relationships_vdb,
                            text_chunks_db=self.text_chunks_db,
                            query_param=self.q_param,
                            use_model_func=self.global_config["llm_model_func"],
                            need_info_list=True
                            )
                        self.info_relation_list.extend(info_relation_list)
                        self.info_text_list.extend(info_text_list)
                        # sub_query_info = await self.get_sub_query_info()
                        ans3 = await self.getting_response_from_info(information, q3, who_call="q3")
                        answer3.append(ans3)
                        print('Ans_3:' + str(ans3))
                        question3.append(q3)
                        # info3.append(information)
                    self.q3 = question3
                    # self.info3 = info3
                    self.ans3 = answer3
                    print('Sub_Question3:' + q3)
                    print('All Ans_3:' + str(self.ans3))
                    if self.ans3:
                        error_flag = True
                    else:
                        break
                if('Sub_Question_4: str = ' in line):
                    error_flag = False
                    break
                if('Final_Answer: str = ' in line):
                    await self.Final_QA(line)

        if (self.final_ans is None or len(self.final_ans) == 0 or error_flag == False):
            use_model_func=self.global_config["llm_model_func"]
            information = await get_query_context_with_keywords(
                keywords=self.original_q,
                knowledge_graph_inst=self.knowledge_graph_inst,
                relationships_vdb=self.relationships_vdb,
                text_chunks_db=self.text_chunks_db,
                query_param=self.final_summary_q_param,
                use_model_func=use_model_func
                )
            if self.summary_model and self.q_param.task_name=="qa":
                summary_prompt = PROMPTS["summary_response_RL"]
                # summary_prompt = PROMPTS["summary_response_2.0"]
                summary_prompt = summary_prompt.replace("{input_question}", self.original_q)
                summary_prompt = summary_prompt.replace("{input_data_table}", information)

                system_prompt = """
Respond in the following format. Please note that there should be no symbols before "Because" and after "<|end|>".
Because ...
...
Therefore ...
...
Final answer: <|start|>...<|end|>"""
                summary_response = self.summary_model.process_question(summary_prompt, system_prompt=system_prompt)
                final_response = await self.extract_content(summary_response)
            else:
                summary_prompt = PROMPTS["summary_response_2.0"]
                summary_prompt = summary_prompt.format(input_question=self.original_q, input_data_table=information)
                final_response = await use_model_func(prompt=summary_prompt)
                final_response = await self.extract_content(final_response)
                # final_response = await self.getting_response_from_info(information, self.original_q)
            if not final_response:
                final_response = await self.getting_response_from_info(information, self.original_q)
            return final_response , self.num_commars
        else:
            if isinstance(self.final_ans, set):
                final_ans = ", ".join(self.final_ans)
            else:
                final_ans = self.final_ans
            print('Final Answer:'+str(final_ans))
            return str(final_ans), self.num_commars
        
    async def extract_content(self, s):

        parts = s.split("<|start|>")
        if len(parts) < 2:
            return ""  
        
        content_part = parts[1].split("<|end|>")[0]
        return content_part.strip()

async def query_with_plan_(
        query: str,
        planning_model: PlanningModelConfig,
        summary_model: PlanningModelConfig,
        knowledge_graph_inst: BaseGraphStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        q_param: QueryParameters,
        global_config: dict
):
    if planning_model:
        planning_prompt = PROMPTS["question_planning_2.0"]
        planning_prompt_ = planning_prompt.replace("{_original_question_}", query)
        plan_text = planning_model.process_question(planning_prompt_)

    else:
        plan_text = ''
    pipline = QA_Pipeline_v2(query, knowledge_graph_inst, relationships_vdb,
                        text_chunks_db, q_param, global_config, summary_model)

    response , num_commars = await pipline.output_parse(plan_text)
    return response, num_commars