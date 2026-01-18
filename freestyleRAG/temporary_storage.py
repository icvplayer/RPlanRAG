import asyncio
import os
from collections import defaultdict
from freestyleRAG.utils import (logger, compute_mdhash_id, 
                            EmbeddingFunc, temp_write_json)
from typing import cast
from dataclasses import dataclass, field
from freestyleRAG.baseClasses import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace
)

from freestyleRAG.storageClasses import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from typing import Type
from freestyleRAG.llm import openai_embedding
import json
import ast
import copy


@dataclass
class TemporarySave(StorageNameSpace):
    full_docs: Type[BaseKVStorage] = field(default=JsonKVStorage)
    text_chunks: Type[BaseKVStorage] = field(default=JsonKVStorage)
    llm_response_cache: Type[BaseKVStorage] = field(default=JsonKVStorage)
    relationships_vdb: Type[BaseVectorStorage] = field(default=NanoVectorDBStorage)
    chunks_vdb: Type[BaseVectorStorage] = field(default=NanoVectorDBStorage)
    chunk_entity_relation_graph: Type[BaseGraphStorage] = field(default=NetworkXStorage)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    empty_relationships_vdb: Type[BaseVectorStorage] = field(default=NanoVectorDBStorage)

    def __post_init__(self):
        self.relationships_file_name = os.path.join(self.global_config["working_dir"]+ f"/vdb_relationships_temp.json")
        self.empty_relationships_vdb = self.create_vdb(namespace="relationships")
         
    async def save_if_need(self, results, new_docs, inserting_chunks, current_count):
        if results:
            current_count = current_count + 1

            self.old_relationships_vdb = self.create_vdb(namespace="relationships")
            
            maybe_edges = defaultdict(list)
            
            if os.path.exists(self.relationships_file_name):

                with open(self.relationships_file_name, 'r') as file:
                    temp_maybe_edges = json.load(file)
                    if temp_maybe_edges:
                        for key, value in temp_maybe_edges.items():
                            key = ast.literal_eval(key)
                            maybe_edges[key].extend([value])
            
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

            from freestyleRAG.rag_operating import merge_edges_then_upsert
                
            all_relationships_data1 = await asyncio.gather(
                    *[
                   merge_edges_then_upsert(k[0], k[1], v, self.chunk_entity_relation_graph, self.global_config)
                    for k, v in maybe_edges.items()
                    ]
            )

            all_relationships_tosave = [result[1] for result in all_relationships_data1]
            all_relationships_data = [result[0] for result in all_relationships_data1]
            
            all_to_save = {}
            for one_re_tosave in all_relationships_tosave:
                all_to_save.update(one_re_tosave)
            temp_write_json(all_to_save, self.relationships_file_name)

            if not len(all_relationships_data):
                logger.warning(
                    "Didn't extract any relationships, maybe your LLM is not working")
                return None
            if self.old_relationships_vdb is not None:
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
                await self.old_relationships_vdb.upsert(data_for_vdb)  # update nanoVectorDB

            maybe_edges.clear()
           
            if self.chunk_entity_relation_graph is None:
                logger.warning("No new entities and relationships found")
                return
            
            await self.full_docs.upsert(new_docs)  # JsonKVStorage
            await self.text_chunks.upsert(inserting_chunks)  # JsonKVStorage
            await self.tem_insert_done_()  # dump into json file

            self.old_relationships_vdb = copy.deepcopy(self.empty_relationships_vdb)

            with open(os.path.join(self.global_config["working_dir"]+"/processed_chunk_num.json"), 'w') as file:
                json.dump({"processed_num": current_count-1}, file)

            print("Saving the info of the chunk {}".format(current_count-1))

    async def tem_insert_done_(self):
        tasks = []
        for storage_inst in [
            self.llm_response_cache,
            self.old_relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).temp_index_done_callback())
        await asyncio.gather(*tasks)
        return

    def create_vdb(self, namespace: str):
        if namespace == "entities":
            return NanoVectorDBStorage(
                                namespace=namespace,
                                global_config=self.global_config,
                                embedding_func=self.embedding_func,
                                meta_fields={"entity_name"},
                                )
        else:
            return NanoVectorDBStorage(
                                namespace=namespace,
                                global_config=self.global_config,
                                embedding_func=self.embedding_func,
                                meta_fields={"src_id", "tgt_id"},
                                )


 