import os
import asyncio
from functools import partial
from freestyleRAG.rag_operating import (chunking_by_token_size, 
                                        triples_extraction, 
                                        query_with_origional_keywords_,
                                        query_with_plan_)
from dataclasses import asdict, dataclass, field
from freestyleRAG.utils import (logger, EmbeddingFunc, 
                                set_logger, compute_mdhash_id, 
                                limit_async_func_call, PlanningModelConfig)
from datetime import datetime
from freestyleRAG.llm import openai_embedding, gpt_4o_mini_complete
from typing import Type, cast
from freestyleRAG.baseClasses import (BaseKVStorage, BaseVectorStorage, 
                                      StorageNameSpace, BaseGraphStorage, QueryParameters)
from freestyleRAG.storageClasses import (JsonKVStorage, NanoVectorDBStorage, 
                                         KVCacheProcess, NetworkXStorage)
from freestyleRAG.temporary_storage import TemporarySave
from freestyleRAG.neo_kg.neo4j_impl import Neo4JStorage


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()

    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        return loop


@dataclass
class freestyleRAG:
    working_dir: str = field(default_factory=lambda: f"./freestylerag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
    kg: str = field(default="NetworkXStorage")
    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = field(default=1)
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    enable_llm_cache: bool = True
    prompt_cache_manage = KVCacheProcess

    # temporary storage
    full_docs: Type[BaseKVStorage] = JsonKVStorage
    text_chunks: Type[BaseKVStorage] = JsonKVStorage
    llm_response_cache: Type[BaseKVStorage] = JsonKVStorage
    # entities_vdb: Type[BaseVectorStorage] = NanoVectorDBStorage
    relationships_vdb: Type[BaseVectorStorage] = NanoVectorDBStorage
    chunks_vdb: Type[BaseVectorStorage] = NanoVectorDBStorage
    chunk_entity_relation_graph: Type[BaseGraphStorage] = NetworkXStorage
    tem_stor: Type[StorageNameSpace] = TemporarySave

    one_client: bool = field(default=False)
    client_size: int = field(default=4)
    hf_model_promt_cache: bool = field(default=False)
    use_question_plan: bool = field(default=False)

    def __post_init__(self):

        log_file = os.path.join(self.working_dir, "freestylerag.log")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        set_logger(log_file)  # setting the log level
        logger.setLevel(self.log_level)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"RAG init with param:\n  {_print_config}\n")

        # @TODO: should move all storage setup here to leverage initial start params attached to self.
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[  # select NetworkXStorage to manage nodes
            self.kg
        ]        

        # loading old docs
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )
        # loading old chunks
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )
        # use for chunk embedding
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.chunk_entity_relation_graph = self.graph_storage_cls(  # load the graph, if none, then create a new one
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(  
            self.embedding_func
        )

        # self.entities_vdb = self.vector_db_storage_cls(  # NanoVectorDB
        #     namespace="entities",
        #     global_config=asdict(self),
        #     embedding_func=self.embedding_func,
        #     meta_fields={"entity_name"},
        # )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )
        
        if self.hf_model_promt_cache == True:
            self.prompt_cache_manage = KVCacheProcess(self.llm_model_name)

        self.tem_save = self.tem_stor(
            namespace="temp_storage",
            global_config=asdict(self),
            full_docs=self.full_docs,
            text_chunks=self.text_chunks,
            llm_response_cache=self.llm_response_cache,
            relationships_vdb=self.relationships_vdb,
            chunks_vdb=self.chunks_vdb,
            chunk_entity_relation_graph=self.chunk_entity_relation_graph,
            embedding_func = self.embedding_func,
        )        


    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            "Neo4JStorage": Neo4JStorage,
            "NetworkXStorage": NetworkXStorage,
        }

    # Building a simple KG
    def construct_graph(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.build_kg(string_or_strings))
    
    async def build_kg(self, string_or_strings):

        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        
        # Computing hash id for docs
        new_docs = {
            compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
            for c in string_or_strings
            }
        _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}

        if not len(new_docs):
            logger.warning("All docs are already in the storage")
            return
        logger.info(f"[New Docs] inserting {len(new_docs)} docs")

        # chunk
        inserting_chunks = {}

        for doc_key, doc in new_docs.items():
            chunks = {
                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                    **dp,
                    "full_doc_id": doc_key,
                }
                for dp in chunking_by_token_size(
                    doc["content"],
                    overlap_token_size=self.chunk_overlap_token_size,  # 100
                    max_token_size=self.chunk_token_size,  # 1200
                    tiktoken_model=self.tiktoken_model_name,  # gpt4o-mini
                )
            }
            inserting_chunks.update(chunks)

        _add_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }
        if not len(inserting_chunks):
            logger.warning("All chunks are already in the storage")
            return
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

        # Updating chunk storage
        await self.chunks_vdb.upsert(inserting_chunks)
        # triples extraction
        logger.info("[Triples Extraction]...")
        maybe_new_kg = await triples_extraction(
            chunks=inserting_chunks, 
            global_config=asdict(self),
            relationships_vdb=self.relationships_vdb,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            prompt_cache_manage=self.prompt_cache_manage, 
            new_docs=new_docs, tem_save=self.tem_save)
        
        if maybe_new_kg is None:
            logger.warning("No new entities and relationships found")
            return
        self.chunk_entity_relation_graph = maybe_new_kg
        await self.full_docs.upsert(new_docs)
        await self.text_chunks.upsert(inserting_chunks)
        await self._insert_done()


    async def _insert_done(self):
        print("Final save is beginning")
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query_with_origional_keywords(self, query: str, q_param: QueryParameters):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            query_with_origional_keywords_(
            query,
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            self.text_chunks,
            q_param,
            asdict(self)))
    
    def query_with_plan(
            self, query: str, q_param: QueryParameters,
            planning_model: PlanningModelConfig, summary_model: PlanningModelConfig):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            query_with_plan_(
                query, 
                planning_model,
                summary_model,
                self.chunk_entity_relation_graph,
                self.relationships_vdb,
                self.text_chunks,
                q_param,
                asdict(self))
        )





