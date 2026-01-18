# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from argparse import ArgumentParser
from freestyleRAG.freestylerag import freestyleRAG
from freestyleRAG.llm import hf_model_complete, hf_embedding
from freestyleRAG.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer



def main(args):
    rag = freestyleRAG(working_dir=args.working_dir, 
                       llm_model_func=hf_model_complete, 
                       llm_model_name=args.gen_llm_name,
                       embedding_func=EmbeddingFunc(
                           embedding_dim=args.embedding_dim,
                           max_token_size=args.max_token_size,
                           func=lambda texts: hf_embedding(
                               texts, tokenizer=AutoTokenizer.from_pretrained(args.embedding_model_name),
                               embed_model=AutoModel.from_pretrained(args.embedding_model_name))),
                               hf_model_promt_cache=False
                               )
    
    with open("./example_data/book.txt", "r", encoding="utf-8") as f:
        rag.construct_graph(f.read())


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--working_dir", type=str, default="./huggingface_results", help="The path to storage results")
    parser.add_argument("--gen_llm_name", type=str, default="../models/Qwen2.5-7B-Instruct", help="The model name which is used to generate triples")
    parser.add_argument("--llm_model_max_async", type=int, default=4, help="Add restriction of maximum async calling times for a async func")
    parser.add_argument("--llm_model_max_token_size", type=int, default=32768, help="The maximum tokensize of the generating model")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="The embedding dim of the embedding model")
    parser.add_argument("--max_token_size", type=int, default=8192, help="The maximum tokensize of the embedding model")
    parser.add_argument("--embedding_model_name", type=str, default="../models/bge-m3", help="The embedding model name")
    
    args = parser.parse_args()

    main(args)