from argparse import ArgumentParser
from freestyleRAG.freestylerag import freestyleRAG
from freestyleRAG.llm import ollama_model_complete, ollama_embedding
from freestyleRAG.utils import EmbeddingFunc
from freestyleRAG.baseClasses import QueryParameters

# import sys
# # import os
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main(args):

    rag = freestyleRAG(working_dir=args.working_dir, llm_model_func=ollama_model_complete, 
                       llm_model_name=args.gen_llm_name, llm_model_max_async=args.llm_model_max_async,
                       llm_model_max_token_size=args.llm_model_max_token_size, 
                       llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}, "timeout":args.time_out},
                       embedding_func=EmbeddingFunc(
                           embedding_dim=args.embedding_dim,
                           max_token_size=args.max_token_size,
                           func=lambda texts: ollama_embedding(texts, embed_model="bge-m3", host="http://localhost:11434"),                           
                       ),
                       one_client=args.one_client,
                       client_size=args.client_size)
    
    with open("./example_data/book2.txt", "r", encoding="utf-8") as f:
        rag.construct_graph(f.read())


    query = 'your query'
    q_param = QueryParameters(top_k=10)
    answer = rag.query_with_origional_keywords(query, q_param)
    print(answer)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--working_dir", type=str, default="./ollama_results", help="The path to storage results")
    parser.add_argument("--gen_llm_name", type=str, default="qwen2.5:7b-instruct", help="The model name which is used to generate triples")
    parser.add_argument("--llm_model_max_async", type=int, default=4, help="Add restriction of maximum async calling times for a async func")
    parser.add_argument("--llm_model_max_token_size", type=int, default=32768, help="The maximum tokensize of the generating model")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="The embedding dim of the embedding model")
    parser.add_argument("--max_token_size", type=int, default=8192, help="The maximum tokensize of the embedding model")
    parser.add_argument("--time_out", type=int, default=150, help="The maximum request time of the ollama client")
    parser.add_argument("--one_client", type=bool, default=False, help="Using ollama client pool or not")
    parser.add_argument("--client_size", type=int, default=4, help="Number of ollama clients")
    
    args = parser.parse_args()

    main(args)