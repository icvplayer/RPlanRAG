import os
from argparse import ArgumentParser
from freestyleRAG.freestylerag import freestyleRAG
from freestyleRAG.llm import ollama_model_complete, ollama_embedding, openai_complete_if_cache
from freestyleRAG.utils import EmbeddingFunc
from freestyleRAG.baseClasses import QueryParameters

import os
os.environ['API_KEY'] = ''

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "qwen2.5-7b-instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("API_KEY"),
        base_url="your_url",
        **kwargs,
    )

def main(args):

    rag = freestyleRAG(working_dir=args.working_dir, llm_model_func=llm_model_func,
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
    parser.add_argument("--embedding_dim", type=int, default=1024, help="The embedding dim of the embedding model")
    parser.add_argument("--max_token_size", type=int, default=8192, help="The maximum tokensize of the embedding model")
    parser.add_argument("--one_client", type=bool, default=False, help="Using ollama client pool or not")
    parser.add_argument("--client_size", type=int, default=4, help="Number of ollama clients")
    
    args = parser.parse_args()

    main(args)