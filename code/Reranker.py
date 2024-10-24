from transformers import pipeline
from ElasticsearchEmbedding import ElasticSearchEmbedding
from openAIEmbedding import OpenAISearch, EvalAnswer
import datetime
from transformers import AutoTokenizer
from pytz import timezone
import json
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self) -> None:
        self.es = ElasticSearchEmbedding()
        self.model = CrossEncoder(model_name='cross-encoder/nli-deberta-v3-base')
        self.top_n = 3

    def fetch_from_elasticsearch(self, query_str, top_k=10, sort="_score"):
        query = {
            "match": {
                "content": {
                    "query": query_str
                }
            }
        }        

        response = self.es.get_elasticsearch_search(query, top_k, sort=sort)
        return [hit["_source"] for hit in response['hits']['hits']]

    def rerank_documents(self, query):
        response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

        retrieved_docs = self.fetch_from_elasticsearch(query)

        pairs = [(query, doc["content"]) for doc in retrieved_docs]
        print(retrieved_docs[0].keys())
        # reranked_scores = self.model.predict(pairs)

        # reranked_docs = sorted(
        #     zip(retrieved_docs, reranked_scores, doc_ids), 
        #     key=lambda x: x[1], 
        #     reverse=True
        # )

        # print(reranked_docs)

def main():
    reranker = Reranker()
    timestamp = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')

    input_file = "../../../data/translated_eval_origin.jsonl"
    output_file = f"../../../eval_data/rerank_submission{timestamp}.csv"

    try:
        with open(input_file, encoding='utf-8') as f, open(output_file, "w", encoding='utf-8') as out_f:
            for idx, line in enumerate(f):
                if idx > 0:
                    break
                try:
                    j = json.loads(line)
                    query = j["query_translation_msg"][0]["content"]
                    print(f'Processing Question: {j["msg"][0]["content"]}')
                    response = reranker.rerank_documents(query)
                    
                    # if response:
                    #     # 응답을 저장하는 부분 추가 (필요에 따라 수정)
                    #     out_f.write(json.dumps(response, ensure_ascii=False) + "\n")
                
                except json.JSONDecodeError as json_err:
                    print(f"JSON decode error at line {idx}: {json_err}")
                except Exception as e:
                    print(f"Error at line {idx}: {e}")

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
