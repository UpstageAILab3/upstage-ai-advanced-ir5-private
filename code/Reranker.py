from transformers import pipeline
from ElasticsearchEmbedding import ElasticSearchEmbedding
from openAIEmbedding import OpenAISearch, EvalAnswer
import datetime
from transformers import AutoTokenizer
from pytz import timezone
import json
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self) -> None:
        self.es = ElasticSearchEmbedding()
        self.evaluation = EvalAnswer()
        self.model = CrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-4-v2')
        self.top_n = 5

    def fetch_from_elasticsearch(self, query_str, top_k=10, sort="_score"):
        query = {
            "match": {
                "content": {
                    "query": query_str
                }
            }
        }        

        response = self.es.get_elasticsearch_search(query, top_k, sort=sort)
        return [hit for hit in response['hits']['hits']]

    def normalize_reranker_scores_deberta(self, reranker_scores):
        min_score = min(reranker_scores)
        max_score = max(reranker_scores)

        normalized_scores = [(score - min_score) / (max_score - min_score) * 100 for score in reranker_scores]
        return normalized_scores
    
    def rerank_documents(self, query):
        response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

        retrieved_docs = self.fetch_from_elasticsearch(query)

        pairs = [(query, doc["_source"]["content"]) for doc in retrieved_docs]

        raw_reranked_scores = self.model.predict(pairs)
        retrieved_docs_content = [(doc["_source"]["content"], doc["_id"]) for doc in retrieved_docs]


        # reranked_scores = np.mean(raw_reranked_scores, axis=1).tolist()
        # reranked_scores = self.normalize_reranker_scores(reranked_scores)

        # if isinstance(reranked_scores, np.ndarray):
        #     reranked_scores = reranked_scores.tolist()

        reranked_docs = sorted(
            zip(raw_reranked_scores, retrieved_docs_content), 
            key=lambda x: x[0], 
            reverse=True
        )

        for score, source in reranked_docs[:self.top_n]:

            idx = 0
            response["topk"].append(source[1])
            response["references"].append({
                "score": float(score),
                "content": source[0]
            })
            idx+= 1

        return response


def main():
    reranker = Reranker()
    timestamp = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')

    input_file = "../../../data/translated_eval_origin.jsonl"
    output_file = f"../../../eval_data/rerank_submission{timestamp}.csv"
    new_input_file =f"../../../data/translated_eval_origin_with_isscience_standalone_query_{timestamp}.jsonl"

    try:
        with open(input_file, encoding='utf-8') as f, open(output_file, "w", encoding='utf-8') as out_f, open(new_input_file, "w", encoding='utf-8') as new:
            for idx, line in enumerate(f):
                try:
                    j = json.loads(line)
                    find_out_its_question = reranker.evaluation.find_out_its_question(j["query_translation_msg"])
                    print(f'Processing Question: {find_out_its_question}')
                    function_arguments = json.loads(find_out_its_question)
                    is_science_question = function_arguments["is_science_question"]
                    standalone_query = function_arguments["standalone_query"]

                    if is_science_question and len(j["query_translation_msg"]) > 0:
                        reranker_response = reranker.rerank_documents(standalone_query)
                    else:
                        reranker_response = reranker.rerank_documents(j["query_translation_msg"][0]["content"])

                    response = {
                        "eval_id": j["eval_id"],
                        "standalone_query": standalone_query,
                        "topk": reranker_response["topk"],
                        "answer" : "",
                        "references" : reranker_response["references"]
                    }

                    out_f.write(json.dumps(response, ensure_ascii=False) + "\n")

                    new_line = {
                        "eval_id": j["eval_id"],
                        "msg": j["msg"],
                        "query_translation_msg": j["query_translation_msg"],
                        "is_science_question" : is_science_question,
                        "standalone_query": standalone_query
                    }
                    
                    new.write(json.dumps(new_line, ensure_ascii=False) + "\n")
                
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
