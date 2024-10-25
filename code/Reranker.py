from transformers import pipeline
from ElasticsearchEmbedding import ElasticSearchEmbedding
from openAIEmbedding import OpenAISearch, EvalAnswer
import datetime
from transformers import AutoTokenizer
from pytz import timezone
import json
from sentence_transformers import CrossEncoder
import numpy as np
from SoftVoting import SortVotingOneDocument


class Reranker:
    def __init__(self) -> None:
        self.esembedding = ElasticSearchEmbedding()
        self.essearch =OpenAISearch(search_doc="../../../new_data/translated_document.jsonl",)
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

        response = self.esembedding.get_elasticsearch_search(query, top_k, sort=sort)
        return [hit for hit in response['hits']['hits']]
    
    def fetch_from_elasticsearch_dense(self, query_str, top_k=10, sort="_score"):
        
        response = self.essearch.dense_retrieve(query_str=query_str)
        return response
    
    def normalize_reranker_scores_deberta(self, reranker_scores):
        min_score = min(reranker_scores)
        max_score = max(reranker_scores)

        normalized_scores = [(score - min_score) / (max_score - min_score) * 100 for score in reranker_scores]
        return normalized_scores
    
    def rerank_documents_dense(self, query):
        response = { "topk": [], "references": [], "answer": ""}

        retrieved_docs = self.fetch_from_elasticsearch_dense(query)
        print(retrieved_docs)


    
    def rerank_documents(self, query, idx):
        response = { "topk": [], "references": [], "answer": ""}

        retrieved_docs = self.fetch_from_elasticsearch(query)

        pairs = [(query, doc["_source"]["content"]) for doc in retrieved_docs]

        raw_reranked_scores = self.model.predict(pairs)
        retrieved_docs_content = [(doc["_source"]["content"], doc["_id"]) for doc in retrieved_docs]

        set_score = list(zip(raw_reranked_scores, retrieved_docs_content))

        selected_indice = SortVotingOneDocument(set_score).selected_indice()
        
        selected_docs = [set_score[i] for i in selected_indice]
        # reranked_scores = np.mean(raw_reranked_scores, axis=1).tolist()
        # reranked_scores = self.normalize_reranker_scores(reranked_scores)

        # if isinstance(reranked_scores, np.ndarray):
        #     reranked_scores = reranked_scores.tolist()
        # print(selected_docs)
        # reranked_docs = sorted(
        #     set_score, 
        #     key=lambda x: x[0], 
        #     reverse=True
        # )



        for score, source in selected_docs[:self.top_n]:

            response["topk"].append(source[1])
            response["references"].append({
                "score": float(score),
                "content": source[0]
            })

        return response

    def rerank_hyde(self, query, idx):
        response = { "topk": [], "references": [], "answer": ""}

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

            response["topk"].append(source[1])
            response["references"].append({
                "score": float(score),
                "content": source[0]
            })

        return response


def main():
    reranker = Reranker()
    timestamp = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')

    input_file = "../../../data/translated_eval_origin_with_isscience_standalone_query_20241024_124841_change.jsonl"
    output_file = f"../../../eval_data/rerank_softvoting_onedoc_submission_{timestamp}.csv"

    # try:
    #     with open(input_file, encoding='utf-8') as f, open(output_file, "w", encoding='utf-8') as out_f:
    #         for idx, line in enumerate(f):
    #             try:
    #                 j = json.loads(line)
                    
    #                 is_science_question = j["is_science_question"]

    #                 query = ""
    #                 if len(j["query_translation_msg"]) > 1:
    #                     query = j["standalone_query"]
    #                 else:
    #                     query = j["query_translation_msg"][0]["content"]

    #                 reranker_response = reranker.rerank_documents_dense(query)

    #                 # if is_science_question:
    #                 #     response = {
    #                 #         "eval_id": j["eval_id"],
    #                 #         "standalone_query": query,
    #                 #         "topk": reranker_response["topk"],
    #                 #         "answer" : "",
    #                 #         "references" : reranker_response["references"]
    #                 #     }
    #                 # else:
    #                 #     response = {
    #                 #         "eval_id": j["eval_id"],
    #                 #         "standalone_query": query,
    #                 #         "topk": [],
    #                 #         "answer" : "",
    #                 #         "references" : []
    #                 #     }

    #                 # out_f.write(json.dumps(response, ensure_ascii=False) + "\n")

                
    #             except json.JSONDecodeError as json_err:
    #                 print(f"JSON decode error at line {idx}: {json_err}")
    #             except Exception as e:
    #                 print(f"Error at line {idx}: {e}")
    #             break
    # except FileNotFoundError as fnf_error:
    #     print(f"File not found: {fnf_error}")
    # except Exception as e:
    #     print(f"Unexpected error: {e}")


    try:
        with open(input_file, encoding='utf-8') as f, open(output_file, "w", encoding='utf-8') as out_f:
            for idx, line in enumerate(f):
                try:
                    j = json.loads(line)
                    
                    is_science_question = j["is_science_question"]

                    query = ""
                    if len(j["query_translation_msg"]) > 1:
                        query = j["standalone_query"]
                    else:
                        query = j["query_translation_msg"][0]["content"]

                    reranker_response = reranker.rerank_documents(query, idx)
                    if is_science_question:
                        response = {
                            "eval_id": j["eval_id"],
                            "standalone_query": query,
                            "topk": reranker_response["topk"],
                            "answer" : "",
                            "references" : reranker_response["references"]
                        }
                    else:
                        response = {
                            "eval_id": j["eval_id"],
                            "standalone_query": query,
                            "topk": [],
                            "answer" : "",
                            "references" : []
                        }

                    out_f.write(json.dumps(response, ensure_ascii=False) + "\n")

                
                except json.JSONDecodeError as json_err:
                    print(f"JSON decode error at line {idx}: {json_err}")
                except Exception as e:
                    print(f"Error at line {idx}: {e}")
                
    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except Exception as e:
        print(f"Unexpected error: {e}")


    # try:
    #     with open(input_file, encoding='utf-8') as f, open(output_file, "w", encoding='utf-8') as out_f:
    #         for idx, line in enumerate(f):
    #             try:
    #                 j = json.loads(line)
                    
    #                 is_science_question = j["is_science_question"]
                    
    #                 query = j["documentation_msg"]
    #                 reranker_response = reranker.rerank_documents(query, idx)


    #                 if is_science_question:
    #                     response = {
    #                         "eval_id": j["eval_id"],
    #                         "standalone_query": query,
    #                         "topk": reranker_response["topk"],
    #                         "answer" : "",
    #                         "references" : reranker_response["references"]
    #                     }
    #                 else:
    #                     response = {
    #                         "eval_id": j["eval_id"],
    #                         "standalone_query": query,
    #                         "topk": [],
    #                         "answer" : "",
    #                         "references" : []
    #                     }

    #                 out_f.write(json.dumps(response, ensure_ascii=False) + "\n")

                
    #             except json.JSONDecodeError as json_err:
    #                 print(f"JSON decode error at line {idx}: {json_err}")
    #             except Exception as e:
    #                 print(f"Error at line {idx}: {e}")

    # except FileNotFoundError as fnf_error:
    #     print(f"File not found: {fnf_error}")
    # except Exception as e:
    #     print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
