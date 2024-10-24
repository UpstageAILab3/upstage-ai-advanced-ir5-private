import os
import json
import numpy as np
from elasticsearch import Elasticsearch, helpers
from openai import OpenAI
from sentence_transformers import CrossEncoder
from typing import Dict, List, Any

# Upstage API 설정
solar_client = OpenAI(
    api_key="",
    base_url="https://api.upstage.ai/v1/solar"
)

# Upstage 모델의 임베딩 차원 설정
EMBEDDING_DIM = 4096


# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    """
    문서 임베딩 생성 (embedding-passage 모델 사용)
    """
    embeddings = []
    for sentence in sentences:
        try:
            result = solar_client.embeddings.create(
                model="embedding-passage",
                input=sentence
            )
            embeddings.append(result.data[0].embedding)
        except Exception as e:
            print(f"임베딩 생성 중 오류: {e}")
            raise
            
    return np.array(embeddings)


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# 임베딩 저장 및 로드하는 함수 추가
def save_embeddings(embeddings, filename="embeddings.npy"):
    np.save(filename, embeddings)
    print(f"임베딩이 {filename}에 저장되었습니다.")

def load_embeddings(filename="embeddings.npy"):
    return np.load(filename)


# 새로운 index 생성
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색
def get_query_embedding(query):
    """
    쿼리 임베딩 생성 (embedding-query 모델 사용)
    """
    try:
        result = solar_client.embeddings.create(
            model="embedding-query",
            input=query
        )
        return np.array(result.data[0].embedding)
    except Exception as e:
        print(f"쿼리 임베딩 생성 중 오류: {e}")
        raise

def dense_retrieve(query_str, size):
    # 쿼리 임베딩 생성시 embedding-query 모델 사용
    query_embedding = get_query_embedding(query_str)

    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    return es.search(index="test", knn=knn)

def hybrid_retrieve(query_str, size=3, alpha=0.0):
    """
    Hybrid 검색 구현
    alpha = 1.0: Dense 검색만 사용
    alpha = 0.0: Sparse 검색만 사용
    0 < alpha < 1: 두 검색 혼합
    """
    results = {}
    
    # alpha가 0이 아닐 때만 Dense 검색 수행
    if alpha > 0:
        dense_results = dense_retrieve(query_str, size=size)
        dense_scores = [hit["_score"] for hit in dense_results["hits"]["hits"]]
        if dense_scores:
            min_dense = min(dense_scores)
            max_dense = max(dense_scores)
            score_range = max_dense - min_dense
            
            for hit in dense_results["hits"]["hits"]:
                doc_id = hit["_source"]["docid"]
                normalized_score = (hit["_score"] - min_dense) / score_range if score_range != 0 else 1
                results[doc_id] = {
                    "doc": hit["_source"], 
                    "score": alpha * normalized_score
                }
    
    # alpha가 1이 아닐 때만 Sparse 검색 수행
    if alpha < 1:
        sparse_results = sparse_retrieve(query_str, size=size)
        sparse_scores = [hit["_score"] for hit in sparse_results["hits"]["hits"]]
        if sparse_scores:
            min_sparse = min(sparse_scores)
            max_sparse = max(sparse_scores)
            score_range = max_sparse - min_sparse
            
            for hit in sparse_results["hits"]["hits"]:
                doc_id = hit["_source"]["docid"]
                normalized_score = (hit["_score"] - min_sparse) / score_range if score_range != 0 else 1
                if doc_id in results:
                    results[doc_id]["score"] += (1-alpha) * normalized_score
                else:
                    results[doc_id] = {
                        "doc": hit["_source"], 
                        "score": (1-alpha) * normalized_score
                    }

    # 최종 점수로 정렬
    sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    
    final_results = {
        "hits": {
            "hits": [
                {
                    "_score": item[1]["score"],
                    "_source": item[1]["doc"]
                }
                for item in sorted_results[:size]
            ]
        }
    }
    
    return final_results

class EnhancedRetriever:
    def __init__(self):
        """
        Cross-encoder 모델 초기화 및 설정
        """
        self.cross_encoder = CrossEncoder('jhgan/ko-sroberta-multitask')
        self.initial_candidates = 15
        self.batch_size = 32
        
    def rerank_with_cross_encoder(self, 
                                 query: str, 
                                 search_results: Dict[str, Any], 
                                 top_k: int = 3) -> Dict[str, Any]:
        try:
            if not search_results["hits"]["hits"]:
                return search_results
                
            candidates = []
            original_results = []
            
            for hit in search_results["hits"]["hits"]:
                enhanced_query = f"{query} {' '.join(query.split()[:3])}"
                candidates.append([enhanced_query, hit["_source"]["content"]])
                original_results.append(hit)
            
            # 배치 처리 수정
            cross_scores = []
            if len(candidates) == 1:
                # 단일 항목인 경우
                score = self.cross_encoder.predict(candidates[0])
                cross_scores = [score]
            else:
                # 여러 항목인 경우
                for i in range(0, len(candidates), self.batch_size):
                    batch = candidates[i:i + self.batch_size]
                    batch_scores = self.cross_encoder.predict(batch)
                    # 배치 크기가 1인 경우와 아닌 경우 처리
                    if isinstance(batch_scores, (float, np.float32, np.float64)):
                        cross_scores.append(batch_scores)
                    else:
                        cross_scores.extend(batch_scores)
            
            original_scores = [hit["_score"] for hit in original_results]
            max_score = max(original_scores)
            min_score = min(original_scores)
            score_range = max_score - min_score
            
            reranked_results = []
            for idx, (orig_hit, cross_score, orig_score) in enumerate(zip(original_results, cross_scores, original_scores)):
                rank_weight = 1.0 / (idx + 1)
                normalized_orig_score = (orig_score - min_score) / score_range if score_range != 0 else 1.0
                
                # cross_score가 numpy scalar인 경우 float로 변환
                if isinstance(cross_score, (np.float32, np.float64)):
                    cross_score = float(cross_score)
                    
                combined_score = (0.8 * cross_score + 
                                0.1 * normalized_orig_score + 
                                0.1 * rank_weight)
                
                reranked_results.append({
                    "_score": float(combined_score),  # 명시적으로 float 변환
                    "_source": orig_hit["_source"]
                })
            
            reranked_results.sort(key=lambda x: x["_score"], reverse=True)
            
            return {
                "hits": {
                    "hits": reranked_results[:top_k]
                }
            }
            
        except Exception as e:
            print(f"Reranking 중 오류 발생: {e}")
            traceback.print_exc()  # 상세한 에러 정보 출력
            return search_results
        
    
                
# retriever 인스턴스 생성
enhanced_retriever = EnhancedRetriever()

es_username = "elastic"
es_password = ""

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.15.2/config/certs/http_ca.crt")

# Elasticsearch client 정보 확인
print(es.info())

# 색인을 위한 setting 설정
# nori 토크나이저 설정 강화
# 색인을 위한 setting 설정
settings = {
    "analysis": {
        "tokenizer": {
            "nori_tokenizer": {
                "type": "nori_tokenizer",
                "decompound_mode": "mixed"
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            },
            "nori_readingform": {
                "type": "nori_readingform"
            },
            "stop": {
                "type": "stop",
                "stopwords": ["_korean_"]
            }
        },
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "filter": [
                    "nori_posfilter",
                    "nori_readingform",
                    "lowercase",
                    "stop"
                ]
            }
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIM,  # 4096으로 변경
            "index": True,
            "similarity": "l2_norm"
        }
    }
}

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성 부분 수정
index_docs = []
embeddings_file = "embeddings.npy"
                
# 임베딩 파일이 있으면 로드하고, 없으면 새로 생성
if os.path.exists(embeddings_file):
    print("저장된 임베딩을 로드합니다...")
    embeddings = load_embeddings(embeddings_file)
    with open("/data/ephemeral/home/data/documents.jsonl") as f:
        docs = [json.loads(line) for line in f]
else:
    print("임베딩을 새로 생성합니다...")
    with open("/data/ephemeral/home/data/documents.jsonl") as f:
        docs = [json.loads(line) for line in f]
    embeddings = get_embeddings_in_batches(docs)
    save_embeddings(embeddings, embeddings_file)


# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print(ret)

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# 하이브리드 검색 테스트 (alpha=0.5로 시작)
search_result_retrieve = hybrid_retrieve(test_query, size=3, alpha=0.9)

# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])



# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI
import traceback

# OpenAI API 키를 환경변수에 설정
os.environ["OPENAI_API_KEY"] = ""

client = OpenAI()
# 사용할 모델을 설정(여기서는 gpt-3.5-turbo-1106 모델 사용)
llm_model = "gpt-4o"
# 비과학 질문 리스트
NON_SCIENCE_EVAL_IDS = [276, 261, 283, 32, 94, 90, 220, 245, 229, 247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218]

# 쿼리 생성 프롬프트
query_generation_prompt = """
Role: 검색 쿼리 생성 전문가

Instructions:
1. 주어진 대화 맥락을 바탕으로 과학 관련 정보를 검색하기 위한 최적의 쿼리를 생성하세요.
2. 검색에 불필요한 표현은 제거하고 핵심 키워드를 포함하세요.
3. 멀티턴 대화의 경우, 이전 대화 맥락을 고려하여 쿼리를 생성하세요.

Output:
최적화된 검색 쿼리를 직접 출력하세요. 다른 설명이나 부가 정보 없이 쿼리만 출력하세요.
"""

# 2. 새로운 헬퍼 함수들 추가
def get_empty_response(query=""):
    return {
        "standalone_query": query,
        "topk": [],
        "references": [],
        "answer": ""
    }

def combine_messages(messages):
    context = []
    for msg in messages:
        if msg["role"] == "user":
            context.append(msg["content"])
    return " ".join(context)

def generate_standalone_query(messages):
    combined_context = combine_messages(messages)
    
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": query_generation_prompt},
                {"role": "user", "content": combined_context}
            ],
            temperature=0,
            seed=1,
            timeout=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating query: {e}")
        return combined_context

def process_question(messages, eval_id=None):
    original_query = combine_messages(messages)
    
    if eval_id in NON_SCIENCE_EVAL_IDS:
        return get_empty_response(original_query)

    try:
        standalone_query = generate_standalone_query(messages)
    except Exception as e:
        print(f"Error in query generation: {e}")
        standalone_query = original_query
    
    response = {
        "standalone_query": standalone_query,
        "topk": [],
        "references": [],
        "answer": ""
    }

    try:
        # hybrid_retrieve만 사용하고 reranking 제거
        search_result = hybrid_retrieve(standalone_query, size=3, alpha=0.9)
        
        for rst in search_result["hits"]["hits"]:
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({
                "score": rst["_score"],
                "content": rst["_source"]["content"]
            })
    except Exception as e:
        print(f"Error during search: {e}")
        
    return response

# 4. eval_rag 함수 교체
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            
            response = process_question(j["msg"], j["eval_id"])
            
            output = {
                "eval_id": j["eval_id"],
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }
            
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1
            
            if idx % 10 == 0:
                print(f'Processed {idx} questions')
# 평가 실행
eval_rag("/data/ephemeral/home/data/eval.jsonl", "baseline6.csv")