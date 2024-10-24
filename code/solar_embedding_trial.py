import os
import json
from elasticsearch import Elasticsearch, helpers
import numpy as np
from openai import OpenAI
import traceback

# Upstage API 설정
# OpenAI 클라이언트 설정 수정
solar_client = OpenAI(
    api_key= os.getenv('UPSTAGE_API_KEY'),  # Upstage API 키
    base_url="https://api.upstage.ai/v1/solar"
)

def get_embedding(sentences):
    """
    OpenAI 클라이언트를 사용하여 SOLAR Embedding 생성
    """
    embeddings = []
    for sentence in sentences:
        try:
            result = solar_client.embeddings.create(
                model="embedding-passage",  # 문서 임베딩용
                input=sentence
            )
            embeddings.append(result.data[0].embedding)
        except Exception as e:
            print(f"임베딩 생성 중 오류: {e}")
            raise
            
    return np.array(embeddings)

def get_query_embedding(query):
    """
    쿼리용 임베딩 생성
    """
    try:
        result = solar_client.embeddings.create(
            model="embedding-query",  # 쿼리 임베딩용
            input=query
        )
        return np.array(result.data[0].embedding)
    except Exception as e:
        print(f"쿼리 임베딩 생성 중 오류: {e}")
        raise


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    """
    문서 리스트에서 배치 단위로 임베딩 생성
    """
    batch_embeddings = []
    contents = [doc["content"] for doc in docs]
    total_batches = (len(contents) - 1) // batch_size + 1
    
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        try:
            embeddings = get_embedding(batch)
            batch_embeddings.extend(embeddings)
            print(f'Processed batch {i//batch_size + 1}/{total_batches}')
        except Exception as e:
            print(f"배치 {i//batch_size + 1} 처리 중 오류: {e}")
            raise
            
    return batch_embeddings

# 새로운 index 생성
def create_es_index(index, settings, mappings):
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)

# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)

# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)

# 역색인을 이용한 검색
def sparse_retrieve(query_str, size=3):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


def hybrid_retrieve(query_str, size=3, sparse_weight=0.3, dense_weight=0.7):
    """
    sparse 검색과 dense 검색을 결합한 하이브리드 검색
    """
    # 각각의 검색 수행
    sparse_results = sparse_retrieve(query_str, size*2)
    dense_results = dense_retrieve(query_str, size*2)
    
    # 결과 통합을 위한 dict
    combined_results = {}
    
    # Sparse 결과 처리 (BM25 스코어 정규화)
    max_sparse_score = max(hit['_score'] for hit in sparse_results['hits']['hits']) if sparse_results['hits']['hits'] else 1
    for hit in sparse_results['hits']['hits']:
        doc_id = hit['_id']
        normalized_score = hit['_score'] / max_sparse_score * sparse_weight
        if doc_id not in combined_results:
            combined_results[doc_id] = {
                'content': hit['_source']['content'],
                'docid': hit['_source'].get('docid', ''),
                'score': normalized_score
            }
            
    # Dense 결과 처리 (코사인 유사도 스코어 정규화)
    max_dense_score = max(hit['_score'] for hit in dense_results['hits']['hits']) if dense_results['hits']['hits'] else 1
    for hit in dense_results['hits']['hits']:
        doc_id = hit['_id']
        normalized_score = hit['_score'] / max_dense_score * dense_weight
        if doc_id in combined_results:
            combined_results[doc_id]['score'] += normalized_score
        else:
            combined_results[doc_id] = {
                'content': hit['_source']['content'],
                'docid': hit['_source'].get('docid', ''),
                'score': normalized_score
            }
    
    # 최종 점수로 정렬
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['score'], reverse=True)
    return {
        'hits': {
            'hits': [
                {'_id': doc_id, '_score': info['score'], '_source': {'content': info['content'], 'docid': info['docid']}}
                for doc_id, info in sorted_results[:size]
            ]
        }
    }

def rerank_by_context(results, query_str):
    """
    검색 결과를 컨텍스트 유사도에 기반하여 재순위화
    """
    reranked = []
    query_tokens = set(query_str.split())
    
    for hit in results['hits']['hits']:
        content = hit['_source']['content']
        content_tokens = set(content.split())
        
        # 컨텍스트 관련성 점수 계산
        context_score = len(query_tokens & content_tokens) / len(query_tokens)
        
        # 최종 점수 계산 (원래 점수 + 컨텍스트 점수)
        final_score = hit['_score'] * 0.7 + context_score * 0.3
        
        reranked.append({
            '_id': hit['_id'],
            '_score': final_score,
            '_source': hit['_source']
        })
    
    # 재순위화된 결과 정렬
    reranked.sort(key=lambda x: x['_score'], reverse=True)
    return {'hits': {'hits': reranked}}

# Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size=3):
    query_embedding = get_query_embedding(query_str)
    query_vector = query_embedding.tolist()
    
    knn = [
        {
            "field": "embeddings_1",
            "query_vector": query_vector[:2048],
            "k": size,
            "num_candidates": size * 2
        },
        {
            "field": "embeddings_2",
            "query_vector": query_vector[2048:],
            "k": size,
            "num_candidates": size * 2
        }
    ]
    
    # 두 부분의 유사도 점수를 평균내서 사용
    script_score = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.queryVector1, 'embeddings_1') + cosineSimilarity(params.queryVector2, 'embeddings_2')",
                "params": {
                    "queryVector1": query_vector[:2048],
                    "queryVector2": query_vector[2048:]
                }
            }
        }
    }
    
    return es.search(index="test", query=script_score, size=size)

# Elasticsearch 설정
es_username = "elastic"
es_password = os.getenv('ELASTIC_PASSWORD')
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt")

print(es.info())

# 매핑 설정 (SOLAR 임베딩용)
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings_1": {
            "type": "dense_vector",
            "dims": 2048,
            "index": True,
            "similarity": "cosine"
        },
        "embeddings_2": {
            "type": "dense_vector",
            "dims": 2048,
            "index": True,
            "similarity": "cosine"
        }
    }
}

# Nori 분석기 설정
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

# 인덱스 생성 및 문서 처리
create_es_index("test", settings, mappings)

# 문서 임베딩 및 인덱싱
index_docs = []
with open("/data/ephemeral/home/data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]

embeddings = get_embeddings_in_batches(docs)

for doc, embedding in zip(docs, embeddings):
    # 4096 차원 벡터를 2048씩 두 부분으로 나눔
    embedding_list = embedding.tolist()
    doc["embeddings_1"] = embedding_list[:2048]
    doc["embeddings_2"] = embedding_list[2048:]
    index_docs.append(doc)

ret = bulk_add("test", index_docs)
print(ret)

# OpenAI 설정
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

client = OpenAI()
llm_model = "gpt-4o"

# 프롬프트 설정
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성해.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답해.
- 한국어로 답변을 생성해.
- 양질의 결과물 생성시 너는 50$의 보상을 받아.
"""

persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 사용자가 대화를 통해 과학 지식에 관한 내용을 말하면 search api를 호출해.
- 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성해.
- 답변형태는 줄글의 형태로 설명해. 항목별 설명은 하지마.
- 양질의 결과물 생성시 너는 50$의 보상을 받아.
"""

# Function calling 설정
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query suitable for use in search from the user messages history."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    }
]

# answer_question 함수를 다음과 같이 수정
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")
        
        # 하이브리드 검색 수행
        search_result = hybrid_retrieve(standalone_query)
        
        # 컨텍스트 기반 재순위화
        reranked_result = rerank_by_context(search_result, standalone_query)
        
        response["standalone_query"] = standalone_query
        retrieved_context = []
        
        for rst in reranked_result['hits']['hits']:
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({
                "score": rst["_score"],
                "content": rst["_source"]["content"]
            })

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                model=llm_model,
                messages=msg,
                temperature=0,
                seed=1,
                timeout=30
            )
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message.content

    else:
        response["answer"] = result.choices[0].message.content

    return response

# 평가 실행
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            if idx > 5:
                break
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question([{"role": "user", "content": j["msg"]}])
            print(f'Answer: {response["answer"]}\n')

            output = {
                "eval_id": j["eval_id"],
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 실행
eval_rag("/data/ephemeral/home/data/ver4/ver4_final.jsonl", "submission_ver4.csv")