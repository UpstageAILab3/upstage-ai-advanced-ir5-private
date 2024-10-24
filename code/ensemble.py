import json
import numpy as np
from collections import defaultdict

def softmax(x):
    """소프트맥스 함수"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def load_results(file_path):
    """결과 파일 로드"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

def ensemble_results(file_weights):
    """파일별 결과를 앙상블"""
    # 모든 파일의 결과 로드
    all_results = {}
    for file_path, weight in file_weights.items():
        all_results[file_path] = load_results(file_path)
    
    # eval_id별로 결과 통합
    final_results = []
    
    # 첫 번째 파일의 결과 수만큼 반복
    for idx in range(len(all_results[list(file_weights.keys())[0]])):
        eval_id = all_results[list(file_weights.keys())[0]][idx]["eval_id"]
        
        # 모든 모델의 해당 eval_id 결과 수집
        doc_scores = defaultdict(list)  # 문서별 점수 저장
        doc_contents = {}  # 문서 내용 저장
        query = ""  # standalone_query 저장
        
        # 각 모델의 결과 처리
        for file_path, weight in file_weights.items():
            result = all_results[file_path][idx]
            query = result["standalone_query"]  # 가장 마지막 쿼리 사용
            
            # 각 문서의 점수와 내용 저장
            for rank, (doc_id, ref) in enumerate(zip(result["topk"], result["references"])):
                doc_scores[doc_id].append((ref["score"] * weight, rank))
                doc_contents[doc_id] = ref["content"]
        
        # 각 문서의 최종 점수 계산
        final_scores = {}
        for doc_id, scores_ranks in doc_scores.items():
            scores = [s[0] for s in scores_ranks]
            ranks = [r[1] for r in scores_ranks]
            # 점수와 순위를 모두 고려
            final_scores[doc_id] = np.mean(scores) / (np.mean(ranks) + 1)
        
        # 상위 3개 문서 선택
        top_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 최종 결과 생성
        final_result = {
            "eval_id": eval_id,
            "standalone_query": query,
            "topk": [doc_id for doc_id, _ in top_docs],
            "answer": "",
            "references": [
                {
                    "score": score,
                    "content": doc_contents[doc_id]
                }
                for doc_id, score in top_docs
            ]
        }
        
        final_results.append(final_result)
    
    return final_results

# 파일 경로와 점수 설정
file_weights = {
    "/data/ephemeral/home/data/ensemble/output (2).csv": 0.8424,
    "/data/ephemeral/home/data/ensemble/output (3).csv": 0.8364,
    "/data/ephemeral/home/data/ensemble/output (4).csv": 0.8318,
    "/data/ephemeral/home/data/ensemble/output (5).csv": 0.7970,
    "/data/ephemeral/home/data/ensemble/output (3).csv": 0.7439
}

# 결과 앙상블
final_results = ensemble_results(file_weights)

# 결과 저장
output_path = "/data/ephemeral/home/data/ensemble/final_ensemble.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for result in final_results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"앙상블 결과가 {output_path}에 저장되었습니다.")