from openai import OpenAI
import os

# Upstage API 설정
solar_client = OpenAI(
    api_key=os.getenv('UPSTAGE_API_KEY'),  # 여기에 본인의 Upstage API 키를 넣으세요
    base_url="https://api.upstage.ai/v1/solar"
)

def check_embedding_dimension():
    """
    SOLAR Embedding의 차원을 확인하는 함수
    """
    # 테스트용 간단한 문장
    test_sentence = "이것은 SOLAR 임베딩 차원을 테스트하기 위한 문장입니다."
    
    try:
        # passage 임베딩 테스트
        print("1. Passage 임베딩 테스트 중...")
        passage_result = solar_client.embeddings.create(
            model="embedding-passage",
            input=test_sentence
        )
        passage_embedding = passage_result.data[0].embedding
        print(f"Passage 임베딩 차원: {len(passage_embedding)}")
        print(f"Passage 임베딩 타입: {type(passage_embedding)}")
        print(f"Passage 임베딩 처음 5개 값: {passage_embedding[:5]}\n")
        
        # query 임베딩 테스트
        print("2. Query 임베딩 테스트 중...")
        query_result = solar_client.embeddings.create(
            model="embedding-query",
            input=test_sentence
        )
        query_embedding = query_result.data[0].embedding
        print(f"Query 임베딩 차원: {len(query_embedding)}")
        print(f"Query 임베딩 타입: {type(query_embedding)}")
        print(f"Query 임베딩 처음 5개 값: {query_embedding[:5]}\n")
        
        # 두 임베딩이 같은 차원인지 확인
        print("3. 임베딩 차원 비교")
        print(f"Passage와 Query 임베딩의 차원이 같은지: {len(passage_embedding) == len(query_embedding)}")
        
    except Exception as e:
        print(f"임베딩 테스트 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    print("SOLAR 임베딩 차원 테스트를 시작합니다...\n")
    check_embedding_dimension()