import openai
import json
import os

# GPT API 키 설정
client = openai.OpenAI(api_key= os.getenv('OPENAI_API_KEY'))

def generate_related_document(content):
    # GPT API를 사용하여 관련 문서를 생성하기 위한 프롬프트 정의
    prompt = f"""
    '질문'을 읽고, '관련 문서'를 생성합니다. '관련 문서'의 목적은 사용자가 질문에 대해 스스로 답을 찾을 수 있도록 배경지식을 제공하는 것입니다. 질문에 대한 직접적인 답변이 아닌 문맥적인 정보를 제공하는 문서를 생성합니다. '관련문서'의 길이는 너무 길지 않도록 생성합니다.

    "예시:
    '질문': 식물은 토양에서 발견되는 영양분을 사용합니다. 다음 중 토양에서 분해되어 영양분이 될 수 있는 것은?
    '관련 문서': 지구의 정원 식물은 생존하기 위해 토양, 공기, 물, 햇빛 네 가지 자원이 필요합니다. 이 자원들은 식물이 성장하고 생명을 유지하는 데 필수적입니다. 토양은 식물이 뿌리를 통해 영양분을 흡수하고 뿌리를 고정시키는 역할을 합니다. 공기는 식물이 호흡을 하고 이산화탄소를 흡수하여 산소를 생성하는 데 필요합니다. 물은 식물의 세포 내에서 화학 반응이 일어나는 데 필요한 용매로 작용하며, 식물의 조직을 유지하고 영양분을 운반하는 역할을 합니다. 햇빛은 식물의 광합성에 필요한 에너지원으로 작용하며, 식물이 태양에서 에너지를 흡수하여 이를 식물이 사용하는 형태로 변환합니다. 따라서, 지구의 정원 식물은 이 네 가지 자원이 없으면 생존할 수 없습니다. 이와 마찬가지로, 달이나 다른 행성에서 생명이 존재하기 위해서도 토양, 공기, 물, 햇빛 네 가지 자원이 필요합니다. 따라서, 달이나 다른 행성에서 생명이 존재하기 위해서는 이러한 자원들이 적절하게 제공되어야 합니다.
    "
    
    '질문': {content}
    '관련 문서':
    """
    
    # GPT 모델 호출 (gpt-3.5-turbo 사용)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # GPT-3.5 Turbo 모델 선택
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,  # 최대 토큰 수 설정
        temperature=0.6,  # 출력의 창의성 조정
    )
    
    # GPT의 응답에서 생성된 관련 문서 추출
    return response.choices[0].message.content.strip()

def process_jsonl_file(file_path):
    results = []
    
    # JSONL 파일을 열고 각 라인별로 처리
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)  # 각 라인을 JSON 객체로 변환
            eval_id = entry.get("eval_id")
            msg_content = entry.get("msg")[0].get("content")
            
            # 관련 문서 생성
            related_document = generate_related_document(msg_content)
            
            # 결과를 리스트에 추가
            results.append({
                "eval_id": eval_id,
                "question": msg_content,
                "related_document": related_document
            })
    
    return results

def save_json_file(data, file_path):
    # 데이터를 JSON 파일로 저장하는 함수
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# JSONL 파일 경로
input_file_path = '/data/ephemeral/home/data/eval.jsonl'
output_file_path = '/data/ephemeral/home/data/result.json'  # 결과를 저장할 파일 경로

# JSONL 파일에서 데이터 처리
results = process_jsonl_file(input_file_path)

# 결과를 JSON 파일로 저장
save_json_file(results, output_file_path)