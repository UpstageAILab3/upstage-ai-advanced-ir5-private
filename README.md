# Let's create the updated README.md file with the provided and expanded content.

readme_content_updated = """
# IR: Scientific Knowledge Question Answering

## 팀

| 팀장 | 홍진영 | 이승민 |
| :---: | :---: | :---: |
| [한지웅](https://github.com/UpstageAILab) | [홍진영](https://github.com/UpstageAILab) | [이승민](https://github.com/UpstageAILab) |
| 팀장 | 팀원 | 팀원 |

## 0. 개요
**개발 환경**:  
- 윈도우, 리눅스(서버), VS Code 등 다양한 환경에서 개발을 진행.  
**필수 사항**:  
- Python, OpenAI API, Upstage 플랫폼, Elasticsearch 등.

## 1. 대회 정보
**개요**:  
- LLM(대형 언어 모델)의 대표적 약점인 할루시에이션 문제를 해결하기 위해, RAG(검색 기반 생성)을 구현하는 프로젝트입니다.  
- 4200여 개의 과학 상식 문서와 200개 이상의 질문이 포함된 JSONL 파일로 데이터가 제공됩니다.

**타임라인**:  
- 2024.10.02 - 대회 시작  
- 2024.10.24 - 최종 제출 마감

## 2. 구성 요소
_(이 부분은 이후 단계에서 구체적으로 추가)_

## 3. 데이터 설명
**데이터셋 개요**:  
- 'doc_id': 문서별 UUID로 부여된 고유 식별자.  
- 'src': 데이터 출처를 나타내는 필드.  
- 'content': RAG에서 참조할 수 있는 지식 정보가 포함된 필드.

**EDA**: 
- 'is_science_question': 과학질의인지 분류
- 'document_msg' : 질문의 문서화

**데이터 처리**: 
- Hyde를 위하여 질문을 문서화하는 작업을 'document_msg' 키에 저장
- 영어로 역번역하여 임베딩하는 경우 openai 임베딩 모델 사용
- eval.jsonl의 멀티턴 대화 싱글턴으로 바꾸기

## 7. 각자 임무
- **한지웅**: 과학질문 20개 선처리하기, Upstage embedding 모델 적용하기, Sparse/Dense 검색을 합친 Hybride Retrieval 구현하기, HYDE 적용하기(query로부터 doc 생성하기), 멀티턴 대화 싱글턴으로 바꾸기, Rerank 시도하기

- **홍진영**: 한글 문서를 klue 모델로 summarize, 문서를 영어로 역번역하여 Hyde, Rerank(Cross Encoder), Elastics Search에 영어 질문 및 문서 인덱싱 진행. 

- **이승민**: 소프트 보팅 기법을 적용하여 다수의 모델에서 나온 예측을 결합해 최종 결과를 도출하는 작업을 맡고 있습니다. 이 과정에서 Solar API를 사용해 데이터를 처리하며, jhgan/ko-sroberta-nli와 jhgan/ko-sbert-nli 모델을 활용하여 문장 임베딩 및 문서 검색 성능을 향상시키고 있습니다. 이를 통해 다양한 모델의 강점을 결합하여 보다 정확한 답변을 생성할 수 있도록 기여하고 있습니다.
"""

# Define the file path to save the updated README file
updated_file_path = '/mnt/data/README_updated.md'

# Write the updated content to the new file
with open(updated_file_path, 'w', encoding='utf-8') as file:
    file.write(readme_content_updated)

updated_file_path  # Return the file path for user to download the updated file
