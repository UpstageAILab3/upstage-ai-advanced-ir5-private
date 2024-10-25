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

**EDA**: _(생략)_  
**데이터 처리**: _(생략)_

## 7. 각자 임무
- **한지웅**: (업무 내용 추가해주세요)

- **홍진영**: (업무 내용 추가해주세요)

- **이승민**: 소프트 보팅 기법을 적용하여 다수의 모델에서 나온 예측을 결합해 최종 결과를 도출하는 작업을 맡고 있습니다. 이 과정에서 Solar API를 사용해 데이터를 처리하며, jhgan/ko-sroberta-nli와 jhgan/ko-sbert-nli 모델을 활용하여 문장 임베딩 및 문서 검색 성능을 향상시키고 있습니다. 이를 통해 다양한 모델의 강점을 결합하여 보다 정확한 답변을 생성할 수 있도록 기여하고 있습니다.
"""

# Define the file path to save the updated README file
updated_file_path = '/mnt/data/README_updated.md'

# Write the updated content to the new file
with open(updated_file_path, 'w', encoding='utf-8') as file:
    file.write(readme_content_updated)

updated_file_path  # Return the file path for user to download the updated file
