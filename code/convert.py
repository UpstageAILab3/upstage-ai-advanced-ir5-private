import json

def convert_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 각 줄을 JSON 객체로 파싱
            data = json.loads(line.strip())
            
            # 새로운 형식으로 변환
            new_format = {
                "eval_id": data["eval_id"],
                "msg": [
                    {
                        "role": "user",
                        "content": data["related_document"]
                    }
                ]
            }
            
            # 변환된 데이터를 새 파일에 쓰기
            json.dump(new_format, outfile, ensure_ascii=False)
            outfile.write('\n')  # 각 JSON 객체를 새 줄에 작성

# 스크립트 사용
input_file = '/data/ephemeral/home/data/eval_doc.jsonl'  # 입력 파일 이름
output_file = '/data/ephemeral/home/data/eval_doc_conv.jsonl'  # 출력 파일 이름
convert_format(input_file, output_file)