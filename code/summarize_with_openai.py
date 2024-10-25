from openai import OpenAI
import traceback
import os
import json
import datetime
from dotenv import load_dotenv
load_dotenv()

class OpenAISummarization:
    def __init__(self, savefile) -> None:
        self.api_key = os.environ.get("OPENAI_API")
        self.model_name = "gpt-4o"
        self.instructions = "한국어로 추출요약을 하는데, 기존 문장의 반정도의 길이로 요약"
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.documentfile = "../../../data/documents.jsonl"
        self.savefile = savefile
        self.client = OpenAI()
        self.tools = [
            {
                "type": "function",
                "function":{
                    "name": "summary",
                    "description": "extractive_summarization",
                    "parameters": {
                        "properties":{
                            "summary": {
                                "type": "string",
                                "description": "content summary"
                            }
                        },
                        "required": ["summary"],
                        "type": "object",
                        "additionalProperites": False
                    }
                }
            }
        ]

    def summarization(self, given_sentences):
        response = {"summary": ""}
        msg = [{
            "role": "system",
            "content": self.instructions
        },{
            "role": "assistant",
            "content": given_sentences
        }]

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=msg,
                tools=self.tools,
                temperature=0.7,
                seed=1,
                timeout=10
            )
            response["summary"] = json.loads(result.choices[0].message.tool_calls[0].function.arguments)["summary"]
        except Exception as e:
            traceback.print_exc()
            return response
        
        return response
    
    def summary_save_file(self):

        with open(self.documentfile, encoding='utf-8') as doc, open(self.savefile, "a", encoding='utf-8') as sf:
            idx = 0
            for line in doc:
                line_j = json.loads(line)
                print(f'Document {line_j["content"]}')
                response = self.summarization(line_j["content"])
                print(f'Summary: {response}')
                
                combined_dict = {
                    "docid": line_j['docid'],
                    "src": line_j['src'],
                    "content": line_j['content'],
                    "summary": response["summary"]
                }
                print(idx)
                sf.write(f'{json.dumps(combined_dict, ensure_ascii=False)}\n')

            doc.close()
            sf.close()


def __main__():

    openai_summary = OpenAISummarization(f"../../../new_data/openai_data_{datetime.datetime.now()}_gpt_4o.jsonl")

    openai_summary.summary_save_file()

if __name__ == "__main__":
    __main__()