import os
from openai import OpenAI
import traceback
import json
from dotenv import load_dotenv
import datetime

load_dotenv()

class OpenAITranslation:
    def __init__(self, original_location, save_location, doc_type) -> None:
        self.doc_type = doc_type
        self.original_location = original_location
        self.save_location =save_location
        self.api_key = os.environ.get("OPENAI_API")
        self.model_name = "gpt-3.5-turbo-1106"
        # self.model_name = "gpt-4o"
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.client = OpenAI()

        self.tools = [
            {
                "type": "function",
                "function":{
                    "name": "translation",
                    "description": "from korean to english translation",
                    "parameters": {
                        "properties":{
                            "translation": {
                                "type": "string",
                                "description": "korean content to english translation"
                            }
                        },
                        "required": ["translation"],
                        "type": "object",
                        "additionalProperites": False
                    }
                }
            }
        ]

    def translastion_line(self, korean_sentence):

        response = {"translated": ""}
        msg = [
            {"role": "system", "content": "You are a helpful translation from korean to enlish"},
            {"role": "user", "content": korean_sentence}
        ]

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=msg,
                tools=self.tools,
                temperature=0.6,
                seed=1,
                timeout=20,
                stop=None
            )
            response["translated"] = json.loads(result.choices[0].message.tool_calls[0].function.arguments)["translation"]


        except Exception as e:
            traceback.print_exc()
            return response
        
        print("result: ",result)
        return response
    
    def save_file(self):
        if self.doc_type == "document":
            with open(self.original_location, encoding='utf-8') as original, open(self.save_location, 'w', encoding='utf-8') as save_file:
                for line in original:
                    line_j = json.loads(line)
                    
                    translated_summary = self.translastion_line(line_j["content"])

                    combined_dict = {
                        "docid": line_j['docid'],
                        "src": line_j['src'],
                        "content": line_j['content'],
                        "summary_in_english": translated_summary["translated"]
                    }
                    print("origin: ", line_j["content"])
                    print("transle: ", translated_summary["translated"])
                    save_file.write(f'{json.dumps(combined_dict, ensure_ascii=False)}\n')

        elif self.doc_type == "eval":
            with open(self.original_location, encoding='utf-8') as original, open(self.save_location, 'w', encoding='utf-8') as save_file:
                idx = 0
                for line in original:
                    line_j = json.loads(line)
                    translated_msg = []
                    for query in line_j["msg"]:
                        msg_line = {"role": query["role"], "content":""}
                        msg_line["content"] = self.translastion_line(query["content"])["translated"]
                        translated_msg.append(msg_line)

                    combined_dict = {
                        "eval_id": line_j["eval_id"],
                        "msg": line_j["msg"],
                        "query_translation_msg": translated_msg
                    }

                    print("origin: ", line_j["msg"])
                    print("translate: ", combined_dict["query_translation_msg"])
                    save_file.write(f'{json.dumps(combined_dict, ensure_ascii=False)}\n')

def __main__():
    translator = OpenAITranslation(original_location="../../../data/eval.jsonl", save_location=f"../../../data/translated_eval_origin_{datetime.datetime.now()}jsonl", doc_type="eval")
    translator.save_file()

if __name__ == "__main__":
    __main__()