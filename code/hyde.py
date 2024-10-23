from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import traceback
import datetime

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API")

class HYde:
    def __init__(self, original_eval, save_location_documatized_eval) -> None:

        self.depart_location = original_eval
        self.destination = save_location_documatized_eval

        self.client = OpenAI()
    
        self.model_name = "gpt-4o"
        self.instructions = "마지막에 주어진 영어질문 문장을 영어로 문서화할때 100 단어정도로만 문서화해줘"
        self.client = OpenAI()
        self.tools = [
            {
                "type": "function",
                "function":{
                    "name": "query_to_documents",
                    "description": "documentation",
                    "parameters": {
                        "properties":{
                            "documentation": {
                                "type": "string",
                                "description": "query to documents"
                            }
                        },
                        "required": ["documentation"],
                        "type": "object",
                        "additionalProperites": False
                    }
                }
            }
        ]

    def query_to_doc(self, query):

        response = {"document": ""}
        msg = [{
            "role": "system",
            "content": self.instructions
        },{
            "role": "assistant",
            "content": query
        }]

        print(msg)

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=msg,
                tools=self.tools,
                temperature=0.7,
                seed=1,
                timeout=10
            )
            response["document"] = json.loads(result.choices[0].message.tool_calls[0].function.arguments)["documentation"]
        except Exception as e:
            traceback.print_exc()
            return response
        
        return response
    

    def save_file(self):

        with open(self.depart_location, encoding='utf-8') as doc, open(self.destination, "w", encoding='utf-8') as sf:
            idx = 0
            for line in doc:
                line_j = json.loads(line)
                print(f'Document {line_j["msg"], line_j["query_translation_msg"]})')
                response = self.query_to_doc(line_j["query_translation_msg"][0]["content"])
                print()
                print(f'Documentation: {response["document"]}')
                
                combined_dict = {
                    "eval_id": line_j['eval_id'],
                    "msg": line_j['msg'],
                    "query_translation_msg": line_j['query_translation_msg'],
                    "documentation_msg": response["document"] 
                }
                print(idx)
                sf.write(f'{json.dumps(combined_dict, ensure_ascii=False)}\n')
            doc.close()
            sf.close()


        
    # def make_hyde(self):
    #     with open(self.depart_location, encoding="utf-8") as f , open(self.destination, "w", encoding="utf-8") as sf:

if __name__ == "__main__":

    hyde = HYde("../../../data/translated_eval_origin.jsonl", f"../../../generated_eval_data/eval_data_hyde_english_{datetime.datetime.now()}.jsonl")
    hyde.save_file()