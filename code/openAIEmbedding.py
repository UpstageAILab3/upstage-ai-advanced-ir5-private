from openai import OpenAI
import os
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
import json
import traceback
import datetime

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API")
class OpenAISearch:
    def __init__(self, search_doc, index_name="202410221914"):
        self.client = OpenAI()
        self.search_doc = search_doc
        es_username = "elastic"
        es_password = os.environ.get('ELASTIC_PASSWORD')
        self.index_name = index_name
        self.es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="~/elasticsearch-8.15.2/config/certs/http_ca.crt")
        self.check_connection()
        self.settings = {
            "analysis": {
                "filter": {
                    "english_stop": {
                        "type": "stop",
                        "stopwords": "_english"
                    },
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "light_english" 
                    }
                },
                "analyzer": {
                    "english_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard", 
                        "filter": [
                            "lowercase", 
                            "english_stop", 
                            "english_stemmer" 
                        ]
                    }
                }
            }
        }

        self.mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "english_analyzer"  
                },
                "embeddings": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "l2_norm" 
                }
            }
        }

        
        if not self.es.indices.exists(index=self.index_name):
                self.create_index()
                self.index_data()
        else:
            pass
            
       

    def check_connection(self):
        try:
            if self.es.ping():
                print("connected elasticsearch")
            else:
                print("connected failed")
        except Exception as e:
            print("elasticsearch connection failed", e)

    def create_index(self):
        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name, settings=self.settings, mappings=self.mappings)
                print(f'Index {self.index_name} created.')
            print(f'Index {self.index_name} exists.')
        except Exception as e:
            print(f"Error creating index:{e}")

    def get_embedding(self, text):
        print(text)
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            ).data[0].embedding
            print(response)
            return response

        except Exception as e:
            print(f"make embedding error: {e}")

    def index_data(self):

        with open(self.search_doc) as f:
            documents = [json.loads(line) for line in f]

        actions = [
        {
            "_index": self.index_name,
            "_id": doc['docid'],
            "_source": {
                "content": doc.get('summary_in_english', ''),
                "embedding": self.get_embedding(doc.get('summary_in_english', ''))
            }
        }
        for doc in documents
        ]


        helpers.bulk(self.es, actions)
        print(f'{len(documents)} documents indexed.')

    def sparse_retrieve(self, query_str, size=5):
        query = {
            "match": {
                "content": {
                    "query": query_str
                }
            }
        }

        try:

            response = self.es.search(index=self.index_name, query=query, size=size, sort="_score")
        
            results = []

            for hit in response['hits']['hits']:
                results.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'content': hit['_source']['content']
                })

            return results

        except Exception as e:
            print(f"Search failed: {e}")
            return []

    
class EvalAnswer:
    def __init__(self, eval_filename, output_filename ):
        self.eval_filename = eval_filename
        self.output_filename = output_filename
        # self.client = OpenAI()
        self.openAISearch = OpenAISearch(search_doc="../../../new_data/translated_document.jsonl", )
        # self.llm_model = "gpt-3.5-turbo-1106"
        # self.llm_model = "gpt-4o"
        self.persona_qa = """
        ## Role: 과학 상식 전문가

        ## Instructions
        - 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
        - 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
        - 한국어로 답변을 생성한다.
        """
        self.persona_function_calling = """
        ## Role: 과학 상식 전문가

        ## Instruction
        - 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
        - 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.
        """
        self.tools = [
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
                            },
                            # "is_science_question":{
                            #     "type": "boolean",
                            #     "description": "if query is a science question, return True, else False"
                            # }
                        },
                        "required": ["standalone_query", "is_science_question"],
                        "type": "object"
                    }
                }
            },
        ]
    def answer_question(self, messages):
        response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

        # msg = [{"role": "system", "content": self.persona_function_calling}] + messages
        # try:
        #     result = self.client.chat.completions.create(
        #         model=self.llm_model,
        #         messages=msg,
        #         tools=self.tools,
        #         tool_choice={"type": "function", "function": {"name": "search"}},
        #         temperature=0,
        #         seed=1,
        #         timeout=10
        #     )
        # except Exception as e:
        #     traceback.print_exc()
        #     return response
        
        # is_science = bool(json.loads(result.choices[0].message.tool_calls[0].function.arguments)["is_science_question"])
        # if is_science:
        # if result.choices[0].message.tool_calls:
        #     print("called:", result.choices[0].message.tool_calls)
        #     tool_call = result.choices[0].message.tool_calls[0]
        #     function_args = json.loads(tool_call.function.arguments)
        #     standalone_query = function_args.get("standalone_query")
    
        #     search_result = self.openAISearch.sparse_retrieve(standalone_query, 5)

        #     response["standalone_query"] = standalone_query
        #     retrieved_context = []
        #     for i, rst in enumerate(search_result):
        #         retrieved_context.append(rst["content"])
        #         response["topk"].append(rst["id"])
        #         response["references"].append({"score": rst["score"], "content": rst["content"]})
        #     print(response)

        search_result = self.openAISearch.sparse_retrieve(messages, 3)
        response["standalone_query"] = messages

        retrieved_context = []
        for i, rst in enumerate(search_result):
            retrieved_context.append(rst["content"])
            response["topk"].append(rst["id"])
            response["references"].append({"score": rst["score"], "content": rst["content"]})
        print(response)

            
        return response
    
    def eval_rag(self):
        with open(self.eval_filename, encoding='utf-8') as f, open(self.output_filename, "w", encoding='utf-8') as of:
            try:
                idx = 0
                for line in f:
                    j = json.loads(line)
                    print(f'Test {idx}\nQuestion: {j["msg"][0]["content"]}')
                    print(f'English Question : {j["query_translation_msg"], j["documentation_msg"]}')
                    response = self.answer_question(j["documentation_msg"])
                    print(f'Answer: {response["answer"]}\n')

                    output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
                    of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
                    idx += 1
            except Exception as e:
                print(f"error at {idx}")
                

if __name__ == "__main__":

    eval_answer = EvalAnswer(eval_filename="../../../generated_eval_data/eval_data_hyde_english_2024-10-22 14:06:25.877078.jsonl", output_filename=f"../../../eval_data/index-202410221914_역색인_openai-embedding_english_{datetime.datetime.now()}.csv")
    eval_answer.eval_rag()

    # result = eval_answer.answer_question([{"role": "user", "content": "wow you are cool"}])
    # print(result)