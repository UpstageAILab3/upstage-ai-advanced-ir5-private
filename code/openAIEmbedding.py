from openai import OpenAI
import os
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
import json

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API")
class OpenAISearch:
    def __init__(self, search_doc, index_name="ver0.2422"):
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

    


if __name__ == "__main__":
    search = OpenAISearch(search_doc="../../../new_data/translated_document.jsonl", )
    s = search.sparse_retrieve("Why is glycogen breakdown necessary in the human body?")

    print(s)