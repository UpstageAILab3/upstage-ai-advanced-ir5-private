from elasticsearch import Elasticsearch
import csv
from dotenv import load_dotenv
import os
import datetime
from pytz import timezone
import json
from progress_rate import progress_decorator, progress_bar


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API")

class ElasticSearchEmbedding:
    def __init__(self) -> None:
        self.index_name="202410221914"
        es_username = "elastic"
        es_password = os.environ.get('ELASTIC_PASSWORD')
        self.es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="~/elasticsearch-8.15.2/config/certs/http_ca.crt")
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

    @progress_decorator
    def to_csv(self):
        query = {
            "match_all": {}
            
        }

        response = self.es.search(index=self.index_name, query=query, size=5000)
        timestamp = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')

        total_hits = len(response['hits']['hits'])
        with open(f"../../../es_index/{timestamp}.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["_id", "content", "embedding", "embeddings"])  # 필요한 헤더 작성
            for i, hit in enumerate(response['hits']['hits']):
                
                _id = hit["_id"]
                content = hit["_source"].get("content", "N/A")
                embedding = hit["_source"].get("embedding", "N/A") 
                embeddings = hit["_source"].get("embeddings", "N/A") 

                writer.writerow([_id, content, embedding, embeddings])

                progress_bar(i + 1, total_hits)


    def get_elasticsearch_search(self,query,size, sort):
        return self.es.search(index=self.index_name, query=query, size=size, sort=sort)

if __name__ == "__main__":
    
    es = ElasticSearchEmbedding()
    es.to_csv()