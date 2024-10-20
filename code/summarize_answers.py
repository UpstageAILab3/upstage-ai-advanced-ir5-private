import pandas as pd
import numpy as np
import json
import datetime
import torch
from transformers import PreTrainedTokenizerFast, AutoModel, AutoTokenizer, BartForConditionalGeneration
from torch.nn.functional import softmax
import re

class SummerizeAnswers:
    def __init__(self, filename, summarize_way = "generation") -> None:
        self.filename = filename
        self.savefilename = f'../../../new_data/new_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
        self.generative_tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        self.generative_model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
        self.extractive_model = AutoModel.from_pretrained("klue/bert-base")
        self.extractive_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    def summarize_line(self, answer):
        inputs = self.tokenizer.encode(answer, return_tensors="pt")
        input_ids = torch.cat([torch.tensor([[self.tokenizer.bos_token_id]]), inputs, torch.tensor([[self.tokenizer.eos_token_id]])], dim=1)
        
        with torch.no_grad():
            summary_ids = self.model.generate(input_ids, num_beams=4, max_length=300, eos_token_id=self.tokenizer.eos_token_id)
        summarized_line = self.tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
        return summarized_line
    
    def split_into_sentences(self, text):
    # 마침표, 물음표, 느낌표 등으로 문장을 구분
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        return sentences

    def extractive_summarize_line(self, answer):
        # 문장 단위로 나눔
        sentences = self.split_into_sentences(answer)
        
        # 입력 데이터를 토크나이징
        input = self.extractive_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=256)

        with torch.no_grad():
            outputs = self.extractive_model(**input)

        # 임베딩 벡터
        embeddings = outputs.last_hidden_state 
        sentence_scores = embeddings.mean(dim=1)  
        sentence_scores = sentence_scores.mean(dim=1)

        importance_threshold = sentence_scores.mean().item() 

        important_sentences = []
        for i, score in enumerate(sentence_scores):
            if score.item() > importance_threshold:  
                important_sentences.append(sentences[i]) 
        
        summary = " ".join(important_sentences) 
        return summary



    def save_line_to_file(self,summerize_function):

        with open(self.filename, 'r', encoding='utf-8') as f, open(self.savefilename, 'w', encoding='utf-8') as new_file:
            for original_line in f:
                original_content = json.loads(original_line)
                answer_original_content = original_content["content"]
                summary = summerize_function(answer_original_content)

                combined_dict = {
                "docid": original_content['docid'],
                "src": original_content['src'],
                "content": original_content['content'],
                "summary": summary
                }

                new_file.write(f'{json.dumps(combined_dict, ensure_ascii=False)}\n')

def __main__():
    summarize = SummerizeAnswers("../../../data/documents.jsonl")
    # summarize.save_line_to_file(summarize.summarize_line)
    summarize.save_line_to_file(summarize.extractive_summarize_line)

if __name__ == "__main__":
    __main__()