import pandas as pd
import numpy as np
import json
import datetime
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration




class SummerizeAnswers:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.savefilename = f'../../../new_data/new_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        self.model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
    
    def summerize_line(self, query):
        inputs = self.tokenizer.encode(query, return_tensors="pt")
        input_ids = torch.cat([torch.tensor([[self.tokenizer.bos_token_id]]), inputs, torch.tensor([[self.tokenizer.eos_token_id]])], dim=1)
        
        with torch.no_grad():
            summary_ids = self.model.generate(input_ids, num_beams=4, max_length=300, eos_token_id=self.tokenizer.eos_token_id)
        summarized_line = self.tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
        return summarized_line

    def save_line_to_file(self):

        with open(self.filename, 'r', encoding='utf-8') as f, open(self.savefilename, 'w', encoding='utf-8') as new_file:
            for original_line in f:
                original_content = json.loads(original_line)
                query_original_content = original_content["content"]
                summary = self.summerize_line(query_original_content)

                combined_dict = {
                "docid": original_content['docid'],
                "src": original_content['src'],
                "content": original_content['content'],
                "summary": summary
                }

                new_file.write(f'{json.dumps(combined_dict, ensure_ascii=False)}\n')

def __main__():
    summerize = SummerizeAnswers("../../../data/documents.jsonl")
    summerize.save_line_to_file()

if __name__ == "__main__":
    __main__()