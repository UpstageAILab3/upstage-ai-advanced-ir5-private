import pandas as pd
import numpy as np
import json
import datetime
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration




class SummerizeAnswer:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.savefilename = f'../../../new_data/new_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        self.model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
    
    def get_original_lines(self):
        with open(self.filename) as f:
            for line in f:
                original_line = json.loads(line)
                query = original_line["content"]
                yield query

    def summerize_line(self):
        for original_line in self.get_original_lines():
            inputs = self.tokenizer.encode(original_line, return_tensors="pt")
            input_ids = torch.cat([torch.tensor([[self.tokenizer.bos_token_id]]), inputs, torch.tensor([[self.tokenizer.eos_token_id]])], dim=1)
            
            with torch.no_grad():
                summary_ids = self.model.generate(input_ids, num_beams=4, max_length=300, eos_token_id=self.tokenizer.eos_token_id)
            summarized_line = self.tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
            yield summarized_line

    def save_line_to_file(self):
        new_content = []

        with open(self.filename, 'r', encoding='utf-8') as f:
            for original_line in f:
                original_content = json.loads(original_line)
                summary = next(self.summerize_line())

                combined_dict = {
                "docid": original_content['docid'],
                "src": original_content['src'],
                "content": original_content['content'],
                "summary": summary
                }
                new_content.append(combined_dict)

        with open(self.savefilename, 'w', encoding='utf-8') as new_file:
            for content in new_content:
                json.dump(content, new_file, ensure_ascii=False)
                new_file.write('\n')

def __main__():
    summerize = SummerizeAnswer("../../../data/documents.jsonl")
    summerize.save_line_to_file()

if __name__ == "__main__":
    __main__()