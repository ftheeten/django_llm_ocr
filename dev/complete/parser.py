from transformers import pipeline, CamembertTokenizerFast, AutoModelForTokenClassification
import os
os.environ["TRANSFORMERS_NO_TIKTOKEN"] = "1"
#pip install protobuff sentencepiece tiktoken
class Parser():  
    model=None
    model_name = "Jean-Baptiste/camembert-ner"
    tokenizer=None
    pipeline=None
    
    def __init__(self):
        if Parser.model is None and Parser.tokenizer is None and Parser.pipeline is None:

            Parser.tokenizer = CamembertTokenizerFast.from_pretrained(self.model_name)
            Parser.model=AutoModelForTokenClassification.from_pretrained(self.model_name)
            Parser.pipeline = pipeline("ner", model=Parser.model, tokenizer=Parser.tokenizer, aggregation_strategy="simple")
        
        
    def process(self, p_text):
        entities = Parser.pipeline(p_text)
        return entities