from transformers import pipeline, CamembertTokenizerFast, AutoModelForTokenClassification
import os
import signal
import traceback

os.environ["TRANSFORMERS_NO_TIKTOKEN"] = "1"

#pip install protobuff sentencepiece tiktoken
class Parser():  
    model=None
    model_name = "Jean-Baptiste/camembert-ner"
    tokenizer=None
    pipeline=None
    
    def __init__(self, p_timeout=120):
        self.timeout=p_timeout
        try:
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout)
            if Parser.model is None and Parser.tokenizer is None and Parser.pipeline is None:

                Parser.tokenizer = CamembertTokenizerFast.from_pretrained(self.model_name)
                Parser.model=AutoModelForTokenClassification.from_pretrained(self.model_name)
                Parser.pipeline = pipeline("ner", model=Parser.model, tokenizer=Parser.tokenizer, aggregation_strategy="simple")
        except TimeoutException:
            print("Inference parser timeout")  
            signal.alarm(0) 
        except Exception as e:
            print("exception parser 3")
            print(traceback.format_exc())
            signal.alarm(0)         
        
    def process(self, p_text):
        try:
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout)
            entities = Parser.pipeline(p_text)
            signal.alarm(0) 
            return entities
        except TimeoutException:
            print("Inference parser timeout")  
            signal.alarm(0) 
        except Exception as e:
            print("exception parser 4")
            print(traceback.format_exc())
            signal.alarm(0) 
            
    def handler(self, signum, frame):
        raise TimeoutException()
    