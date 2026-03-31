from transformers import pipeline, AutoTokenizer,AutoModelForTokenClassification
import itertools
import sys
import traceback
import re
#from happytransformer import HappyTextToText, TTSettings


class Classifier():

    def __init__(self, p_model_name, p_aggregation_strategy=None):
        print(p_model_name)
        self.model_name=p_model_name
        print(p_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.aggregation_strategy=p_aggregation_strategy
        self.pat=re.compile(r".*(-|\u2010\\u2011|\u2012|\u2013|\u2014)$")
        #self.happy_tt = HappyTextToText("T5", "fdemelo/t5-base-spell-correction-fr")
        #self.args = TTSettings(num_beams=5, min_length=1)
        
    def merge_line_break(self, p_texts):
        returned=[]
        previous=None
        for t in p_texts:
            print("===")
            print(t) 
            t=t.strip()
            if self.pat.match(t) is not None:
                print("break_line")
                t=t[:-1]
            else:
                t=t+" "
            returned.append(t)
        return returned
        
    def process(self, p_elems, p_batch_size=1024):
        returned={}
        returned[0]=[]
        #flatten list of list
        #texts=list(itertools.chain(*list(p_elems.values())))
        list_t=[]
        for key, t in p_elems.items():
            list_t.append(t)
        #print(list_t)
        list_t=self.merge_line_break(list_t)
        texts=''.join(list_t)
        
     
        print("--------------------")
        print(texts)
        """
        print(len(texts))
        result = self.happy_tt.generate_text("grammaire: "+texts, args=self.args)
        print(result.text) 
        tmp_keys=list(p_elems.keys())
        """
        obj_pipeline = pipeline(
                task="ner",
                model=self.model_name,
                tokenizer=self.tokenizer,
                aggregation_strategy= self.aggregation_strategy
            )
        results = obj_pipeline(texts, batch_size=p_batch_size)
        print(results)       
   
        for entity in results:
            word_str = self.tokenizer.convert_tokens_to_string([entity['word']])
            start=entity["start"]
            end=entity["end"]
            parsed_word=""
            try:
                parsed_word=texts[start:end]
            except Exception:
                parsed_word=traceback.format_exc()
            returned[0].append({"word": entity["word"],  "entity_group": entity["entity_group"],  "score":float(entity["score"]), "word_str":word_str, "start": start, "end":end, "parsed_word":parsed_word})
        """
        if len(p_elems.keys())==len(texts):
            obj_pipeline = pipeline(
                task="ner",
                model=self.model_name,
                tokenizer=self.tokenizer,
                aggregation_strategy= self.aggregation_strategy
            )
            results = obj_pipeline(texts, batch_size=p_batch_size)
            for i, entities in enumerate(results):
                print(i)
                tmp_key=tmp_keys[i]
                if not tmp_key in returned_tmp:
                    returned[tmp_key]=[]
                for entity in entities:
                    word_str = self.tokenizer.convert_tokens_to_string([entity['word']])
                    start=entity["start"]
                    end=entity["end"]
                    parsed_word=""
                    try:
                        parsed_word=texts[i][start:end]
                    except Exception:
                        parsed_word=traceback.format_exc()
                    returned[tmp_key].append({"word": entity["word"],  "entity_group": entity["entity_group"],  "score":float(entity["score"]), "word_str":word_str, "start": start, "end":end, "parsed_word":parsed_word})    
        """
        return returned
        
            