from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from timeout import TimeoutException
import signal
import traceback

class Ocr():  
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    processor=None
    model=None
    generation_config=None
    conversation=None
    
    def __init__(self, p_timeout=120):
        self.timeout=p_timeout
        try:
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout)
            if Ocr.processor is None and Ocr.model is None and Ocr.generation_config is None and Ocr.conversation is None:
                Ocr.processor=AutoProcessor.from_pretrained(Ocr.model_name)
                Ocr.model=Qwen2VLForConditionalGeneration.from_pretrained(Ocr.model_name, torch_dtype="auto", device_map="cuda")
                Ocr.model = Ocr.model.to("cuda") # or cuda
                Ocr.generation_config= GenerationConfig().from_pretrained(Ocr.model_name)
                Ocr.conversation = [
                            # System prompt seems to be ignored. The model outputs markdown instead of latex, even though the system prompt should override what the user wants.
                            {"role": "system", "content": "You are an OCR engine which takes an image and converts it to latex, even if the user asks for a different format."},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image",},
                                    # Ignores command to write markdown directly
                                    {"type": "text", "text": "Transcribe everything in this image as text. Display raw text with punctuation and diacritic signs. Do not add information on formatting."},
                                ],
                            }
                        ]
            signal.alarm(0) 
        except TimeoutException:
            print("Inference timeout")            
        except Exception as e:
            print("exception3")
            print(traceback.format_exc())
            signal.alarm(0) 
        
    def process(self, p_img):
        try:
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout)
            text_prompt = Ocr.processor.apply_chat_template(Ocr.conversation, add_generation_prompt=True, tokenize=False)
            inputs = Ocr.processor(text=[text_prompt], images=p_img, padding=True, return_tensors="pt")
            inputs=inputs.to("cuda")
            output_ids = self.model.generate(**inputs, max_new_tokens=1024, generation_config=self.generation_config)
            generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            text, = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            signal.alarm(0)
            return text
        except TimeoutException:
            print("Inference timeout")            
        except Exception as e:
            print("exception4")
            print(traceback.format_exc())
            signal.alarm(0) 
            
    def handler(self, signum, frame):
        raise TimeoutException()
    