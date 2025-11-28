from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
import torch.multiprocessing as mp
import torch
from multiprocessing import Queue
import traceback
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class OcrQwenAsync():
    model_name="Qwen/Qwen2-VL-7B-Instruct"
    device_type="cuda" # cpu/cuda
    
    def __init__(self,  p_prompt_1="You are an OCR engine which takes an image and converts it to latex, even if the user asks for a different format.", p_prompt_2="Transcribe everything in this image as text. Display raw text with punctuation and diacritic signs. Do not add information on formatting.", p_max_tokens=2048):        
        self.prompt_1=p_prompt_1
        self.prompt_2=p_prompt_2
        self.max_tokens=p_max_tokens
        
    def run_model(self, image, queue):        
        processor = AutoProcessor.from_pretrained(OcrQwenAsync.model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            OcrQwenAsync.model_name, torch_dtype="auto", device_map=OcrQwenAsync.device_type
        )
        model.to(torch.bfloat16)
        model = model.to(OcrQwenAsync.device_type)
        model.eval()        
        generation_config = GenerationConfig().from_pretrained( OcrQwenAsync.model_name)
        conversation = [
          
            {"role": "system", "content":self.prompt_1},
            {
                "role": "user",
                "content": [
                    {"type": "image",},
                    # Ignores command to write markdown directly
                    {"type": "text", "text":self.prompt_2},
                ],
            }
        ]
        
        print(f"image shape={image.shape}")
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        inputs=inputs.to(OcrQwenAsync.device_type)
        output_ids = model.generate(**inputs, max_new_tokens=self.max_tokens, generation_config=generation_config,do_sample=False)
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        text, = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(text)
        queue.put(text)
        #except Exception:
        #    print(traceback.format_exc())
            
    
    def process(self, p_cv_img, p_time_out=60): 
        q = Queue()
        p = mp.Process(target=self.run_model, args=(p_cv_img, q,))
        p.start()
        
        try:
           
            p.join(timeout=p_time_out)
            if p.is_alive():
                print("timeout")
                # p.kill()
                # p.join()
                raise TimeoutError("Timeout raised from OcrQwenAsync")
            else:
                print("normal exit")
                return q.get()
        except Exception:
            print(traceback.format_exc())
            p.kill()
            p.join()
        
        """
        try:
            text = q.get(timeout=p_time_out)
            p.terminate()
            p.join()
            return text

        except Exception as e:
            print("Exception:", e)
            p.kill()
            p.join()
            raise
        """