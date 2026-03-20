from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import traceback
import os
import time
import cv2
from collections import OrderedDict

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class TimeoutStoppingCriteria(StoppingCriteria):
    def __init__(self, deadline):
        self.deadline = deadline

    def __call__(self, input_ids, scores, **kwargs):
        import time
        return time.time() > self.deadline

class OcrQwen3b8():
    model_path="/opt/panafgeo_ict/src/local_models/qwen_3_vl_8b"
    device_type="cuda" # cpu/cuda
    
    def __init__(self,  p_prompt_1="Transcribe the text of the image like you were an OCR engine. The document is handwritten in French and concerns geology of Central Africa. Keep the formatting in LaTeX.", p_max_tokens=4092):        
        self.prompt_1=p_prompt_1        
        self.max_tokens=p_max_tokens
        
    def run_model(self, image_paths,  p_timeout): 
        returned=OrderedDict()
        print("quant")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print(OcrQwen3b8.model_path)

        
        print("model")        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            OcrQwen3b8.model_path, torch_dtype=torch.bfloat16, device_map="auto",
            quantization_config=quant_config,
            attn_implementation="flash_attention_2",
        )
        print("processor")
        processor = AutoProcessor.from_pretrained(OcrQwen3b8.model_path)
        
        print("loop images")
        for image_path in image_paths:
            print(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },                   
                        {"type": "text", "text": self.prompt_1}
                        
                        
                    ],
                }
            ]
            
            
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            deadline = time.time() + p_timeout 
            criteria = StoppingCriteriaList([TimeoutStoppingCriteria(deadline)])
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_tokens,stopping_criteria=criteria)
            generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print("result")
            print(output_text)
            returned[image_path]=output_text
            #queue.put(output_text)
        #except Exception:
        #    print(traceback.format_exc())
        return returned    
    
    def process(self, p_cv_img_list, p_time_out=300):             
        try:           
            returned=self.run_model(p_cv_img_list, p_time_out)
            return returned
        except Exception:
            print("Exeption in Qwen7")
            print(traceback.print_exc())
            print(traceback.print_stack())
            print(traceback.format_exc())            
            
        
