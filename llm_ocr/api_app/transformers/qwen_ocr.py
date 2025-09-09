from django.conf import settings
from ..models import OcrJob
import signal
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from collections import OrderedDict
import cv2
import os
import sys
import traceback

class TimeoutException(Exception):
    pass
    
class QwenOcr():
    uuid=None
    img=None
    logger=None
    processor=None
    model=None
    generation_config=None
    conversation=None
    timeout=None
    
    #def __init__(self, p_uuid,  p_img, p_logger,  p_processor, p_model, p_generation_config, p_conversation,p_timeout=60 ):
    def __init__(self, p_uuid,  p_img, p_logger,p_timeout=60 ):
        try:
            self.uuid=p_uuid
            self.img=p_img
            self.logger=p_logger
            """
            self.processor=p_processor
            self.generation_config=p_generation_config
            self.conversation=p_conversation
            """
            self.processor=AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            self.model=Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="cuda")
            self.model = self.model.to("cuda") # or cuda
            self.generation_config= GenerationConfig().from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            self.conversation = [
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
            self.timeout=p_timeout
            self.logger.debug("OCR launched %s", self.uuid)
        except Exception:
            self.logger.error("CUSTOM_ERROR : OTHER EXCEPTION %s", self.uuid)
            tb_str = traceback.format_exc()
            self.logger.error(tb_str)
    
    def handler(self, signum, frame):
        raise TimeoutException()
    
    def process(self):
        try:
            self.logger.debug("process launched %s", self.uuid)
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(60)  
            text_prompt = self.processor.apply_chat_template(self.conversation, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=[text_prompt], images=[self.img], padding=True, return_tensors="pt")
            inputs=inputs.to("cuda")
            output_ids = self.model.generate(**inputs, max_new_tokens=1024, generation_config=self.generation_config)
            generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            text, = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            signal.alarm(0)
            self.logger.debug("DEBUG returned %s", self.uuid)
            return text
        except TimeoutException:
            self.logger.error("CUSTOM_ERROR : TIMEOUT OCR %s", self.uuid)
        except Exception:
            self.logger.error("CUSTOM_ERROR : OTHER EXCEPTION %s", self.uuid)
            tb_str = traceback.format_exc()
            self.logger.error(tb_str)