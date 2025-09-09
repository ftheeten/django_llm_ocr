import os 
#for tasks.py
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


from celery import Celery
from django.conf import settings
from logging.config import dictConfig
from celery.signals import setup_logging
#from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig

"""
PROCESSOR = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="cuda"
)
MODEL = MODEL.to("cuda") # or cuda
GENERATION_CONFIG = GenerationConfig().from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

CONVERSATION = [
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
"""


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "llm_ocr.settings")
app = Celery("llm_ocr")
app.config_from_object("django.conf:settings", namespace='CELERY')
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)

@setup_logging.connect
def config_loggers(*args, **kwargs):
    dictConfig(settings.LOGGING)