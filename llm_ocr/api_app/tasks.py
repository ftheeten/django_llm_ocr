import os
import sys
import traceback
import logging
#from celery import shared_task
from llm_ocr.celery import app
import cv2
from django.conf import settings
from .models import OcrJob
from datetime import datetime
#from llm_ocr.celery import app, PROCESSOR, MODEL, GENERATION_CONFIG, CONVERSATION
from .parser.handle_img import HandleImg
from .transformers.qwen_ocr import QwenOcr
import signal
from celery.exceptions import SoftTimeLimitExceeded

logger = logging.getLogger(__name__)

@app.task(time_limit=60, soft_time_limit=55)
def analyseData(p_uuid, p_flow):
    try:
        logger.debug("ASYNC JOB_TRIGGERED %s", p_uuid)
        job=OcrJob.search_by_uuid(p_uuid)
        #job.update(result='some value', status="pending")
        job.status='pending'
        job.save()
        
        iobj=HandleImg(p_flow)
        img=iobj.parse()
        if img is not None:
            height, width, channels = img.shape
            logger.debug("Image created height %d width %d  channels %d", height, width, channels )
            #job=QwenOcr(p_uuid, img, logger, PROCESSOR, MODEL, GENERATION_CONFIG, CONVERSATION, settings.TIMEOUT_INFERENCE_JOB)
            job=QwenOcr(p_uuid, img, logger, settings.TIMEOUT_INFERENCE_JOB)
            text=job.process()
            if text is not None:
                logger.debug("GOT TEXT FOR %s", p_uuid)
                logger.debug("%s", text)
                result={}
                result["text"]=text
                logger.debug("TRY TO SERIALIZE IN DB")
                job=OcrJob.search_by_uuid(p_uuid)
                job.result=result
                job.status='done'
                job.termination_date = datetime.now()
                job.save()
                logger.debug("DONE")
            else:
                logger.error("CUSTOM_ERROR test is null for %s", p_uuid)                
    except SoftTimeLimitExceeded:
        self.logger.error("CUSTOM_ERROR KILL BY Celery SoftTimeLimitExceeded ",p_uuid)
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        self.logger.error("CUSTOM_ERROR : OTHER EXCEPTION %s",p_uuid)
        tb_str = traceback.format_exc()
        self.logger.error(tb_str)
            
            
