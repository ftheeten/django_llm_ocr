from django.shortcuts import render
import json
import logging
from django.http import JsonResponse
# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
#from rest_framework import permissionssudo
from django.core.cache import cache
from  urllib.parse import quote, parse_qs
#from .parser.handle_img import HandleImg
from .models import OcrJob
from django.db import transaction
from api_app.tasks import analyseData
from .serializers import OcrJobSerializer
import signal

logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

class APISubmitOCR(APIView):  
    def handler(self, signum, frame):
        raise TimeoutException()
        
    def post(self, request, *args, **kwargs):        
        flow = request.data.get("base64_img", None)
        if flow is not None:
            try:
                """
                iobj=HandleImg(flow)
                img=iobj.parse()
                height, width, channels = img.shape
                """
                signal.signal(signal.SIGALRM, self.handler)
                signal.alarm(70)
                job=OcrJob.objects.create()
                uuid=job.uuid
                logger.debug("CUSTOM_LOG database object created %s", uuid)
                logger.debug("CUSTOM_LOG transaction and call celery (begin)")
                transaction.on_commit(lambda: analyseData.delay(uuid, flow))
                logger.debug("CUSTOM_LOG transaction and call celery (done, celery message sent)")
                
                #analyseData.delay(uuid, flow)
                cache.clear()  
                return Response({"job_uuid": uuid, "status": "submitted"}, status=status.HTTP_200_OK)
            except Exception:
                logger.error("CUSTOM_ERROR : OTHER EXCEPTION")
                tb_str = traceback.format_exc()
                logger.error(tb_str)
            finally:
                signal.alarm(0)
        cache.clear()  
        return Response({"message": "error no data received"}, status=status.HTTP_200_OK)
        

class APIViewOCR(APIView):
    def get(self, request, *args, **kwargs):        
        data={}
        params_tmp=request.GET.urlencode()
        params=[]
        resp={}
        uuid=None
        #return JsonResponse(params_tmp, safe=False)
        if len(params_tmp)>0:
            params=parse_qs(params_tmp)
            if "uuid" in params:
                uuid=params["uuid"] or []
                if isinstance(uuid, list):
                    if len(uuid)>0:
                        uuid=uuid[0]
                if uuid !="":
                    job=OcrJob.search_by_uuid(uuid)
                    if job is not None:
                        tmp=OcrJobSerializer(job)
                        resp=tmp.data
        cache.clear()  
        return  Response(resp, status=status.HTTP_200_OK)               