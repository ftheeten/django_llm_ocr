import uuid as puuid
from django.db import models
from django.db import connection

class OcrJob(models.Model):
    fpk = models.AutoField(primary_key=True)
    uuid = models.UUIDField(default=puuid.uuid4, editable=False)    
    user=models.CharField(default=None, blank=True, null=True)
    status=models.CharField(default="submitted")
    result= models.JSONField(default=None, blank=True, null=True)
    submission_date=models.DateTimeField(auto_now_add=True)
    termination_date=models.DateTimeField(default=None, blank=True, null=True)

    @staticmethod
    def search_by_uuid(p_uuid):
        q=OcrJob.objects.filter(uuid=p_uuid)
        if q is not None:
            if len(q)>0:
                return q.first()
        return None
    
