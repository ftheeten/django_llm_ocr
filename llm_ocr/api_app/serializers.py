from rest_framework import serializers
from .models import OcrJob


class OcrJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = OcrJob
        fields= ['uuid', 'user','status','result','submission_date','termination_date']
