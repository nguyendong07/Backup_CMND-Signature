from rest_framework import serializers
from .models import Extract

class ExtractImage(serializers.ModelSerializer):
    class Meta:
        model = Extract
        fields = ['title', 'content']