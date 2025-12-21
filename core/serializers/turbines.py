from rest_framework import serializers
from core.models.turbine import Turbine
from core.serializers import BaseSerializer


class TurbineSerializer(BaseSerializer):
    class Meta:
        model = Turbine
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']

