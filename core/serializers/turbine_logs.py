from rest_framework import serializers
from core.models.turbines import (
    TurbineLog, OnTurbineReading, WeatherReading,
    MaintenanceEvent, HealthPrediction, Alert
)
from core.serializers import BaseSerializer


class OnTurbineReadingSerializer(BaseSerializer):
    class Meta:
        model = OnTurbineReading
        fields = '__all__'
        read_only_fields = ['id', 'log']


class WeatherReadingSerializer(BaseSerializer):
    class Meta:
        model = WeatherReading
        fields = '__all__'
        read_only_fields = ['id', 'log']


class TurbineLogSerializer(BaseSerializer):
    on_turbine_readings = OnTurbineReadingSerializer(many=True, required=False)
    weather_readings = WeatherReadingSerializer(many=True, required=False)

    class Meta:
        model = TurbineLog
        fields = '__all__'
        read_only_fields = ['id', 'timestamp', 'turbine']

    def create(self, validated_data):
        on_turbine_readings_data = validated_data.pop('on_turbine_readings', [])
        weather_readings_data = validated_data.pop('weather_readings', [])
        
        log = TurbineLog.objects.create(**validated_data)
        
        for reading_data in on_turbine_readings_data:
            OnTurbineReading.objects.create(log=log, **reading_data)
        
        for reading_data in weather_readings_data:
            WeatherReading.objects.create(log=log, **reading_data)
        
        return log


class MaintenanceEventSerializer(BaseSerializer):
    class Meta:
        model = MaintenanceEvent
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'turbine']


class HealthPredictionSerializer(BaseSerializer):
    class Meta:
        model = HealthPrediction
        fields = '__all__'
        read_only_fields = ['id', 'timestamp', 'turbine']


class AlertSerializer(BaseSerializer):
    class Meta:
        model = Alert
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'turbine']

