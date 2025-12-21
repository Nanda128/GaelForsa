import uuid
from django.db import models
from .turbine_log import TurbineLog


class WeatherReading(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    log = models.ForeignKey(TurbineLog, on_delete=models.CASCADE, related_name='weather_readings')
    wind_speed = models.FloatField(null=True, blank=True)
    wind_direction = models.FloatField(null=True, blank=True)
    wave_height = models.FloatField(null=True, blank=True)
    wave_period = models.FloatField(null=True, blank=True)
    water_temperature = models.FloatField(null=True, blank=True)
    air_temperature = models.FloatField(null=True, blank=True)
    air_humidity = models.FloatField(null=True, blank=True)
    lightning_strikes = models.IntegerField(default=0)

    class Meta:
        indexes = [
            models.Index(fields=['log']),
        ]

