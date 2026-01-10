import uuid
from django.db import models
from .turbine_log import TurbineLog


class OnTurbineReading(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    log = models.ForeignKey(TurbineLog, on_delete=models.CASCADE, related_name='on_turbine_readings')
    rotor_speed = models.FloatField(null=True, blank=True)
    power_output = models.FloatField(null=True, blank=True)
    generator_temperature = models.FloatField(null=True, blank=True)
    gearbox_oil_temperature = models.FloatField(null=True, blank=True)
    gearbox_oil_pressure = models.FloatField(null=True, blank=True)
    vibration_level = models.FloatField(null=True, blank=True)
    blade_pitch_angle = models.FloatField(null=True, blank=True)
    yaw_position = models.FloatField(null=True, blank=True)
    yaw_error = models.FloatField(null=True, blank=True)
    is_dummy_data = models.BooleanField(default=False)

    class Meta:
        indexes = [
            models.Index(fields=['log']),
        ]

