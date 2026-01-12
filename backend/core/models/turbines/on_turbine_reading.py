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
    # Wind measurements
    wind_speed = models.FloatField(null=True, blank=True)
    wind_direction = models.FloatField(null=True, blank=True)
    # Electrical parameters
    voltage_l1 = models.FloatField(null=True, blank=True)
    voltage_l2 = models.FloatField(null=True, blank=True)
    voltage_l3 = models.FloatField(null=True, blank=True)
    current_l1 = models.FloatField(null=True, blank=True)
    current_l2 = models.FloatField(null=True, blank=True)
    current_l3 = models.FloatField(null=True, blank=True)
    reactive_power = models.FloatField(null=True, blank=True)
    power_factor = models.FloatField(null=True, blank=True)
    # Additional temperatures
    front_bearing_temperature = models.FloatField(null=True, blank=True)
    rear_bearing_temperature = models.FloatField(null=True, blank=True)
    nacelle_temperature = models.FloatField(null=True, blank=True)
    transformer_temperature = models.FloatField(null=True, blank=True)
    gear_oil_temperature = models.FloatField(null=True, blank=True)
    # Mechanical parameters
    rotor_speed_rpm = models.FloatField(null=True, blank=True)
    generator_rpm = models.FloatField(null=True, blank=True)
    gearbox_speed = models.FloatField(null=True, blank=True)
    torque = models.FloatField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['log']),
        ]


