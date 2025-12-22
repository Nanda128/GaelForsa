import uuid
from django.db import models
from core.models.turbine import Turbine
from .turbine_log import TurbineLog


class HealthPrediction(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    turbine = models.ForeignKey(Turbine, on_delete=models.CASCADE, related_name='health_predictions')
    log = models.ForeignKey(TurbineLog, on_delete=models.CASCADE, related_name='predictions')
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    health_score = models.FloatField()
    failure_probability = models.FloatField()
    predicted_failure_window_start = models.DateTimeField(null=True, blank=True)
    predicted_failure_window_end = models.DateTimeField(null=True, blank=True)
    model_version = models.CharField(max_length=50, blank=True)
    features_used = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['turbine', '-timestamp']),
            models.Index(fields=['failure_probability']),
        ]

    def __str__(self):
        return f"{self.turbine.name} - Health: {self.health_score:.2f} - Failure Risk: {self.failure_probability:.2%}"

