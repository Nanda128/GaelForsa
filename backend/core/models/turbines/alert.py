import uuid
from django.db import models
from core.models.turbine import Turbine
from .health_prediction import HealthPrediction


class Alert(models.Model):
    SEVERITY_CHOICES = [
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('critical', 'Critical'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    turbine = models.ForeignKey(Turbine, on_delete=models.CASCADE, related_name='alerts')
    prediction = models.ForeignKey(HealthPrediction, on_delete=models.SET_NULL, null=True, blank=True, related_name='alerts')
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    message = models.TextField()
    acknowledged = models.BooleanField(default=False)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['turbine', '-created_at']),
            models.Index(fields=['severity', 'acknowledged']),
        ]

    def __str__(self):
        return f"{self.turbine.name} - {self.severity} - {self.created_at}"

