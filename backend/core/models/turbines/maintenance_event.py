import uuid
from django.db import models
from core.models.turbine import Turbine
from .turbine_log import TurbineLog


class MaintenanceEvent(models.Model):
    EVENT_TYPE_CHOICES = [
        ('scheduled', 'Scheduled'),
        ('repair', 'Repair'),
        ('failure', 'Failure'),
        ('inspection', 'Inspection'),
        ('replacement', 'Replacement'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    turbine = models.ForeignKey(Turbine, on_delete=models.CASCADE, related_name='maintenance_events')
    log = models.ForeignKey(TurbineLog, on_delete=models.SET_NULL, null=True, blank=True, related_name='maintenance_events')
    event_type = models.CharField(max_length=20, choices=EVENT_TYPE_CHOICES)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    description = models.TextField(blank=True)
    parts_replaced = models.TextField(blank=True)
    cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    is_dummy_data = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-start_time']
        indexes = [
            models.Index(fields=['turbine', '-start_time']),
            models.Index(fields=['event_type']),
            models.Index(fields=['log']),
        ]

    def __str__(self):
        return f"{self.turbine.name} - {self.event_type} - {self.start_time}"

