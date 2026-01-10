import uuid
from django.db import models
from core.models.turbine import Turbine


class TurbineLog(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    turbine = models.ForeignKey(Turbine, on_delete=models.CASCADE, related_name='logs')
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    is_dummy_data = models.BooleanField(default=False)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['turbine', '-timestamp']),
        ]

    def __str__(self):
        return f"{self.turbine.name} - {self.timestamp}"

