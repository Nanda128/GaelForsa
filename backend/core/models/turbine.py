import uuid
from django.db import models


class Turbine(models.Model):
    STATUS_CHOICES = [
        ('green', 'Healthy'),
        ('yellow', 'Warning'),
        ('red', 'Critical'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    latitude = models.FloatField()
    longitude = models.FloatField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='green')
    is_dummy_data = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name

