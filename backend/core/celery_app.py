"""
Celery app configuration - alternative import path.
"""
from .celery import app as celery_app

__all__ = ('celery_app',)
