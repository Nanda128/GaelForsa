"""
Import celery app from core to ensure it's loaded when Django starts.
"""
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'api.settings')

from core.celery_app import celery_app

__all__ = ('celery_app',)
