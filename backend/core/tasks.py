"""
Celery tasks for real-time data processing and health predictions.
"""
from datetime import datetime, timedelta
from celery import shared_task
from django.utils import timezone
from django.db.models import Q

from core.models import Turbine, TurbineLog, OnTurbineReading
from core.services.health_prediction_service import HealthPredictionService


@shared_task
def process_realtime_reading(log_id: str):
    """
    Process a single TurbineLog and generate health prediction.
    
    Args:
        log_id: UUID of TurbineLog to process
    """
    try:
        log = TurbineLog.objects.get(id=log_id)
        service = HealthPredictionService()
        prediction = service.predict_health(log)
        return {
            'success': True,
            'log_id': str(log_id),
            'prediction_id': str(prediction.id) if prediction else None
        }
    except TurbineLog.DoesNotExist:
        return {'success': False, 'error': f'Log {log_id} not found'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@shared_task
def process_pending_readings(turbine_id: str = None, batch_size: int = 10):
    """
    Process readings that don't have predictions yet.
    
    Args:
        turbine_id: Optional turbine ID to filter by
        batch_size: Number of readings to process in this batch
    """
    try:
        # Find logs without predictions
        logs_without_predictions = TurbineLog.objects.filter(
            predictions__isnull=True
        ).select_related('turbine')
        
        if turbine_id:
            logs_without_predictions = logs_without_predictions.filter(turbine_id=turbine_id)
        
        # Get most recent logs first
        logs_to_process = logs_without_predictions.order_by('-timestamp')[:batch_size]
        
        service = HealthPredictionService()
        predictions_created = 0
        
        for log in logs_to_process:
            prediction = service.predict_health(log)
            if prediction:
                predictions_created += 1
        
        return {
            'success': True,
            'processed': len(logs_to_process),
            'predictions_created': predictions_created
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


@shared_task
def simulate_historical_feed(start_date: str = None, end_date: str = None,
                             turbine_ids: list = None, interval_seconds: int = 600):
    """
    Simulate real-time feed from historical data using Celery.
    This task processes readings in chronological order at specified intervals.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        turbine_ids: List of turbine IDs to process
        interval_seconds: Seconds between processing each reading (default 10 minutes)
    """
    try:

        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        queryset = OnTurbineReading.objects.select_related(
            'log__turbine'
        ).order_by('log__timestamp')
        
        if start_date or end_date:
            date_filter = Q()
            if start_date:
                date_filter &= Q(log__timestamp__date__gte=start_date)
            if end_date:
                date_filter &= Q(log__timestamp__date__lte=end_date)
            queryset = queryset.filter(date_filter)
        
        if turbine_ids:
            queryset = queryset.filter(log__turbine_id__in=turbine_ids)
        
        readings = queryset.filter(log__predictions__isnull=True)[:1]
        
        if not readings.exists():
            return {'success': True, 'message': 'No more readings to process'}
        
        reading = readings.first()
        log = reading.log
        
        service = HealthPredictionService()
        prediction = service.predict_health(log)
        
        if prediction:
            next_reading = queryset.filter(
                log__timestamp__gt=log.timestamp
            ).first()
            
            if next_reading:
                simulate_historical_feed.apply_async(
                    args=[start_date.strftime('%Y-%m-%d') if start_date else None,
                          end_date.strftime('%Y-%m-%d') if end_date else None,
                          turbine_ids, interval_seconds],
                    countdown=interval_seconds
                )
        
        return {
            'success': True,
            'processed_log_id': str(log.id),
            'prediction_created': prediction is not None,
            'next_scheduled': next_reading is not None
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


@shared_task
def periodic_health_check():
    """
    Periodic task to check and process any pending readings.
    Run this on a schedule (e.g., every 10 minutes).
    """
    return process_pending_readings(batch_size=50)
