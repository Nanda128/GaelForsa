"""
API views for receiving real-time SCADA data and triggering predictions.
"""
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from datetime import datetime
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from core.models import Turbine, TurbineLog, OnTurbineReading
from core.services.health_prediction_service import HealthPredictionService
from core.tasks import process_realtime_reading


@swagger_auto_schema(
    method='post',
    operation_description='Receive real-time SCADA reading and trigger health prediction',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['turbine_id', 'timestamp', 'readings'],
        properties={
            'turbine_id': openapi.Schema(type=openapi.TYPE_STRING, description='Turbine ID or name'),
            'timestamp': openapi.Schema(type=openapi.TYPE_STRING, format=openapi.FORMAT_DATETIME, description='Reading timestamp (ISO 8601)'),
            'readings': openapi.Schema(
                type=openapi.TYPE_OBJECT,
                description='Sensor readings',
                properties={
                    'wind_speed': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'wind_direction': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'power_output': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'rotor_speed_rpm': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'generator_rpm': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'blade_pitch_angle': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'gear_oil_temperature': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'generator_temperature': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'yaw_position': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'reactive_power': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'vibration_level': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'gearbox_oil_temperature': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'gearbox_oil_pressure': openapi.Schema(type=openapi.TYPE_NUMBER),
                }
            ),
            'async': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Process prediction asynchronously via Celery', default=False),
        }
    ),
    responses={
        201: openapi.Response('Reading created and prediction generated'),
        400: openapi.Response('Invalid input data'),
    },
    tags=['Real-time Data']
)
@api_view(['POST'])
@permission_classes([AllowAny])
def receive_realtime_reading(request):
    """
    Receive a real-time SCADA reading and optionally trigger health prediction.
    
    This endpoint accepts sensor data, creates a TurbineLog and OnTurbineReading,
    and can trigger health prediction either synchronously or asynchronously.
    """
    try:
        # Parse request data
        turbine_id = request.data.get('turbine_id')
        timestamp_str = request.data.get('timestamp')
        readings_data = request.data.get('readings', {})
        async_processing = request.data.get('async', False)
        
        if not turbine_id:
            return Response({'error': 'turbine_id is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not timestamp_str:
            return Response({'error': 'timestamp is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Parse timestamp
        try:
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = timestamp_str
            if timezone.is_naive(timestamp):
                timestamp = timezone.make_aware(timestamp)
        except (ValueError, TypeError) as e:
            return Response({'error': f'Invalid timestamp format: {e}'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Get or create turbine
        try:
            if isinstance(turbine_id, str) and turbine_id.replace('-', '').replace('_', '').isalnum() and len(turbine_id) == 36:
                # UUID format
                turbine = Turbine.objects.get(id=turbine_id)
            else:
                # Name format
                turbine = Turbine.objects.get(name=turbine_id)
        except Turbine.DoesNotExist:
            return Response({'error': f'Turbine {turbine_id} not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Create or get TurbineLog
        log, created = TurbineLog.objects.get_or_create(
            turbine=turbine,
            timestamp=timestamp,
            defaults={'is_dummy_data': False}
        )
        
        # Create or update OnTurbineReading
        reading, reading_created = OnTurbineReading.objects.update_or_create(
            log=log,
            defaults=readings_data
        )
        
        # Trigger health prediction
        prediction = None
        if async_processing:
            # Process asynchronously via Celery
            task = process_realtime_reading.delay(str(log.id))
            return Response({
                'status': 'accepted',
                'log_id': str(log.id),
                'reading_id': str(reading.id),
                'task_id': task.id,
                'message': 'Reading received, prediction processing in background'
            }, status=status.HTTP_202_ACCEPTED)
        else:
            # Process synchronously
            service = HealthPredictionService()
            prediction = service.predict_health(log)
            
            return Response({
                'status': 'created',
                'log_id': str(log.id),
                'reading_id': str(reading.id),
                'prediction_id': str(prediction.id) if prediction else None,
                'message': 'Reading received and prediction generated'
            }, status=status.HTTP_201_CREATED)
    
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='post',
    operation_description='Trigger health prediction for a specific log',
    manual_parameters=[
        openapi.Parameter('log_id', openapi.IN_PATH, description='TurbineLog ID', type=openapi.TYPE_STRING),
    ],
    responses={
        200: openapi.Response('Prediction generated'),
        404: openapi.Response('Log not found'),
    },
    tags=['Real-time Data']
)
@api_view(['POST'])
@permission_classes([AllowAny])
def trigger_prediction(request, log_id):
    """
    Manually trigger health prediction for a specific TurbineLog.
    """
    try:
        log = TurbineLog.objects.get(id=log_id)
        service = HealthPredictionService()
        prediction = service.predict_health(log)
        
        if prediction:
            return Response({
                'status': 'success',
                'prediction_id': str(prediction.id),
                'health_score': prediction.health_score,
                'failure_probability': prediction.failure_probability,
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                'status': 'failed',
                'message': 'Could not generate prediction'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    except TurbineLog.DoesNotExist:
        return Response({'error': 'Log not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
