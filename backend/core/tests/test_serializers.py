from django.test import TestCase
from django.utils import timezone
from core.models import Turbine, TurbineLog, OnTurbineReading, WeatherReading
from core.models import MaintenanceEvent, HealthPrediction, Alert
from core.serializers import (
    TurbineSerializer, TurbineLogSerializer, OnTurbineReadingSerializer,
    WeatherReadingSerializer, MaintenanceEventSerializer,
    HealthPredictionSerializer, AlertSerializer
)


class TurbineSerializerTest(TestCase):
    def setUp(self):
        self.turbine_data = {
            'name': 'Test Turbine',
            'latitude': 55.5,
            'longitude': -1.5
        }

    def test_turbine_serializer_create(self):
        serializer = TurbineSerializer(data=self.turbine_data)
        self.assertTrue(serializer.is_valid())
        turbine = serializer.save()
        self.assertEqual(turbine.name, 'Test Turbine')

    def test_turbine_serializer_read_only_fields(self):
        turbine = Turbine.objects.create(**self.turbine_data)
        serializer = TurbineSerializer(turbine)
        data = serializer.data
        self.assertIn('id', data)
        self.assertIn('created_at', data)
        self.assertIn('updated_at', data)


class TurbineLogSerializerTest(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(name='Test Turbine', latitude=55.5, longitude=-1.5)
        self.log = TurbineLog.objects.create(turbine=self.turbine)

    def test_turbine_log_serializer_with_readings(self):
        OnTurbineReading.objects.create(
            log=self.log,
            rotor_speed=15.5,
            power_output=2500.0
        )
        WeatherReading.objects.create(
            log=self.log,
            wind_speed=12.5,
            wind_direction=180.0
        )
        
        serializer = TurbineLogSerializer(self.log)
        data = serializer.data
        self.assertIn('on_turbine_readings', data)
        self.assertIn('weather_readings', data)
        self.assertEqual(len(data['on_turbine_readings']), 1)
        self.assertEqual(len(data['weather_readings']), 1)

    def test_turbine_log_serializer_read_only_fields(self):
        serializer = TurbineLogSerializer(self.log)
        data = serializer.data
        self.assertIn('id', data)
        self.assertIn('timestamp', data)
        self.assertIn('turbine', data)


class MaintenanceEventSerializerTest(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(name='Test Turbine', latitude=55.5, longitude=-1.5)

    def test_maintenance_event_serializer(self):
        event_data = {
            'event_type': 'failure',
            'start_time': timezone.now(),
            'description': 'Test failure',
            'cost': '5000.00'
        }
        serializer = MaintenanceEventSerializer(data=event_data)
        self.assertTrue(serializer.is_valid())
        event = serializer.save(turbine=self.turbine)
        self.assertEqual(event.event_type, 'failure')

    def test_maintenance_event_read_only_fields(self):
        event = MaintenanceEvent.objects.create(
            turbine=self.turbine,
            event_type='failure',
            start_time=timezone.now()
        )
        serializer = MaintenanceEventSerializer(event)
        data = serializer.data
        self.assertIn('id', data)
        self.assertIn('created_at', data)
        self.assertIn('turbine', data)


class HealthPredictionSerializerTest(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(name='Test Turbine', latitude=55.5, longitude=-1.5)
        self.log = TurbineLog.objects.create(turbine=self.turbine)

    def test_health_prediction_serializer(self):
        prediction_data = {
            'log': self.log.id,
            'health_score': 0.85,
            'failure_probability': 0.15,
            'model_version': 'v1.0'
        }
        serializer = HealthPredictionSerializer(data=prediction_data)
        self.assertTrue(serializer.is_valid())
        prediction = serializer.save(turbine=self.turbine)
        self.assertEqual(prediction.health_score, 0.85)

    def test_health_prediction_requires_log(self):
        prediction_data = {
            'health_score': 0.85,
            'failure_probability': 0.15
        }
        serializer = HealthPredictionSerializer(data=prediction_data)
        self.assertFalse(serializer.is_valid())


class AlertSerializerTest(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(name='Test Turbine', latitude=55.5, longitude=-1.5)
        self.log = TurbineLog.objects.create(turbine=self.turbine)
        self.prediction = HealthPrediction.objects.create(
            turbine=self.turbine,
            log=self.log,
            health_score=0.5,
            failure_probability=0.8
        )

    def test_alert_serializer(self):
        alert_data = {
            'prediction': self.prediction.id,
            'severity': 'critical',
            'message': 'High failure probability detected'
        }
        serializer = AlertSerializer(data=alert_data)
        self.assertTrue(serializer.is_valid())
        alert = serializer.save(turbine=self.turbine)
        self.assertEqual(alert.severity, 'critical')

    def test_alert_without_prediction(self):
        alert_data = {
            'severity': 'warning',
            'message': 'Manual alert'
        }
        serializer = AlertSerializer(data=alert_data)
        self.assertTrue(serializer.is_valid())
        alert = serializer.save(turbine=self.turbine)
        self.assertEqual(alert.severity, 'warning')

