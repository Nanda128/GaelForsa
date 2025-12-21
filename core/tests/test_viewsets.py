from django.utils import timezone
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from core.models import Turbine, TurbineLog, OnTurbineReading, WeatherReading
from core.models import MaintenanceEvent, HealthPrediction, Alert


class TurbineViewSetTest(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.turbine_data = {
            'name': 'Test Turbine',
            'latitude': 55.5,
            'longitude': -1.5
        }

    def test_list_turbines_public(self):
        Turbine.objects.create(name='Turbine 1', latitude=55.0, longitude=-1.0)
        Turbine.objects.create(name='Turbine 2', latitude=56.0, longitude=-2.0)
        
        response = self.client.get('/api/v1/turbines/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 2)

    def test_retrieve_turbine_public(self):
        turbine = Turbine.objects.create(**self.turbine_data)
        response = self.client.get(f'/api/v1/turbines/{turbine.id}/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['name'], 'Test Turbine')

    def test_create_turbine(self):
        response = self.client.post('/api/v1/turbines/', self.turbine_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Turbine.objects.count(), 1)

    def test_update_turbine(self):
        turbine = Turbine.objects.create(**self.turbine_data)
        update_data = {'name': 'Updated Turbine', 'latitude': 56.0, 'longitude': -2.0}
        
        response = self.client.put(f'/api/v1/turbines/{turbine.id}/', update_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        turbine.refresh_from_db()
        self.assertEqual(turbine.name, 'Updated Turbine')

    def test_delete_turbine(self):
        turbine = Turbine.objects.create(**self.turbine_data)
        
        response = self.client.delete(f'/api/v1/turbines/{turbine.id}/')
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(Turbine.objects.count(), 0)


class TurbineLogViewSetTest(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.turbine = Turbine.objects.create(name='Test Turbine', latitude=55.5, longitude=-1.5)

    def test_list_logs_public(self):
        log1 = TurbineLog.objects.create(turbine=self.turbine)
        log2 = TurbineLog.objects.create(turbine=self.turbine)
        
        response = self.client.get(f'/api/v1/turbines/{self.turbine.id}/logs/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 2)

    def test_create_log(self):
        response = self.client.post(f'/api/v1/turbines/{self.turbine.id}/logs/', {})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(TurbineLog.objects.filter(turbine=self.turbine).count(), 1)

    def test_create_log_with_readings(self):
        log_data = {
            'on_turbine_readings': [{
                'rotor_speed': 15.5,
                'power_output': 2500.0
            }],
            'weather_readings': [{
                'wind_speed': 12.5,
                'wind_direction': 180.0
            }]
        }
        response = self.client.post(
            f'/api/v1/turbines/{self.turbine.id}/logs/',
            log_data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        log = TurbineLog.objects.get(turbine=self.turbine)
        self.assertEqual(log.on_turbine_readings.count(), 1)
        self.assertEqual(log.weather_readings.count(), 1)

    def test_logs_filtered_by_turbine_public(self):
        turbine2 = Turbine.objects.create(name='Turbine 2', latitude=56.0, longitude=-2.0)
        TurbineLog.objects.create(turbine=self.turbine)
        TurbineLog.objects.create(turbine=turbine2)
        
        response = self.client.get(f'/api/v1/turbines/{self.turbine.id}/logs/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)

    def test_invalid_turbine_id(self):
        response = self.client.get('/api/v1/turbines/invalid-uuid/logs/')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)


class MaintenanceEventViewSetTest(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.turbine = Turbine.objects.create(name='Test Turbine', latitude=55.5, longitude=-1.5)

    def test_list_maintenance_events_public(self):
        MaintenanceEvent.objects.create(
            turbine=self.turbine,
            event_type='failure',
            start_time=timezone.now()
        )
        response = self.client.get(f'/api/v1/turbines/{self.turbine.id}/maintenance-events/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)

    def test_create_maintenance_event(self):
        event_data = {
            'event_type': 'failure',
            'start_time': timezone.now().isoformat(),
            'description': 'Test failure',
            'cost': '5000.00'
        }
        response = self.client.post(
            f'/api/v1/turbines/{self.turbine.id}/maintenance-events/',
            event_data
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(MaintenanceEvent.objects.filter(turbine=self.turbine).count(), 1)

    def test_maintenance_event_with_log(self):
        log = TurbineLog.objects.create(turbine=self.turbine)
        event_data = {
            'event_type': 'failure',
            'start_time': timezone.now().isoformat(),
            'log': str(log.id)
        }
        response = self.client.post(
            f'/api/v1/turbines/{self.turbine.id}/maintenance-events/',
            event_data
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        event = MaintenanceEvent.objects.get(turbine=self.turbine)
        self.assertEqual(event.log, log)

    def test_maintenance_event_log_validation(self):
        turbine2 = Turbine.objects.create(name='Turbine 2', latitude=56.0, longitude=-2.0)
        log2 = TurbineLog.objects.create(turbine=turbine2)
        event_data = {
            'event_type': 'failure',
            'start_time': timezone.now().isoformat(),
            'log': str(log2.id)
        }
        response = self.client.post(
            f'/api/v1/turbines/{self.turbine.id}/maintenance-events/',
            event_data
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class HealthPredictionViewSetTest(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.turbine = Turbine.objects.create(name='Test Turbine', latitude=55.5, longitude=-1.5)
        self.log = TurbineLog.objects.create(turbine=self.turbine)

    def test_list_health_predictions_public(self):
        HealthPrediction.objects.create(
            turbine=self.turbine,
            log=self.log,
            health_score=0.85,
            failure_probability=0.15
        )
        response = self.client.get(f'/api/v1/turbines/{self.turbine.id}/health-predictions/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)

    def test_create_health_prediction(self):
        prediction_data = {
            'log': str(self.log.id),
            'health_score': 0.85,
            'failure_probability': 0.15,
            'model_version': 'v1.0'
        }
        response = self.client.post(
            f'/api/v1/turbines/{self.turbine.id}/health-predictions/',
            prediction_data
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(HealthPrediction.objects.filter(turbine=self.turbine).count(), 1)

    def test_health_prediction_requires_log(self):
        prediction_data = {
            'health_score': 0.85,
            'failure_probability': 0.15
        }
        response = self.client.post(
            f'/api/v1/turbines/{self.turbine.id}/health-predictions/',
            prediction_data
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_health_prediction_log_validation(self):
        turbine2 = Turbine.objects.create(name='Turbine 2', latitude=56.0, longitude=-2.0)
        log2 = TurbineLog.objects.create(turbine=turbine2)
        prediction_data = {
            'log': str(log2.id),
            'health_score': 0.85,
            'failure_probability': 0.15
        }
        response = self.client.post(
            f'/api/v1/turbines/{self.turbine.id}/health-predictions/',
            prediction_data
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class AlertViewSetTest(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.turbine = Turbine.objects.create(name='Test Turbine', latitude=55.5, longitude=-1.5)
        self.log = TurbineLog.objects.create(turbine=self.turbine)
        self.prediction = HealthPrediction.objects.create(
            turbine=self.turbine,
            log=self.log,
            health_score=0.5,
            failure_probability=0.8
        )

    def test_list_alerts_public(self):
        Alert.objects.create(
            turbine=self.turbine,
            severity='warning',
            message='Test alert'
        )
        response = self.client.get(f'/api/v1/turbines/{self.turbine.id}/alerts/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)

    def test_create_alert(self):
        alert_data = {
            'severity': 'critical',
            'message': 'High failure probability detected'
        }
        response = self.client.post(
            f'/api/v1/turbines/{self.turbine.id}/alerts/',
            alert_data
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Alert.objects.filter(turbine=self.turbine).count(), 1)

    def test_create_alert_with_prediction(self):
        alert_data = {
            'prediction': str(self.prediction.id),
            'severity': 'critical',
            'message': 'High failure probability detected'
        }
        response = self.client.post(
            f'/api/v1/turbines/{self.turbine.id}/alerts/',
            alert_data
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        alert = Alert.objects.get(turbine=self.turbine)
        self.assertEqual(alert.prediction, self.prediction)

    def test_acknowledge_alert(self):
        alert = Alert.objects.create(
            turbine=self.turbine,
            severity='warning',
            message='Test alert'
        )
        update_data = {'acknowledged': True}
        
        response = self.client.patch(
            f'/api/v1/turbines/{self.turbine.id}/alerts/{alert.id}/',
            update_data
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        alert.refresh_from_db()
        self.assertTrue(alert.acknowledged)

