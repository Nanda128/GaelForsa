from django.test import TestCase
from django.core.exceptions import ValidationError
from django.utils import timezone
from core.models import Turbine, TurbineLog, OnTurbineReading, WeatherReading
from core.models import MaintenanceEvent, HealthPrediction, Alert


class TurbineModelTest(TestCase):
    def test_create_turbine(self):
        turbine = Turbine.objects.create(
            name="Test Turbine",
            latitude=55.5,
            longitude=-1.5
        )
        self.assertIsNotNone(turbine.id)
        self.assertEqual(turbine.name, "Test Turbine")
        self.assertEqual(turbine.latitude, 55.5)
        self.assertEqual(turbine.longitude, -1.5)
        self.assertIsNotNone(turbine.created_at)
        self.assertIsNotNone(turbine.updated_at)

    def test_turbine_str(self):
        turbine = Turbine.objects.create(
            name="Test Turbine",
            latitude=55.5,
            longitude=-1.5
        )
        self.assertEqual(str(turbine), "Test Turbine")


class TurbineLogModelTest(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(
            name="Test Turbine",
            latitude=55.5,
            longitude=-1.5
        )

    def test_create_turbine_log(self):
        log = TurbineLog.objects.create(turbine=self.turbine)
        self.assertIsNotNone(log.id)
        self.assertEqual(log.turbine, self.turbine)
        self.assertIsNotNone(log.timestamp)

    def test_turbine_log_str(self):
        log = TurbineLog.objects.create(turbine=self.turbine)
        self.assertIn(self.turbine.name, str(log))
        self.assertIn(str(log.timestamp), str(log))


class OnTurbineReadingModelTest(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(
            name="Test Turbine",
            latitude=55.5,
            longitude=-1.5
        )
        self.log = TurbineLog.objects.create(turbine=self.turbine)

    def test_create_on_turbine_reading(self):
        reading = OnTurbineReading.objects.create(
            log=self.log,
            rotor_speed=15.5,
            power_output=2500.0,
            generator_temperature=45.0,
            vibration_level=0.5
        )
        self.assertIsNotNone(reading.id)
        self.assertEqual(reading.rotor_speed, 15.5)
        self.assertEqual(reading.power_output, 2500.0)

    def test_on_turbine_reading_nullable_fields(self):
        reading = OnTurbineReading.objects.create(log=self.log)
        self.assertIsNone(reading.rotor_speed)
        self.assertIsNone(reading.power_output)


class WeatherReadingModelTest(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(
            name="Test Turbine",
            latitude=55.5,
            longitude=-1.5
        )
        self.log = TurbineLog.objects.create(turbine=self.turbine)

    def test_create_weather_reading(self):
        reading = WeatherReading.objects.create(
            log=self.log,
            wind_speed=12.5,
            wind_direction=180.0,
            wave_height=2.5,
            wave_period=8.0,
            air_temperature=15.0
        )
        self.assertIsNotNone(reading.id)
        self.assertEqual(reading.wind_speed, 12.5)
        self.assertEqual(reading.wave_period, 8.0)

    def test_lightning_strikes_default(self):
        reading = WeatherReading.objects.create(log=self.log)
        self.assertEqual(reading.lightning_strikes, 0)


class MaintenanceEventModelTest(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(
            name="Test Turbine",
            latitude=55.5,
            longitude=-1.5
        )

    def test_create_maintenance_event(self):
        event = MaintenanceEvent.objects.create(
            turbine=self.turbine,
            event_type='failure',
            start_time=timezone.now(),
            description="Test failure",
            cost=5000.00
        )
        self.assertIsNotNone(event.id)
        self.assertEqual(event.event_type, 'failure')
        self.assertEqual(event.cost, 5000.00)

    def test_maintenance_event_str(self):
        event = MaintenanceEvent.objects.create(
            turbine=self.turbine,
            event_type='failure',
            start_time=timezone.now()
        )
        self.assertIn(self.turbine.name, str(event))
        self.assertIn('failure', str(event))


class HealthPredictionModelTest(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(
            name="Test Turbine",
            latitude=55.5,
            longitude=-1.5
        )
        self.log = TurbineLog.objects.create(turbine=self.turbine)

    def test_create_health_prediction(self):
        prediction = HealthPrediction.objects.create(
            turbine=self.turbine,
            log=self.log,
            health_score=0.85,
            failure_probability=0.15,
            model_version="v1.0"
        )
        self.assertIsNotNone(prediction.id)
        self.assertEqual(prediction.health_score, 0.85)
        self.assertEqual(prediction.failure_probability, 0.15)

    def test_health_prediction_requires_log(self):
        with self.assertRaises(Exception):
            HealthPrediction.objects.create(
                turbine=self.turbine,
                health_score=0.85,
                failure_probability=0.15
            )


class AlertModelTest(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(
            name="Test Turbine",
            latitude=55.5,
            longitude=-1.5
        )
        self.log = TurbineLog.objects.create(turbine=self.turbine)
        self.prediction = HealthPrediction.objects.create(
            turbine=self.turbine,
            log=self.log,
            health_score=0.5,
            failure_probability=0.8
        )

    def test_create_alert(self):
        alert = Alert.objects.create(
            turbine=self.turbine,
            prediction=self.prediction,
            severity='critical',
            message="High failure probability detected"
        )
        self.assertIsNotNone(alert.id)
        self.assertEqual(alert.severity, 'critical')
        self.assertFalse(alert.acknowledged)

    def test_alert_without_prediction(self):
        alert = Alert.objects.create(
            turbine=self.turbine,
            severity='warning',
            message="Manual alert"
        )
        self.assertIsNone(alert.prediction)

