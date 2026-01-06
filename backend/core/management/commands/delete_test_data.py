from django.core.management.base import BaseCommand
from core.models.turbine import Turbine
from core.models.turbines.turbine_log import TurbineLog
from core.models.turbines.on_turbine_reading import OnTurbineReading
from core.models.turbines.weather_reading import WeatherReading
from core.models.turbines.alert import Alert
from core.models.turbines.health_prediction import HealthPrediction
from core.models.turbines.maintenance_event import MaintenanceEvent


class Command(BaseCommand):
    help = 'Delete all test data (where is_dummy_data=True)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--confirm',
            action='store_true',
            help='Confirm deletion without prompt',
        )

    def handle(self, *args, **options):
        confirm = options.get('confirm', False)
        
        if not confirm:
            self.stdout.write(self.style.WARNING('This will delete all records where is_dummy_data=True'))
            response = input('Are you sure you want to continue? (yes/no): ')
            if response.lower() != 'yes':
                self.stdout.write(self.style.SUCCESS('Deletion cancelled.'))
                return
        
        self.stdout.write('Deleting test data...')
        
        alerts_count = Alert.objects.filter(is_dummy_data=True).count()
        health_predictions_count = HealthPrediction.objects.filter(is_dummy_data=True).count()
        maintenance_events_count = MaintenanceEvent.objects.filter(is_dummy_data=True).count()
        on_turbine_readings_count = OnTurbineReading.objects.filter(is_dummy_data=True).count()
        weather_readings_count = WeatherReading.objects.filter(is_dummy_data=True).count()
        turbine_logs_count = TurbineLog.objects.filter(is_dummy_data=True).count()
        turbines_count = Turbine.objects.filter(is_dummy_data=True).count()
        
        Alert.objects.filter(is_dummy_data=True).delete()
        HealthPrediction.objects.filter(is_dummy_data=True).delete()
        MaintenanceEvent.objects.filter(is_dummy_data=True).delete()
        OnTurbineReading.objects.filter(is_dummy_data=True).delete()
        WeatherReading.objects.filter(is_dummy_data=True).delete()
        TurbineLog.objects.filter(is_dummy_data=True).delete()
        Turbine.objects.filter(is_dummy_data=True).delete()
        
        self.stdout.write(self.style.SUCCESS(f'\nSuccessfully deleted test data:'))
        self.stdout.write(f'  - Turbines: {turbines_count}')
        self.stdout.write(f'  - Turbine Logs: {turbine_logs_count}')
        self.stdout.write(f'  - On-Turbine Readings: {on_turbine_readings_count}')
        self.stdout.write(f'  - Weather Readings: {weather_readings_count}')
        self.stdout.write(f'  - Health Predictions: {health_predictions_count}')
        self.stdout.write(f'  - Alerts: {alerts_count}')
        self.stdout.write(f'  - Maintenance Events: {maintenance_events_count}')

