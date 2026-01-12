import time
from datetime import datetime
from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone
from core.models import OnTurbineReading, Turbine
from core.services.health_prediction_service import HealthPredictionService


class Command(BaseCommand):
    help = 'Simulate real-time data feed from 2016-2018, generating health predictions'

    def add_arguments(self, parser):
        parser.add_argument(
            '--speed-multiplier',
            type=float,
            default=10.0,
            help='Speed multiplier for simulation (e.g., 10.0 = 10x faster than real-time)'
        )
        parser.add_argument(
            '--start-date',
            type=str,
            help='Start date for simulation (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--end-date',
            type=str,
            help='End date for simulation (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--turbines',
            type=str,
            help='Comma-separated list of turbine numbers to filter (e.g., "1,2,3")'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run simulation without storing predictions'
        )

    def handle(self, *args, **options):
        speed_multiplier = options['speed_multiplier']
        start_date_str = options['start_date']
        end_date_str = options['end_date']
        turbines_str = options['turbines']
        dry_run = options['dry_run']

        # Parse dates
        start_date = None
        end_date = None
        if start_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        # Parse turbine IDs
        turbine_ids = None
        if turbines_str:
            turbine_ids = [int(x.strip()) for x in turbines_str.split(',')]

        # Build query
        queryset = OnTurbineReading.objects.select_related('log__turbine').order_by('log__timestamp')

        if start_date or end_date:
            date_filter = Q()
            if start_date:
                date_filter &= Q(log__timestamp__date__gte=start_date)
            if end_date:
                date_filter &= Q(log__timestamp__date__lte=end_date)
            queryset = queryset.filter(date_filter)
        if turbine_ids:

            turbine_names = [f"Kelmarsh {tid}" for tid in turbine_ids]

            queryset = queryset.filter(log__turbine__name__in=turbine_names)

        # Get total count for progress
        total_readings = queryset.count()
        if total_readings == 0:
            self.stdout.write(self.style.WARNING('No readings found matching the criteria'))
            return

        self.stdout.write(f'Starting simulation with {total_readings} readings')
        self.stdout.write(f'Speed multiplier: {speed_multiplier}x')
        if dry_run:
            self.stdout.write('DRY RUN MODE - No predictions will be stored')

        # Initialize service
        service = HealthPredictionService()

        # Process readings
        processed = 0
        predictions_created = 0
        last_timestamp = None

        for reading in queryset.iterator():
            current_timestamp = reading.log.timestamp

            # Calculate sleep time
            if last_timestamp:
                time_diff = (current_timestamp - last_timestamp).total_seconds()
                sleep_time = time_diff / speed_multiplier
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Generate prediction
            try:
                if not dry_run:
                    prediction = service.predict_health(reading.log)
                    if prediction:
                        predictions_created += 1
                else:
                    # In dry run, just simulate the call
                    prediction = None

                processed += 1

                # Progress update every 100 readings
                if processed % 100 == 0:
                    progress = (processed / total_readings) * 100
                    self.stdout.write(f'Progress: {progress:.1f}% ({processed}/{total_readings})')

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing reading at {current_timestamp}: {e}'))

            last_timestamp = current_timestamp

        # Final statistics
        self.stdout.write(self.style.SUCCESS(
            f'Simulation completed. Processed {processed} readings, created {predictions_created} predictions'
        ))