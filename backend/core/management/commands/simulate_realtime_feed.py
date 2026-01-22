"""
Management command for simulating real-time data feed.

NOTE: This command is for batch processing historical data.
For actual real-time simulation, use either:
1. The standalone realtime_feed_service.py script
2. Celery tasks (core.tasks.simulate_historical_feed)

This command processes readings in batch and generates predictions.
"""
from datetime import datetime
from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone
from core.models import OnTurbineReading, Turbine
from core.services.health_prediction_service import HealthPredictionService
from core.tasks import process_pending_readings


class Command(BaseCommand):
    help = 'Batch process historical readings and generate health predictions'

    def add_arguments(self, parser):
        parser.add_argument(
            '--start-date',
            type=str,
            help='Start date for processing (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--end-date',
            type=str,
            help='End date for processing (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--turbines',
            type=str,
            help='Comma-separated list of turbine numbers to filter (e.g., "1,2,3")'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of readings to process in each batch (default: 100)'
        )
        parser.add_argument(
            '--async',
            action='store_true',
            help='Process predictions asynchronously using Celery'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run simulation without storing predictions'
        )

    def handle(self, *args, **options):
        start_date_str = options['start_date']
        end_date_str = options['end_date']
        turbines_str = options['turbines']
        batch_size = options['batch_size']
        use_async = options['async']
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

        # Build query for readings without predictions
        queryset = OnTurbineReading.objects.select_related(
            'log__turbine'
        ).filter(
            log__predictions__isnull=True
        ).order_by('log__timestamp')

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

        self.stdout.write(f'Starting batch processing with {total_readings} readings')
        self.stdout.write(f'Batch size: {batch_size}')
        if use_async:
            self.stdout.write('Using asynchronous processing (Celery)')
        if dry_run:
            self.stdout.write('DRY RUN MODE - No predictions will be stored')

        if use_async:
            # Use Celery for async processing
            if turbine_ids:
                # Convert turbine names to IDs
                turbines = Turbine.objects.filter(name__in=[f"Kelmarsh {tid}" for tid in turbine_ids])
                turbine_uuids = [str(t.id) for t in turbines]
            else:
                turbine_uuids = None
            
            # Process in batches via Celery
            processed = 0
            while processed < total_readings:
                result = process_pending_readings.delay(
                    turbine_id=turbine_uuids[0] if turbine_uuids and len(turbine_uuids) == 1 else None,
                    batch_size=batch_size
                )
                batch_result = result.get(timeout=300)  # 5 minute timeout
                processed += batch_result.get('processed', 0)
                
                self.stdout.write(
                    f'Batch processed: {batch_result.get("processed", 0)} readings, '
                    f'{batch_result.get("predictions_created", 0)} predictions created'
                )
                
                if batch_result.get('processed', 0) == 0:
                    break
        else:
            # Synchronous processing
            service = HealthPredictionService()
            processed = 0
            predictions_created = 0

            # Process in batches
            for i in range(0, total_readings, batch_size):
                batch = queryset[i:i + batch_size]
                
                for reading in batch:
                    if not dry_run:
                        try:
                            prediction = service.predict_health(reading.log)
                            if prediction:
                                predictions_created += 1
                        except Exception as e:
                            self.stdout.write(
                                self.style.ERROR(f'Error processing reading at {reading.log.timestamp}: {e}')
                            )
                    
                    processed += 1

                    # Progress update
                    if processed % 100 == 0:
                        progress = (processed / total_readings) * 100
                        self.stdout.write(f'Progress: {progress:.1f}% ({processed}/{total_readings})')

            # Final statistics
            self.stdout.write(self.style.SUCCESS(
                f'Processing completed. Processed {processed} readings, created {predictions_created} predictions'
            ))