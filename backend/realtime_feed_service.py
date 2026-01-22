#!/usr/bin/env python
"""
Standalone real-time feed service.

This service reads historical SCADA data from the database and feeds it to the API
at real-time intervals, simulating a live data stream.

Usage:
    python realtime_feed_service.py --start-date 2016-01-01 --end-date 2018-12-31 --interval 600 --api-url http://localhost:8000

This service runs independently of Django and can be started/stopped separately.
"""
import os
import sys
import time
import argparse
import requests
from datetime import datetime, timedelta
from typing import Optional

# Add Django to path
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'api.settings')

import django
django.setup()

from django.db.models import Q
from django.utils import timezone
from core.models import Turbine, TurbineLog, OnTurbineReading


class RealtimeFeedService:
    """Service that feeds historical data to API at real-time intervals."""
    
    def __init__(self, api_url: str = 'http://localhost:8000', interval_seconds: int = 600):
        """
        Initialize the service.
        
        Args:
            api_url: Base URL of the API
            interval_seconds: Seconds between readings (default 10 minutes)
        """
        self.api_url = api_url.rstrip('/')
        self.interval_seconds = interval_seconds
        self.processed_count = 0
        self.error_count = 0
    
    def get_reading_data(self, reading: OnTurbineReading) -> dict:
        """Convert OnTurbineReading to API format."""
        return {
            'turbine_id': str(reading.log.turbine.id),
            'timestamp': reading.log.timestamp.isoformat(),
            'readings': {
                'wind_speed': reading.wind_speed,
                'wind_direction': reading.wind_direction,
                'power_output': reading.power_output,
                'rotor_speed_rpm': reading.rotor_speed_rpm,
                'generator_rpm': reading.generator_rpm,
                'blade_pitch_angle': reading.blade_pitch_angle,
                'gear_oil_temperature': reading.gear_oil_temperature or reading.gearbox_oil_temperature,
                'generator_temperature': reading.generator_temperature,
                'yaw_position': reading.yaw_position,
                'reactive_power': reading.reactive_power,
                'vibration_level': reading.vibration_level,
                'gearbox_oil_temperature': reading.gearbox_oil_temperature or reading.gear_oil_temperature,
                'gearbox_oil_pressure': reading.gearbox_oil_pressure,
                'front_bearing_temperature': reading.front_bearing_temperature,
                'rear_bearing_temperature': reading.rear_bearing_temperature,
                'nacelle_temperature': reading.nacelle_temperature,
                'transformer_temperature': reading.transformer_temperature,
            },
            'async': True,  # Process predictions asynchronously
        }
    
    def send_reading(self, reading: OnTurbineReading) -> bool:
        """
        Send a reading to the API.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self.get_reading_data(reading)
            response = requests.post(
                f'{self.api_url}/api/v1/realtime/reading/',
                json=data,
                timeout=10
            )
            
            if response.status_code in [201, 202]:
                self.processed_count += 1
                return True
            else:
                print(f"Error sending reading: {response.status_code} - {response.text}")
                self.error_count += 1
                return False
        except Exception as e:
            print(f"Exception sending reading: {e}")
            self.error_count += 1
            return False
    
    def run(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
            turbine_ids: Optional[list] = None, speed_multiplier: float = 1.0):
        """
        Run the feed service.
        
        Args:
            start_date: Start date for readings
            end_date: End date for readings
            turbine_ids: List of turbine IDs to process
            speed_multiplier: Speed multiplier (1.0 = real-time, 10.0 = 10x faster)
        """
        # Build query
        queryset = OnTurbineReading.objects.select_related(
            'log__turbine'
        ).order_by('log__timestamp')
        
        if start_date:
            queryset = queryset.filter(log__timestamp__gte=start_date)
        if end_date:
            queryset = queryset.filter(log__timestamp__lte=end_date)
        if turbine_ids:
            queryset = queryset.filter(log__turbine_id__in=turbine_ids)
        
        # Get readings that haven't been processed yet (no predictions)
        queryset = queryset.filter(log__predictions__isnull=True)
        
        total_readings = queryset.count()
        print(f"Starting real-time feed service")
        print(f"Total readings to process: {total_readings}")
        print(f"Interval: {self.interval_seconds} seconds")
        print(f"Speed multiplier: {speed_multiplier}x")
        print(f"API URL: {self.api_url}")
        print("-" * 60)
        
        last_timestamp = None
        
        for reading in queryset.iterator():
            current_timestamp = reading.log.timestamp
            
            # Calculate sleep time based on actual time difference
            if last_timestamp:
                time_diff = (current_timestamp - last_timestamp).total_seconds()
                sleep_time = max(0, time_diff / speed_multiplier)
                
                if sleep_time > 0:
                    print(f"Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
            
            # Send reading to API
            print(f"[{current_timestamp}] Sending reading for {reading.log.turbine.name}...", end=' ')
            success = self.send_reading(reading)
            
            if success:
                print("✓")
            else:
                print("✗")
            
            # Progress update
            if (self.processed_count + self.error_count) % 100 == 0:
                progress = ((self.processed_count + self.error_count) / total_readings) * 100
                print(f"Progress: {progress:.1f}% ({self.processed_count + self.error_count}/{total_readings})")
            
            last_timestamp = current_timestamp
        
        print("-" * 60)
        print(f"Feed service completed")
        print(f"Processed: {self.processed_count}")
        print(f"Errors: {self.error_count}")


def main():
    parser = argparse.ArgumentParser(description='Real-time feed service for SCADA data')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--turbines', type=str, help='Comma-separated turbine IDs')
    parser.add_argument('--interval', type=int, default=600, help='Interval between readings in seconds (default: 600)')
    parser.add_argument('--speed-multiplier', type=float, default=1.0, help='Speed multiplier (default: 1.0 = real-time)')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000', help='API base URL')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        start_date = timezone.make_aware(start_date)
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        end_date = timezone.make_aware(end_date)
        # Set to end of day
        end_date = end_date.replace(hour=23, minute=59, second=59)
    
    # Parse turbine IDs
    turbine_ids = None
    if args.turbines:
        turbine_ids = [tid.strip() for tid in args.turbines.split(',')]
    
    # Create and run service
    service = RealtimeFeedService(
        api_url=args.api_url,
        interval_seconds=args.interval
    )
    
    try:
        service.run(
            start_date=start_date,
            end_date=end_date,
            turbine_ids=turbine_ids,
            speed_multiplier=args.speed_multiplier
        )
    except KeyboardInterrupt:
        print("\nService stopped by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
