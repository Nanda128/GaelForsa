import csv
import json
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.db.models import Q, Avg, Count
from core.models import HealthPrediction, MaintenanceEvent, Turbine


class Command(BaseCommand):
    help = 'Validate ML model predictions against historical data using maintenance events as ground truth'

    def add_arguments(self, parser):
        parser.add_argument(
            '--start-date',
            type=str,
            help='Start date for validation (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--end-date',
            type=str,
            help='End date for validation (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--turbines',
            type=str,
            help='Comma-separated list of turbine numbers to filter (e.g., "1,2,3")'
        )
        parser.add_argument(
            '--fault-threshold',
            type=float,
            default=0.5,
            help='Threshold for failure probability to consider as predicted fault (default: 0.5)'
        )
        parser.add_argument(
            '--output-format',
            type=str,
            choices=['console', 'csv', 'json'],
            default='console',
            help='Output format for validation report (default: console)'
        )
        parser.add_argument(
            '--output-file',
            type=str,
            help='Output file path for CSV/JSON formats'
        )