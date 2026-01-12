import os
import csv
from datetime import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone
from core.models import Turbine, TurbineLog, OnTurbineReading


class Command(BaseCommand):
    help = 'Load Kelmarsh SCADA data from 2016-2018 into the database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--data-dir',
            type=str,
            default='testGaelForsa/ai_model/5841834',
            help='Path to the data directory containing SCADA files'
        )

    def handle(self, *args, **options):
        data_dir = options['data_dir']
        years = [2016, 2017, 2018]
        turbines = range(1, 7)  # 1 to 6

        # Mapping from CSV column names to OnTurbineReading field names
        column_mapping = {
            'Wind speed (m/s)': 'wind_speed',
            'Wind direction (°)': 'wind_direction',
            'Power (kW)': 'power_output',
            'Rotor speed (RPM)': 'rotor_speed_rpm',
            'Generator RPM (RPM)': 'generator_rpm',
            'Gearbox speed (RPM)': 'gearbox_speed',
            'Blade angle (pitch position) A (°)': 'blade_pitch_angle',
            'Voltage L1 / U (V)': 'voltage_l1',
            'Voltage L2 / V (V)': 'voltage_l2',
            'Voltage L3 / W (V)': 'voltage_l3',
            'Current L1 / U (A)': 'current_l1',
            'Current L2 / V (A)': 'current_l2',
            'Current L3 / W (A)': 'current_l3',
            'Front bearing temperature (°C)': 'front_bearing_temperature',
            'Rear bearing temperature (°C)': 'rear_bearing_temperature',
            'Nacelle temperature (°C)': 'nacelle_temperature',
            'Transformer temperature (°C)': 'transformer_temperature',
            'Gear oil temperature (°C)': 'gear_oil_temperature',
            'Power factor (cosphi)': 'power_factor',
            'Reactive power (kvar)': 'reactive_power',
            'Yaw bearing angle (°)': 'yaw_position',
        }