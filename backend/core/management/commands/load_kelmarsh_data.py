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

        # Ensure Turbine instances exist
        for turbine_num in turbines:
            turbine_name = f"Kelmarsh {turbine_num}"
            Turbine.objects.get_or_create(
                name=turbine_name,
                defaults={'latitude': 0.0, 'longitude': 0.0}
            )

        total_files = len(years) * len(turbines)
        processed_files = 0

        for year in years:
            scada_dir = f"Kelmarsh_SCADA_{year}_308{year - 2014}"
            scada_path = os.path.join(data_dir, scada_dir)

            if not os.path.exists(scada_path):
                self.stdout.write(self.style.WARNING(f"Directory {scada_path} not found, skipping year {year}"))
                continue

            for turbine_num in turbines:
                file_pattern = f"Turbine_Data_Kelmarsh_{turbine_num}_{year}-01-01_-_{year+1}-01-01_228.csv"
                file_path = os.path.join(scada_path, file_pattern)

                if not os.path.exists(file_path):
                    # Try alternative naming
                    file_pattern_alt = f"Turbine_Data_Kelmarsh_{turbine_num}_{year}-01-03_-_{year+1}-01-01_228.csv"
                    file_path = os.path.join(scada_path, file_pattern_alt)

                if not os.path.exists(file_path):
                    self.stdout.write(self.style.WARNING(f"File {file_path} not found, skipping"))
                    continue

                self.stdout.write(f"Processing {file_path}")
                turbine = Turbine.objects.get(name=f"Kelmarsh {turbine_num}")

                with open(file_path, 'r', encoding='utf-8') as csvfile:
                    # Skip header lines until data starts
                    lines = csvfile.readlines()
                    header_line = None
                    data_start = 0
                    for i, line in enumerate(lines):
                        if 'Date and time' in line:
                            header_line = line.strip()
                            data_start = i + 1
                            break

                    if not header_line:
                        self.stdout.write(self.style.ERROR(f"No header found in {file_path}"))
                        continue

                    headers = header_line.split(',')
                    reader = csv.reader(lines[data_start:])

                    row_count = 0
                    for row in reader:
                        if not row or row[0].strip() == '':
                            continue

                        try:
                            timestamp_str = row[0].strip()
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            timestamp = timezone.make_aware(timestamp)

                            # Create TurbineLog
                            turbine_log, created = TurbineLog.objects.get_or_create(
                                turbine=turbine,
                                timestamp=timestamp
                            )

                            # Prepare OnTurbineReading data
                            reading_data = {}
                            for csv_col, model_field in column_mapping.items():
                                if csv_col in headers:
                                    col_index = headers.index(csv_col)
                                    if col_index < len(row):
                                        value = row[col_index].strip()
                                        if value and value != 'NaN':
                                            try:
                                                reading_data[model_field] = float(value)
                                            except ValueError:
                                                pass  # Skip invalid values

                            # Create OnTurbineReading
                            OnTurbineReading.objects.create(log=turbine_log, **reading_data)
                            row_count += 1

                        except Exception as e:
                            self.stdout.write(self.style.ERROR(f"Error processing row: {e}"))

                    self.stdout.write(f"Processed {row_count} rows for turbine {turbine_num}, year {year}")

                processed_files += 1
                self.stdout.write(f"Progress: {processed_files}/{total_files} files processed")

        self.stdout.write(self.style.SUCCESS("Data loading completed"))