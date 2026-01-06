import math
import random
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone
from core.models.turbine import Turbine
from core.models.turbines.turbine_log import TurbineLog
from core.models.turbines.on_turbine_reading import OnTurbineReading
from core.models.turbines.weather_reading import WeatherReading
from core.models.turbines.alert import Alert
from core.models.turbines.health_prediction import HealthPrediction
from core.models.turbines.maintenance_event import MaintenanceEvent


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance


def is_valid_turbine_position(lat, lon, existing_turbines, min_distance=1.5, max_distance=3.0):
    if not (-5.986782 <= lon <= -5.722422):
        return False
    if not (52.920988 <= lat <= 53.280342):
        return False
    
    if not existing_turbines:
        return True
    
    distances = [haversine_distance(lat, lon, t.latitude, t.longitude) for t in existing_turbines]
    min_dist = min(distances)
    
    if min_dist < min_distance:
        return False
    
    if min_dist > max_distance:
        return False
    
    return True


def generate_turbine_position(existing_turbines, max_attempts=5000, warning_callback=None):
    min_lon, min_lat, max_lon, max_lat = -5.986782, 52.920988, -5.722422, 53.280342
    
    if not existing_turbines:
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        lat = center_lat + random.uniform(-0.05, 0.05)
        lon = center_lon + random.uniform(-0.05, 0.05)
        return lat, lon
    
    for attempt in range(max_attempts):
        if attempt < len(existing_turbines) * 100:
            base_turbine = random.choice(existing_turbines)
            distance_km = random.uniform(1.5, 3.0)
            angle = random.uniform(0, 2 * math.pi)
            lat_offset = (distance_km / 111.0) * math.cos(angle)
            lon_offset = (distance_km / (111.0 * math.cos(math.radians(base_turbine.latitude)))) * math.sin(angle)
            lat = base_turbine.latitude + lat_offset
            lon = base_turbine.longitude + lon_offset
        else:
            lat = random.uniform(min_lat, max_lat)
            lon = random.uniform(min_lon, max_lon)
        
        if is_valid_turbine_position(lat, lon, existing_turbines):
            return lat, lon
    
    if warning_callback:
        warning_callback(
            f'Warning: Could not find ideal position after {max_attempts} attempts. '
            f'Using position with relaxed constraints.'
        )
    for _ in range(100):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        if existing_turbines:
            distances = [haversine_distance(lat, lon, t.latitude, t.longitude) for t in existing_turbines]
            if min(distances) >= 1.5:
                return lat, lon
    
    return random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon)


def generate_turbine_name(index):
    return f"Turbine-{index:03d}"


def generate_status():
    weights = [0.7, 0.2, 0.1]
    return random.choices(['green', 'yellow', 'red'], weights=weights)[0]


def generate_turbine_log(turbine, log_index, start_time, interval_minutes=10):
    timestamp = start_time + timedelta(minutes=log_index * interval_minutes)
    
    log = TurbineLog.objects.create(
        turbine=turbine,
        timestamp=timestamp,
        is_dummy_data=True
    )
    return log


def generate_on_turbine_reading(log, wind_speed=None, turbine_status='green', previous_reading=None):
    if wind_speed is None:
        wind_speed = random.uniform(5.0, 25.0)
    
    cut_in_speed = 3.5
    rated_speed = 12.5
    cut_out_speed = 25.0
    rated_power_kw = 5000.0
    
    if wind_speed < cut_in_speed:
        power_output = 0.0
        rotor_speed = random.uniform(0.0, 2.0)
        blade_pitch = random.uniform(85.0, 90.0)
    elif wind_speed > cut_out_speed:
        power_output = 0.0
        rotor_speed = random.uniform(0.0, 2.0)
        blade_pitch = random.uniform(85.0, 90.0)
    elif wind_speed < rated_speed:
        power_factor = ((wind_speed - cut_in_speed) / (rated_speed - cut_in_speed)) ** 3
        power_output = rated_power_kw * power_factor
        rotor_speed = 11.0 + (wind_speed - cut_in_speed) / (rated_speed - cut_in_speed) * 5.4
        blade_pitch = random.uniform(-1.0, 2.0)
    else:
        power_output = rated_power_kw
        rotor_speed = 16.0
        pitch_increase = (wind_speed - rated_speed) / (cut_out_speed - rated_speed)
        blade_pitch = 0.0 + pitch_increase * 27.0
    
    power_output += random.gauss(0, power_output * 0.03)
    power_output = max(0, min(power_output, rated_power_kw * 1.05))
    
    rotor_speed += random.gauss(0, 0.3)
    rotor_speed = max(0, min(rotor_speed, 18.5))
    
    if previous_reading:
        yaw_position = (previous_reading.yaw_position + random.gauss(0, 1.5)) % 360.0
    else:
        yaw_position = random.uniform(0.0, 360.0)
    
    yaw_error = random.gauss(0, 1.2)
    yaw_error = max(-15.0, min(yaw_error, 15.0))
    
    base_gen_temp = 55.0 + (power_output / rated_power_kw) * 25.0
    base_gearbox_temp = 50.0 + (power_output / rated_power_kw) * 20.0
    
    if turbine_status == 'green':
        gen_temp = base_gen_temp + random.gauss(0, 2.5)
        gearbox_temp = base_gearbox_temp + random.gauss(0, 2.0)
        vibration = random.gauss(1.5, 0.4)
        gearbox_pressure = random.gauss(3.5, 0.15)
    elif turbine_status == 'yellow':
        gen_temp = base_gen_temp + random.gauss(8.0, 3.5)
        gearbox_temp = base_gearbox_temp + random.gauss(10.0, 3.0)
        vibration = random.gauss(4.5, 1.2)
        gearbox_pressure = random.gauss(3.0, 0.25)
    else:
        gen_temp = base_gen_temp + random.gauss(20.0, 5.0)
        gearbox_temp = base_gearbox_temp + random.gauss(20.0, 4.0)
        vibration = random.gauss(8.0, 2.0)
        gearbox_pressure = random.gauss(2.5, 0.4)
    
    gen_temp = max(35.0, min(gen_temp, 100.0))
    gearbox_temp = max(40.0, min(gearbox_temp, 90.0))
    vibration = max(0.2, min(vibration, 15.0))
    gearbox_pressure = max(2.0, min(gearbox_pressure, 4.5))
    
    return OnTurbineReading.objects.create(
        log=log,
        rotor_speed=round(rotor_speed, 2),
        power_output=round(power_output, 2),
        generator_temperature=round(gen_temp, 2),
        gearbox_oil_temperature=round(gearbox_temp, 2),
        gearbox_oil_pressure=round(gearbox_pressure, 2),
        vibration_level=round(vibration, 2),
        blade_pitch_angle=round(blade_pitch, 2),
        yaw_position=round(yaw_position, 2),
        yaw_error=round(yaw_error, 2),
        is_dummy_data=True
    )


def generate_weather_reading(log, previous_wind=None, previous_wind_dir=None):
    if previous_wind is None:
        base_wind = random.uniform(8.0, 15.0)
    else:
        base_wind = previous_wind + random.gauss(0, 1.5)
        base_wind = max(2.0, min(base_wind, 28.0))
    
    wind_speed = base_wind + random.gauss(0, 0.8)
    wind_speed = max(0.0, min(wind_speed, 30.0))
    
    if previous_wind_dir is None:
        wind_direction = random.uniform(0.0, 360.0)
    else:
        wind_direction = (previous_wind_dir + random.gauss(0, 12.0)) % 360.0
    
    wave_height = 0.3 + (wind_speed / 25.0) * 4.0 + random.gauss(0, 0.25)
    wave_height = max(0.1, min(wave_height, 8.0))
    
    wave_period = 3.5 + (wave_height / 5.0) * 10.0 + random.gauss(0, 0.8)
    wave_period = max(2.5, min(wave_period, 18.0))
    
    water_temp = random.gauss(11.0, 2.5)
    water_temp = max(5.0, min(water_temp, 19.0))
    
    air_temp = water_temp + random.gauss(1.5, 2.5)
    air_temp = max(-5.0, min(air_temp, 25.0))
    
    air_humidity = random.gauss(82.0, 10.0)
    air_humidity = max(50.0, min(air_humidity, 100.0))
    
    lightning_strikes = random.choices([0, 1, 2], weights=[0.98, 0.018, 0.002])[0]
    
    return WeatherReading.objects.create(
        log=log,
        wind_speed=round(wind_speed, 2),
        wind_direction=round(wind_direction, 2),
        wave_height=round(wave_height, 2),
        wave_period=round(wave_period, 2),
        water_temperature=round(water_temp, 2),
        air_temperature=round(air_temp, 2),
        air_humidity=round(air_humidity, 2),
        lightning_strikes=lightning_strikes,
        is_dummy_data=True
    ), wind_speed, wind_direction


def generate_alert(turbine, reading, prediction=None, previous_reading=None):
    alerts = []
    
    if previous_reading and previous_reading.power_output and reading.power_output:
        power_drop_percent = ((previous_reading.power_output - reading.power_output) / previous_reading.power_output) * 100
        if power_drop_percent > 50 and previous_reading.power_output > 500:
            severity = 'critical' if power_drop_percent > 75 else 'warning'
            alerts.append({
                'severity': severity,
                'message': f'Significant power drop detected ({power_drop_percent:.1f}% decrease: {previous_reading.power_output:.0f} kW → {reading.power_output:.0f} kW)'
            })
        elif power_drop_percent > 30 and previous_reading.power_output > 1000:
            alerts.append({
                'severity': 'warning',
                'message': f'Power output decreased ({power_drop_percent:.1f}%: {previous_reading.power_output:.0f} kW → {reading.power_output:.0f} kW)'
            })
    
    if reading.vibration_level > 3.0:
        severity = 'critical' if reading.vibration_level > 6.0 else 'warning'
        alerts.append({
            'severity': severity,
            'message': f'High vibration detected ({reading.vibration_level:.2f} m/s²) - inspection required'
        })
    
    if reading.gearbox_oil_temperature > 70.0:
        severity = 'critical' if reading.gearbox_oil_temperature > 80.0 else 'warning'
        alerts.append({
            'severity': severity,
            'message': f'Gearbox oil temperature elevated ({reading.gearbox_oil_temperature:.2f}°C)'
        })
    
    if reading.generator_temperature > 80.0:
        severity = 'critical' if reading.generator_temperature > 90.0 else 'warning'
        alerts.append({
            'severity': severity,
            'message': f'Generator temperature high ({reading.generator_temperature:.2f}°C)'
        })
    
    if reading.gearbox_oil_pressure < 3.0:
        severity = 'critical' if reading.gearbox_oil_pressure < 2.5 else 'warning'
        alerts.append({
            'severity': severity,
            'message': f'Gearbox oil pressure low ({reading.gearbox_oil_pressure:.2f} bar)'
        })
    
    if abs(reading.yaw_error) > 5.0:
        severity = 'critical' if abs(reading.yaw_error) > 10.0 else 'warning'
        alerts.append({
            'severity': severity,
            'message': f'Yaw misalignment detected ({reading.yaw_error:.2f}°)'
        })
    
    if prediction and prediction.failure_probability > 0.5:
        alerts.append({
            'severity': 'critical',
            'message': f'High failure probability predicted ({prediction.failure_probability:.1%}) - preventive action recommended'
        })
    
    if not alerts and random.random() < 0.1:
        alerts.append({
            'severity': 'info',
            'message': random.choice([
                'Routine maintenance scheduled',
                'Performance within normal parameters',
                'Data collection system operating normally'
            ])
        })
    
    created_alerts = []
    for alert_data in alerts:
        created_at = reading.log.timestamp
        acknowledged = random.random() < 0.3
        acknowledged_at = None
        if acknowledged:
            acknowledged_at = created_at + timedelta(hours=random.uniform(1, 48))
        
        alert = Alert.objects.create(
            turbine=turbine,
            prediction=prediction,
            severity=alert_data['severity'],
            message=alert_data['message'],
            acknowledged=acknowledged,
            acknowledged_at=acknowledged_at,
            created_at=created_at,
            is_dummy_data=True
        )
        created_alerts.append(alert)
    
    return created_alerts


def generate_health_prediction(turbine, log, prediction_index, reading):
    base_health = 1.0
    base_failure = 0.0
    
    if reading.vibration_level > 3.0:
        base_health -= (reading.vibration_level - 3.0) * 0.08
        base_failure += (reading.vibration_level - 3.0) * 0.04
    
    if reading.gearbox_oil_temperature > 65.0:
        base_health -= (reading.gearbox_oil_temperature - 65.0) * 0.025
        base_failure += (reading.gearbox_oil_temperature - 65.0) * 0.025
    
    if reading.generator_temperature > 75.0:
        base_health -= (reading.generator_temperature - 75.0) * 0.025
        base_failure += (reading.generator_temperature - 75.0) * 0.025
    
    if reading.gearbox_oil_pressure < 3.2:
        base_health -= (3.2 - reading.gearbox_oil_pressure) * 0.12
        base_failure += (3.2 - reading.gearbox_oil_pressure) * 0.06
    
    if abs(reading.yaw_error) > 5.0:
        base_health -= abs(reading.yaw_error - 5.0) * 0.015
        base_failure += abs(reading.yaw_error - 5.0) * 0.01
    
    if turbine.status == 'green':
        base_health = max(0.75, min(base_health, 1.0))
        base_failure = max(0.0, min(base_failure, 0.15))
    elif turbine.status == 'yellow':
        base_health = max(0.45, min(base_health, 0.75))
        base_failure = max(0.15, min(base_failure, 0.45))
    else:
        base_health = max(0.15, min(base_health, 0.55))
        base_failure = max(0.4, min(base_failure, 0.9))
    
    health_score = base_health + random.gauss(0, 0.04)
    health_score = max(0.0, min(health_score, 1.0))
    
    failure_probability = base_failure + random.gauss(0, 0.04)
    failure_probability = max(0.0, min(failure_probability, 1.0))
    
    timestamp = log.timestamp
    
    predicted_failure_window_start = None
    predicted_failure_window_end = None
    if failure_probability > 0.3:
        days_ahead = max(1, int(30 * (1 - failure_probability)))
        predicted_failure_window_start = timestamp + timedelta(days=days_ahead)
        predicted_failure_window_end = predicted_failure_window_start + timedelta(days=random.uniform(3, 10))
    
    return HealthPrediction.objects.create(
        turbine=turbine,
        log=log,
        timestamp=timestamp,
        health_score=round(health_score, 3),
        failure_probability=round(failure_probability, 3),
        predicted_failure_window_start=predicted_failure_window_start,
        predicted_failure_window_end=predicted_failure_window_end,
        model_version=f"v{random.randint(1, 3)}.{random.randint(0, 9)}",
        features_used={
            'rotor_speed': True,
            'power_output': True,
            'vibration': True,
            'temperature': True,
            'wind_speed': True
        },
        is_dummy_data=True
    )


def generate_maintenance_event(turbine, event_index, logs=None):
    event_types = ['scheduled', 'repair', 'failure', 'inspection', 'replacement']
    weights = [0.4, 0.2, 0.1, 0.2, 0.1]
    event_type = random.choices(event_types, weights=weights)[0]
    
    days_ago = random.uniform(0, 90)
    start_time = timezone.now() - timedelta(days=days_ago, hours=random.uniform(0, 23))
    
    end_time = None
    if random.random() < 0.8:
        duration_hours = random.uniform(2, 48)
        end_time = start_time + timedelta(hours=duration_hours)
    
    log = None
    if logs and random.random() < 0.4:
        log = random.choice(logs)
    
    descriptions = {
        'scheduled': [
            'Routine annual maintenance',
            'Scheduled component replacement',
            'Preventive maintenance check',
            'Regular service inspection',
            'Scheduled oil change and filter replacement'
        ],
        'repair': [
            'Gearbox bearing replacement',
            'Blade pitch motor repair',
            'Generator cooling system repair',
            'Yaw system component replacement',
            'Electrical system repair'
        ],
        'failure': [
            'Emergency shutdown due to overheating',
            'Critical component failure',
            'Structural damage detected',
            'Electrical system failure',
            'Gearbox failure requiring replacement'
        ],
        'inspection': [
            'Visual inspection of blades',
            'Structural integrity assessment',
            'Safety system inspection',
            'Performance evaluation',
            'Environmental impact assessment'
        ],
        'replacement': [
            'Blade replacement',
            'Gearbox replacement',
            'Generator replacement',
            'Control system upgrade',
            'Complete nacelle replacement'
        ]
    }
    
    description = random.choice(descriptions[event_type])
    
    cost = None
    if random.random() < 0.6:
        if event_type == 'scheduled':
            cost = random.uniform(5000, 25000)
        elif event_type == 'repair':
            cost = random.uniform(10000, 50000)
        elif event_type == 'failure':
            cost = random.uniform(50000, 200000)
        elif event_type == 'replacement':
            cost = random.uniform(100000, 500000)
        else:
            cost = random.uniform(2000, 10000)
    
    parts_replaced = ""
    if event_type in ['repair', 'replacement']:
        parts = [
            'Gearbox bearings',
            'Blade pitch motors',
            'Generator components',
            'Yaw system parts',
            'Electrical connectors',
            'Cooling system components'
        ]
        if random.random() < 0.7:
            parts_replaced = random.choice(parts)
    
    return MaintenanceEvent.objects.create(
        turbine=turbine,
        log=log,
        event_type=event_type,
        start_time=start_time,
        end_time=end_time,
        description=description,
        parts_replaced=parts_replaced,
        cost=cost,
        is_dummy_data=True
    )


class Command(BaseCommand):
    help = 'Generate test data for wind turbines with realistic values'

    def add_arguments(self, parser):
        parser.add_argument(
            '--turbines',
            type=int,
            default=5,
            help='Number of turbines to generate (default: 5)',
        )
        parser.add_argument(
            '--entries',
            type=int,
            default=5,
            help='Number of related entries per turbine (default: 5)',
        )

    def handle(self, *args, **options):
        num_turbines = options['turbines']
        num_entries = options['entries']
        
        self.stdout.write(self.style.SUCCESS(f'Generating {num_turbines} turbines with {num_entries} entries each...'))
        
        existing_turbines = []
        
        for i in range(1, num_turbines + 1):
            self.stdout.write(f'Creating turbine {i}/{num_turbines}...')
            
            def warn(msg):
                self.stdout.write(self.style.WARNING(msg))
            lat, lon = generate_turbine_position(existing_turbines, warning_callback=warn)
            
            turbine = Turbine.objects.create(
                name=generate_turbine_name(i),
                latitude=lat,
                longitude=lon,
                status=generate_status(),
                is_dummy_data=True
            )
            existing_turbines.append(turbine)
            
            self.stdout.write(f'  Created turbine: {turbine.name} at ({lat:.6f}, {lon:.6f})')
            
            start_time = timezone.now() - timedelta(days=30)
            interval_minutes = 10
            
            logs = []
            readings = []
            alerts_count = 0
            previous_wind = None
            previous_wind_dir = None
            previous_reading = None
            
            for j in range(num_entries):
                log = generate_turbine_log(turbine, j, start_time, interval_minutes)
                logs.append(log)
                
                weather, wind_speed, wind_dir = generate_weather_reading(log, previous_wind, previous_wind_dir)
                previous_wind = wind_speed
                previous_wind_dir = wind_dir
                
                reading = generate_on_turbine_reading(log, wind_speed, turbine.status, previous_reading)
                readings.append(reading)
                
                prediction = generate_health_prediction(turbine, log, j, reading)
                
                created_alerts = generate_alert(turbine, reading, prediction, previous_reading)
                alerts_count += len(created_alerts)
                
                previous_reading = reading
            
            for j in range(num_entries):
                generate_maintenance_event(turbine, j, logs)
            
            self.stdout.write(self.style.SUCCESS(f'  ✓ Created {num_entries} logs, readings, predictions, {alerts_count} alerts, and {num_entries} maintenance events'))
        
        self.stdout.write(self.style.SUCCESS(
            f'\nSuccessfully created {num_turbines} turbines with {num_entries} entries each!'
        ))
        self.stdout.write(self.style.SUCCESS(
            f'Total records created:'
        ))
        self.stdout.write(f'  - Turbines: {num_turbines}')
        self.stdout.write(f'  - Turbine Logs: {num_turbines * num_entries} (every 10 minutes over 30 days)')
        self.stdout.write(f'  - On-Turbine Readings: {num_turbines * num_entries}')
        self.stdout.write(f'  - Weather Readings: {num_turbines * num_entries}')
        self.stdout.write(f'  - Health Predictions: {num_turbines * num_entries}')
        self.stdout.write(f'  - Alerts: Variable (generated based on sensor readings)')
        self.stdout.write(f'  - Maintenance Events: {num_turbines * num_entries}')
