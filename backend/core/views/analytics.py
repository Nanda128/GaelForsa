from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Avg, Count, Sum, Max, Min, Q
from django.utils import timezone
from datetime import timedelta
import math
from core.models.turbine import Turbine
from core.models.turbines import HealthPrediction, Alert, TurbineLog, OnTurbineReading, MaintenanceEvent
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi


def calculate_energy_from_readings(power_readings):
    # moved to trapezoidal integration instead of instantaneous reading
    if not power_readings.exists():
        return 0
    
    readings = list(power_readings.select_related('log').order_by('log__timestamp'))
    if len(readings) < 2:
        return 0
    
    total_energy_kwh = 0
    
    for i in range(1, len(readings)):
        prev_reading = readings[i-1]
        curr_reading = readings[i]
        
        if prev_reading.power_output is None or curr_reading.power_output is None:
            continue
        
        time_diff_seconds = (curr_reading.log.timestamp - prev_reading.log.timestamp).total_seconds()
        time_diff_hours = time_diff_seconds / 3600
        
        if time_diff_hours <= 0:
            continue
        
        avg_power_kw = (prev_reading.power_output + curr_reading.power_output) / 2
        energy_kwh = avg_power_kw * time_diff_hours
        total_energy_kwh += energy_kwh
    
    return total_energy_kwh


def calculate_downtime_hours(maintenance_events):
    total_hours = 0
    now = timezone.now()
    
    for event in maintenance_events:
        if event.end_time:
            duration = (event.end_time - event.start_time).total_seconds() / 3600
        else:
            duration = (now - event.start_time).total_seconds() / 3600
        
        total_hours += duration
    
    return total_hours


@swagger_auto_schema(
    method='get',
    operation_description='Get overall farm statistics including turbine counts, health metrics, and power output',
    tags=['Analytics']
)
@api_view(['GET'])
def farm_stats(request):
    turbines = Turbine.objects.all()
    total = turbines.count()
    
    status_counts = {
        'green': turbines.filter(status='green').count(),
        'yellow': turbines.filter(status='yellow').count(),
        'red': turbines.filter(status='red').count()
    }
    
    health_predictions = HealthPrediction.objects.filter(
        turbine__in=turbines
    ).order_by('-timestamp')
    
    # Could use annotate with Subquery but this is clearer
    latest_predictions = {}
    for turbine in turbines:
        latest = health_predictions.filter(turbine=turbine).first()
        # latest = health_predictions.filter(turbine=turbine).order_by('-timestamp').first()
        if latest:
            latest_predictions[turbine.id] = {
                'health_score': latest.health_score,
                'failure_probability': latest.failure_probability
            }
    
    avg_health = 0
    avg_failure_risk = 0
    if latest_predictions:
        # avg_health = statistics.mean([p['health_score'] for p in latest_predictions.values()])
        avg_health = sum(p['health_score'] for p in latest_predictions.values()) / len(latest_predictions)
        avg_failure_risk = sum(p['failure_probability'] for p in latest_predictions.values()) / len(latest_predictions)
    
    active_alerts = Alert.objects.filter(
        turbine__in=turbines,
        acknowledged=False
    ).count()
    
    critical_alerts = Alert.objects.filter(
        turbine__in=turbines,
        acknowledged=False,
        severity='critical'
    ).count()
    
    recent_logs = TurbineLog.objects.filter(
        turbine__in=turbines,
        timestamp__gte=timezone.now() - timedelta(days=7)
    )
    
    power_readings = OnTurbineReading.objects.filter(
        log__in=recent_logs,
        power_output__isnull=False
    )
    
    avg_power = power_readings.aggregate(Avg('power_output'))['power_output__avg'] or 0
    max_power = power_readings.aggregate(Max('power_output'))['power_output__max'] or 0
    total_energy_kwh = calculate_energy_from_readings(power_readings)
    
    return Response({
        'turbines': {
            'total': total,
            'by_status': status_counts,
            'health_percentage': (status_counts['green'] / total * 100) if total > 0 else 0
        },
        'health_metrics': {
            'average_health_score': round(avg_health, 3),
            'average_failure_probability': round(avg_failure_risk, 3),
            'turbines_with_predictions': len(latest_predictions)
        },
        'alerts': {
            'active': active_alerts,
            'critical': critical_alerts,
            'acknowledged': Alert.objects.filter(turbine__in=turbines, acknowledged=True).count()
        },
        'power_output': {
            'total_energy_kwh': round(total_energy_kwh, 2),
            'average_kw': round(avg_power, 2),
            'max_kw': round(max_power, 2),
            'period_days': 7
        },
        'timestamp': timezone.now().isoformat()
    })


@swagger_auto_schema(
    method='get',
    operation_description='Get power output summary across all turbines',
    tags=['Analytics'],
    manual_parameters=[
        openapi.Parameter('days', openapi.IN_QUERY, description='Number of days to look back', type=openapi.TYPE_INTEGER, default=7)
    ]
)
@api_view(['GET'])
def farm_power_summary(request):
    days = int(request.query_params.get('days', 7))
    cutoff_date = timezone.now() - timedelta(days=days)
    
    recent_logs = TurbineLog.objects.filter(timestamp__gte=cutoff_date)
    power_readings = OnTurbineReading.objects.filter(
        log__in=recent_logs,
        power_output__isnull=False
    )
    
    avg_power = power_readings.aggregate(Avg('power_output'))['power_output__avg'] or 0
    max_power = power_readings.aggregate(Max('power_output'))['power_output__max'] or 0
    min_power = power_readings.aggregate(Min('power_output'))['power_output__min'] or 0
    count = power_readings.count()
    
    turbines = Turbine.objects.all()
    turbine_power = []
    for turbine in turbines:
        turbine_logs = recent_logs.filter(turbine=turbine)
        turbine_readings = power_readings.filter(log__in=turbine_logs)
        if turbine_readings.exists():
            turbine_energy_kwh = calculate_energy_from_readings(turbine_readings)
            turbine_power.append({
                'turbine_id': str(turbine.id),
                'turbine_name': turbine.name,
                'energy_kwh': round(turbine_energy_kwh, 2),
                'average_kw': round(turbine_readings.aggregate(Avg('power_output'))['power_output__avg'] or 0, 2),
                'max_kw': round(turbine_readings.aggregate(Max('power_output'))['power_output__max'] or 0, 2),
                'readings_count': turbine_readings.count()
            })
    
    total_energy_kwh = calculate_energy_from_readings(power_readings)
    
    return Response({
        'period_days': days,
        'summary': {
            'total_energy_kwh': round(total_energy_kwh, 2),
            'average_kw': round(avg_power, 2),
            'max_kw': round(max_power, 2),
            'min_kw': round(min_power, 2),
            'total_readings': count
        },
        'by_turbine': turbine_power,
        'timestamp': timezone.now().isoformat()
    })


@swagger_auto_schema(
    method='get',
    operation_description='Get health trends over time',
    tags=['Analytics'],
    manual_parameters=[
        openapi.Parameter('days', openapi.IN_QUERY, description='Number of days to look back', type=openapi.TYPE_INTEGER, default=30)
    ]
)
@api_view(['GET'])
def farm_health_trends(request):
    days = int(request.query_params.get('days', 30))
    cutoff_date = timezone.now() - timedelta(days=days)
    
    predictions = HealthPrediction.objects.filter(
        timestamp__gte=cutoff_date
    ).order_by('timestamp')
    
    daily_stats = {}
    for pred in predictions:
        date_key = pred.timestamp.date().isoformat()
        if date_key not in daily_stats:
            daily_stats[date_key] = {
                'date': date_key,
                'health_scores': [],
                'failure_probabilities': [],
                'count': 0
            }
        daily_stats[date_key]['health_scores'].append(pred.health_score)
        daily_stats[date_key]['failure_probabilities'].append(pred.failure_probability)
        daily_stats[date_key]['count'] += 1
    
    trends = []
    for date_key in sorted(daily_stats.keys()):
        stats = daily_stats[date_key]
        trends.append({
            'date': date_key,
            'average_health_score': round(sum(stats['health_scores']) / len(stats['health_scores']), 3),
            'average_failure_probability': round(sum(stats['failure_probabilities']) / len(stats['failure_probabilities']), 3),
            'predictions_count': stats['count']
        })
    
    return Response({
        'period_days': days,
        'trends': trends,
        'timestamp': timezone.now().isoformat()
    })


@swagger_auto_schema(
    method='get',
    operation_description='Get grid impact metrics including reactive power and system availability',
    tags=['Analytics']
)
@api_view(['GET'])
def farm_grid_impact(request):
    # TODO: Add actual values for power factor, reactive power, KVA etc.
    turbines = Turbine.objects.all()
    total_turbines = turbines.count()
    
    operational_turbines = turbines.filter(status='green').count()
    availability = (operational_turbines / total_turbines * 100) if total_turbines > 0 else 0
    
    recent_logs = TurbineLog.objects.filter(
        timestamp__gte=timezone.now() - timedelta(days=7)
    )
    
    power_readings = OnTurbineReading.objects.filter(
        log__in=recent_logs,
        power_output__isnull=False
    )
    
    total_active_power = power_readings.aggregate(Sum('power_output'))['power_output__sum'] or 0
    
    reactive_power_readings = OnTurbineReading.objects.filter(
        log__in=recent_logs
    ).exclude(power_output__isnull=True)
    
    power_factor = 0.95
    estimated_reactive_power = total_active_power * math.tan(math.acos(power_factor))
    
    downtime_turbines = turbines.filter(status__in=['yellow', 'red']).count()
    downtime_percentage = (downtime_turbines / total_turbines * 100) if total_turbines > 0 else 0
    
    return Response({
        'system_availability': {
            'percentage': round(availability, 2),
            'operational_turbines': operational_turbines,
            'total_turbines': total_turbines,
            'downtime_percentage': round(downtime_percentage, 2)
        },
        'power_metrics': {
            'total_active_power_kw': round(total_active_power, 2),
            'estimated_reactive_power_kvar': round(estimated_reactive_power, 2),
            'apparent_power_kva': round((total_active_power ** 2 + estimated_reactive_power ** 2) ** 0.5, 2)
        },
        'grid_stability': {
            'healthy_turbines_ratio': round(availability / 100, 3),
            'critical_alerts_count': Alert.objects.filter(
                turbine__in=turbines,
                acknowledged=False,
                severity='critical'
            ).count()
        },
        'timestamp': timezone.now().isoformat()
    })


@swagger_auto_schema(
    method='get',
    operation_description='Get economic impact metrics including cost savings and ROI',
    tags=['Analytics']
)
@api_view(['GET'])
def farm_economics(request):
    # TODO: Add actual values for energy price, maintenance cost, downtime cost, etc.
    turbines = Turbine.objects.all()
    
    recent_logs = TurbineLog.objects.filter(
        timestamp__gte=timezone.now() - timedelta(days=30)
    )
    
    power_readings = OnTurbineReading.objects.filter(
        log__in=recent_logs,
        power_output__isnull=False
    )
    
    total_energy_kwh = calculate_energy_from_readings(power_readings)
    
    energy_price_per_kwh = 0.12
    total_revenue = total_energy_kwh * energy_price_per_kwh
    
    maintenance_events = MaintenanceEvent.objects.filter(
        turbine__in=turbines,
        start_time__gte=timezone.now() - timedelta(days=30)
    )
    
    total_maintenance_cost = sum(float(event.cost or 0) for event in maintenance_events)
    
    downtime_hours = calculate_downtime_hours(maintenance_events)
    estimated_downtime_cost_per_turbine_per_day = 5000
    downtime_days = downtime_hours / 24
    downtime_cost = downtime_days * estimated_downtime_cost_per_turbine_per_day
    
    active_alerts = Alert.objects.filter(
        turbine__in=turbines,
        acknowledged=False
    )
    
    critical_alerts = active_alerts.filter(severity='critical')
    estimated_failure_cost = 50000
    estimated_preventive_maintenance_cost = 10000
    
    preventive_savings = 0
    for alert in critical_alerts:
        latest_prediction = HealthPrediction.objects.filter(
            turbine=alert.turbine
        ).order_by('-timestamp').first()
        
        if latest_prediction and latest_prediction.failure_probability > 0.5:
            failure_risk = latest_prediction.failure_probability
            expected_failure_cost = failure_risk * estimated_failure_cost
            preventive_savings += max(0, expected_failure_cost - estimated_preventive_maintenance_cost)
    
    return Response({
        'revenue': {
            'total_energy_kwh': round(total_energy_kwh, 2),
            'energy_price_per_kwh': energy_price_per_kwh,
            'total_revenue_eur': round(total_revenue, 2),
            'period_days': 30
        },
        'costs': {
            'maintenance_cost_eur': round(total_maintenance_cost, 2),
            'downtime_cost_eur': round(downtime_cost, 2),
            'total_cost_eur': round(total_maintenance_cost + downtime_cost, 2)
        },
        'savings': {
            'estimated_preventive_savings_eur': round(preventive_savings, 2),
            'net_benefit_eur': round(total_revenue - total_maintenance_cost - downtime_cost + preventive_savings, 2)
        },
        'metrics': {
            'roi_percentage': round(((total_revenue - total_maintenance_cost - downtime_cost) / (total_maintenance_cost + downtime_cost) * 100) if (total_maintenance_cost + downtime_cost) > 0 else 0, 2),
            'cost_per_mwh': round((total_maintenance_cost + downtime_cost) / (total_energy_kwh / 1000), 2) if total_energy_kwh > 0 else 0
        },
        'timestamp': timezone.now().isoformat()
    })

