from django.contrib import admin
from core.models.turbine import Turbine
from core.models.turbines import (
    TurbineLog, OnTurbineReading, WeatherReading,
    MaintenanceEvent, HealthPrediction, Alert
)

admin.site.register(Turbine)
admin.site.register(TurbineLog)
admin.site.register(OnTurbineReading)
admin.site.register(WeatherReading)
admin.site.register(MaintenanceEvent)
admin.site.register(HealthPrediction)
admin.site.register(Alert)

