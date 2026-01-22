from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from rest_framework.routers import DefaultRouter
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from .views import api_root_message, health_status
from core.views import (
    TurbineViewSet, TurbineLogViewSet, MaintenanceEventViewSet,
    HealthPredictionViewSet, AlertViewSet
)
from core.views.analytics import (
    farm_stats, farm_power_summary, farm_health_trends,
    farm_grid_impact, farm_economics
)
from core.views.realtime import receive_realtime_reading, trigger_prediction

router = DefaultRouter()
router.register(r'turbines', TurbineViewSet, basename='turbine')

nested_urlpatterns = [
    path('api/v1/turbines/<uuid:id>/logs/', TurbineLogViewSet.as_view({'get': 'list', 'post': 'create'}), name='turbine-log-list'),
    path('api/v1/turbines/<uuid:id>/logs/<uuid:pk>/', TurbineLogViewSet.as_view({'get': 'retrieve', 'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'}), name='turbine-log-detail'),
    path('api/v1/turbines/<uuid:id>/maintenance-events/', MaintenanceEventViewSet.as_view({'get': 'list', 'post': 'create'}), name='maintenance-event-list'),
    path('api/v1/turbines/<uuid:id>/maintenance-events/<uuid:pk>/', MaintenanceEventViewSet.as_view({'get': 'retrieve', 'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'}), name='maintenance-event-detail'),
    path('api/v1/turbines/<uuid:id>/health-predictions/', HealthPredictionViewSet.as_view({'get': 'list', 'post': 'create'}), name='health-prediction-list'),
    path('api/v1/turbines/<uuid:id>/health-predictions/<uuid:pk>/', HealthPredictionViewSet.as_view({'get': 'retrieve', 'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'}), name='health-prediction-detail'),
    path('api/v1/turbines/<uuid:id>/alerts/', AlertViewSet.as_view({'get': 'list', 'post': 'create'}), name='alert-list'),
    path('api/v1/turbines/<uuid:id>/alerts/<uuid:pk>/', AlertViewSet.as_view({'get': 'retrieve', 'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'}), name='alert-detail'),
]

analytics_urlpatterns = [
    path('api/v1/farm/stats/', farm_stats, name='farm-stats'),
    path('api/v1/farm/power-summary/', farm_power_summary, name='farm-power-summary'),
    path('api/v1/farm/health-trends/', farm_health_trends, name='farm-health-trends'),
    path('api/v1/farm/grid-impact/', farm_grid_impact, name='farm-grid-impact'),
    path('api/v1/farm/economics/', farm_economics, name='farm-economics'),
]

realtime_urlpatterns = [
    path('api/v1/realtime/reading/', receive_realtime_reading, name='receive-realtime-reading'),
    path('api/v1/realtime/logs/<uuid:log_id>/predict/', trigger_prediction, name='trigger-prediction'),
]

schema_view = get_schema_view(
    openapi.Info(
        title="GaelFórsa Wind Turbine Monitoring API",
        default_version='v1',
        description="""
        RESTful API for the GaelFórsa predictive maintenance platform for offshore wind turbines.
        
        This API provides endpoints for:
        - Turbine management and monitoring
        - Health predictions and failure risk analysis
        - Alert management
        - Maintenance event tracking
        - Power output analytics
        - Grid impact metrics
        - Economic impact calculations
        
        All endpoints support filtering, pagination, and comprehensive error handling.
        """,
        contact=openapi.Contact(email="info@gaelforsa.ie"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
    patterns=[
        path('api/v1/', include(router.urls)),
    ] + nested_urlpatterns + analytics_urlpatterns,
)

urlpatterns = [
    path('', api_root_message),
    path('healthz', health_status),
    path('admin/', admin.site.urls),
    path('api/v1/', include(router.urls)),
    path('map/', TemplateView.as_view(template_name='map.html'), name='map'),
] + nested_urlpatterns + analytics_urlpatterns + realtime_urlpatterns + [
    re_path(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
