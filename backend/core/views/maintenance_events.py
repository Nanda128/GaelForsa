from rest_framework import viewsets
from rest_framework.pagination import PageNumberPagination
from django.utils import timezone
from datetime import timedelta
from drf_yasg.utils import swagger_auto_schema
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import ValidationError
from core.models.turbine import Turbine
from core.models.turbines import MaintenanceEvent, TurbineLog
from core.serializers.turbine_logs import MaintenanceEventSerializer


class MaintenanceEventPagination(PageNumberPagination):
    page_size = 50
    page_size_query_param = 'limit'
    max_page_size = 500


class MaintenanceEventViewSet(viewsets.ModelViewSet):
    serializer_class = MaintenanceEventSerializer
    pagination_class = MaintenanceEventPagination

    def get_queryset(self):
        turbine_id = self.kwargs.get('id')
        get_object_or_404(Turbine, pk=turbine_id)
        queryset = MaintenanceEvent.objects.filter(turbine_id=turbine_id)
        
        event_type = self.request.query_params.get('event_type', None)
        if event_type:
            queryset = queryset.filter(event_type=event_type)
        
        days = self.request.query_params.get('days', None)
        if days:
            try:
                days_int = int(days)
                cutoff_date = timezone.now() - timedelta(days=days_int)
                queryset = queryset.filter(start_time__gte=cutoff_date)
            except ValueError:
                pass
        
        return queryset.order_by('-start_time')

    def perform_create(self, serializer):
        turbine_id = self.kwargs.get('id')
        turbine = get_object_or_404(Turbine, pk=turbine_id)
        log = serializer.validated_data.get('log')
        if log and log.turbine_id != turbine_id:
            raise ValidationError({'log': 'Log must belong to this turbine.'})
        serializer.save(turbine=turbine)

    @swagger_auto_schema(
        tags=['Maintenance Events'],
        manual_parameters=[
            {'name': 'event_type', 'in': 'query', 'description': 'Filter by event type', 'type': 'string'},
            {'name': 'days', 'in': 'query', 'description': 'Filter events from last N days', 'type': 'integer'},
            {'name': 'limit', 'in': 'query', 'description': 'Number of results per page', 'type': 'integer'},
        ]
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Maintenance Events'])
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Maintenance Events'])
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Maintenance Events'])
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Maintenance Events'])
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Maintenance Events'])
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)
