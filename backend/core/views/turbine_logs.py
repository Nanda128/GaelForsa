from rest_framework import viewsets
from rest_framework.pagination import PageNumberPagination
from django.utils import timezone
from datetime import timedelta
from drf_yasg.utils import swagger_auto_schema
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import ValidationError
from core.models.turbine import Turbine
from core.models.turbines import TurbineLog
from core.serializers.turbine_logs import TurbineLogSerializer


class TurbineLogPagination(PageNumberPagination):
    page_size = 100
    page_size_query_param = 'limit'
    max_page_size = 1000


class TurbineLogViewSet(viewsets.ModelViewSet):
    serializer_class = TurbineLogSerializer
    pagination_class = TurbineLogPagination

    def get_queryset(self):
        turbine_id = self.kwargs.get('id')
        get_object_or_404(Turbine, pk=turbine_id)
        queryset = TurbineLog.objects.filter(turbine_id=turbine_id)
        
        days = self.request.query_params.get('days', None)
        if days:
            try:
                days_int = int(days)
                cutoff_date = timezone.now() - timedelta(days=days_int)
                queryset = queryset.filter(timestamp__gte=cutoff_date)
            except ValueError:
                pass
        
        return queryset.order_by('-timestamp')

    def perform_create(self, serializer):
        turbine_id = self.kwargs.get('id')
        turbine = get_object_or_404(Turbine, pk=turbine_id)
        serializer.save(turbine=turbine)

    @swagger_auto_schema(
        tags=['Turbine Logs'],
        manual_parameters=[
            {'name': 'days', 'in': 'query', 'description': 'Filter logs from last N days', 'type': 'integer'},
            {'name': 'limit', 'in': 'query', 'description': 'Number of results per page', 'type': 'integer'},
        ]
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Turbine Logs'])
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Turbine Logs'])
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Turbine Logs'])
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Turbine Logs'])
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Turbine Logs'])
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)
