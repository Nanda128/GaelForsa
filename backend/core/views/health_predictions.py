from rest_framework import viewsets
from rest_framework.pagination import PageNumberPagination
from drf_yasg.utils import swagger_auto_schema
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import ValidationError
from core.models.turbine import Turbine
from core.models.turbines import HealthPrediction, TurbineLog
from core.serializers.turbine_logs import HealthPredictionSerializer


class HealthPredictionPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'limit'
    max_page_size = 100


class HealthPredictionViewSet(viewsets.ModelViewSet):
    serializer_class = HealthPredictionSerializer
    pagination_class = HealthPredictionPagination

    def get_queryset(self):
        turbine_id = self.kwargs.get('id')
        get_object_or_404(Turbine, pk=turbine_id)
        return HealthPrediction.objects.filter(turbine_id=turbine_id).order_by('-timestamp')

    def perform_create(self, serializer):
        turbine_id = self.kwargs.get('id')
        turbine = get_object_or_404(Turbine, pk=turbine_id)
        log = serializer.validated_data.get('log')
        if log and log.turbine_id != turbine_id:
            raise ValidationError({'log': 'Log must belong to this turbine.'})
        serializer.save(turbine=turbine)

    @swagger_auto_schema(
        tags=['Health Predictions'],
        manual_parameters=[
            {'name': 'limit', 'in': 'query', 'description': 'Number of results per page', 'type': 'integer'},
        ]
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Health Predictions'])
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Health Predictions'])
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Health Predictions'])
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Health Predictions'])
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Health Predictions'])
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)
