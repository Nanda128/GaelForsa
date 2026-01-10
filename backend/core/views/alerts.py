from rest_framework import viewsets
from rest_framework.pagination import PageNumberPagination
from drf_yasg.utils import swagger_auto_schema
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import ValidationError
from core.models.turbine import Turbine
from core.models.turbines import Alert, HealthPrediction
from core.serializers.turbine_logs import AlertSerializer


class AlertPagination(PageNumberPagination):
    page_size = 50
    page_size_query_param = 'limit'
    max_page_size = 500


class AlertViewSet(viewsets.ModelViewSet):
    serializer_class = AlertSerializer
    pagination_class = AlertPagination

    def get_queryset(self):
        turbine_id = self.kwargs.get('id')
        get_object_or_404(Turbine, pk=turbine_id)
        queryset = Alert.objects.filter(turbine_id=turbine_id)
        
        acknowledged = self.request.query_params.get('acknowledged', None)
        if acknowledged is not None:
            queryset = queryset.filter(acknowledged=acknowledged.lower() == 'true')
        
        severity = self.request.query_params.get('severity', None)
        if severity:
            queryset = queryset.filter(severity=severity)
        
        return queryset.order_by('-created_at')

    def perform_create(self, serializer):
        turbine_id = self.kwargs.get('id')
        turbine = get_object_or_404(Turbine, pk=turbine_id)
        prediction = serializer.validated_data.get('prediction')
        if prediction and prediction.turbine_id != turbine_id:
            raise ValidationError({'prediction': 'Prediction must belong to this turbine.'})
        serializer.save(turbine=turbine)

    @swagger_auto_schema(
        tags=['Alerts'],
        manual_parameters=[
            {'name': 'acknowledged', 'in': 'query', 'description': 'Filter by acknowledged status (true/false)', 'type': 'boolean'},
            {'name': 'severity', 'in': 'query', 'description': 'Filter by severity (info, warning, critical)', 'type': 'string'},
            {'name': 'limit', 'in': 'query', 'description': 'Number of results per page', 'type': 'integer'},
        ]
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Alerts'])
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Alerts'])
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Alerts'])
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Alerts'])
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Alerts'])
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)
