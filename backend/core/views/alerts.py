from rest_framework import viewsets
from drf_yasg.utils import swagger_auto_schema
from django.shortcuts import get_object_or_404
from core.models.turbine import Turbine
from core.models.turbines import Alert
from core.serializers.turbine_logs import AlertSerializer


class AlertViewSet(viewsets.ModelViewSet):
    serializer_class = AlertSerializer

    def get_queryset(self):
        turbine_id = self.kwargs.get('id')
        get_object_or_404(Turbine, pk=turbine_id)
        return Alert.objects.filter(turbine_id=turbine_id)

    def perform_create(self, serializer):
        turbine_id = self.kwargs.get('id')
        turbine = get_object_or_404(Turbine, pk=turbine_id)
        serializer.save(turbine=turbine)

    @swagger_auto_schema(tags=['Alerts'])
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

