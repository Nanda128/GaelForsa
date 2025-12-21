from rest_framework import viewsets
from drf_yasg.utils import swagger_auto_schema
from django.shortcuts import get_object_or_404
from core.models.turbine import Turbine
from core.models.turbines import TurbineLog
from core.serializers.turbine_logs import TurbineLogSerializer


class TurbineLogViewSet(viewsets.ModelViewSet):
    serializer_class = TurbineLogSerializer

    def get_queryset(self):
        turbine_id = self.kwargs.get('id')
        get_object_or_404(Turbine, pk=turbine_id)
        return TurbineLog.objects.filter(turbine_id=turbine_id)

    def perform_create(self, serializer):
        turbine_id = self.kwargs.get('id')
        turbine = get_object_or_404(Turbine, pk=turbine_id)
        serializer.save(turbine=turbine)

    @swagger_auto_schema(tags=['Turbine Logs'])
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

