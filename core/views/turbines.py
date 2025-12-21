from rest_framework import viewsets
from drf_yasg.utils import swagger_auto_schema
from core.models.turbine import Turbine
from core.serializers.turbines import TurbineSerializer


class TurbineViewSet(viewsets.ModelViewSet):
    queryset = Turbine.objects.all()
    serializer_class = TurbineSerializer

    @swagger_auto_schema(tags=['Turbines'])
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Turbines'])
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Turbines'])
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Turbines'])
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Turbines'])
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Turbines'])
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)

