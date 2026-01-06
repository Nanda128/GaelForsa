from rest_framework import viewsets, filters
from rest_framework.pagination import PageNumberPagination
from drf_yasg.utils import swagger_auto_schema
from core.models.turbine import Turbine
from core.serializers.turbines import TurbineSerializer


class TurbinePagination(PageNumberPagination):
    page_size = 50
    page_size_query_param = 'page_size'
    max_page_size = 1000


class TurbineViewSet(viewsets.ModelViewSet):
    queryset = Turbine.objects.all()
    serializer_class = TurbineSerializer
    pagination_class = TurbinePagination
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name']
    ordering_fields = ['name', 'status', 'created_at', 'updated_at']
    ordering = ['name']

    def get_queryset(self):
        queryset = super().get_queryset()
        status = self.request.query_params.get('status', None)
        if status:
            queryset = queryset.filter(status=status)
        return queryset

    @swagger_auto_schema(
        tags=['Turbines'],
        manual_parameters=[
            {'name': 'status', 'in': 'query', 'description': 'Filter by status (green, yellow, red)', 'type': 'string'},
            {'name': 'search', 'in': 'query', 'description': 'Search by turbine name', 'type': 'string'},
            {'name': 'ordering', 'in': 'query', 'description': 'Order by field (name, status, created_at, updated_at)', 'type': 'string'},
        ]
    )
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
