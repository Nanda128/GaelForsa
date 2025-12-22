from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from core.models import Turbine


class HealthEndpointTest(APITestCase):
    def test_health_endpoint(self):
        client = APIClient()
        response = client.get('/healthz')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('database', data)
        self.assertEqual(data['status'], 'healthy')

    def test_health_endpoint_no_auth_required(self):
        client = APIClient()
        response = client.get('/healthz')
        self.assertEqual(response.status_code, status.HTTP_200_OK)


class RootEndpointTest(APITestCase):
    def test_root_endpoint(self):
        client = APIClient()
        response = client.get('/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)


class PaginationTest(APITestCase):
    def setUp(self):
        self.client = APIClient()
        
        for i in range(150):
            Turbine.objects.create(name=f'Turbine {i}', latitude=55.0 + i * 0.01, longitude=-1.0 + i * 0.01)

    def test_pagination_public(self):
        response = self.client.get('/api/v1/turbines/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.data)
        self.assertIn('next', response.data)
        self.assertIn('previous', response.data)
        self.assertEqual(len(response.data['results']), 100)

    def test_pagination_next_page_public(self):
        response = self.client.get('/api/v1/turbines/?page=2')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsNotNone(response.data['previous'])
        self.assertEqual(len(response.data['results']), 50)

