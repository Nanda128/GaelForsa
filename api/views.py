from django.http import JsonResponse
from django.db import connection

def api_root_message(request):
    return JsonResponse({"message": "GaelForsa API"})

def health_status(request):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    return JsonResponse({
        "status": "healthy",
        "database": db_status,
        "version": "v1"
    })

