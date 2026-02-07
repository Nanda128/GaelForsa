from fastapi import FastAPI

from db import Base, engine
from routers import farms, turbines, logs, health


def create_app() -> FastAPI:
    app = FastAPI(
        title="GaelFora API",
        version="0.6.7",
        description="api for turbines or smth",
    )

    Base.metadata.create_all(bind=engine)

    app.include_router(farms.router, prefix="/api/farms", tags=["farms"])
    app.include_router(turbines.router, prefix="/api/turbines", tags=["turbines"])
    app.include_router(logs.router, prefix="/api/turbines", tags=["logs"])
    app.include_router(health.router, prefix="/api/turbines", tags=["health"])

    return app


app = create_app()
