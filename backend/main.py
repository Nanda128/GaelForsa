from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import Base, engine
from routers import farms, turbines, logs, health


def create_app() -> FastAPI:
    app = FastAPI(
        title="GaelFora API",
        version="0.6.7",
        description="api for turbines or smth",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Vite dev server
            "http://localhost:4173",  # Vite preview
            "http://127.0.0.1:3000",
            "http://127.0.0.1:4173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    Base.metadata.create_all(bind=engine)

    app.include_router(farms.router, prefix="/api/farms", tags=["farms"])
    app.include_router(turbines.router, prefix="/api/turbines", tags=["turbines"])
    app.include_router(logs.router, prefix="/api/turbines", tags=["logs"])
    app.include_router(health.router, prefix="/api/turbines", tags=["health"])

    return app


app = create_app()
