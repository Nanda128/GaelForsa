from functools import lru_cache
from pydantic import BaseModel
import os


class Settings(BaseModel):
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg://localhost:5432/gaelforsa"
    )
    data_dir: str = os.getenv("DATA_DIR", "data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
