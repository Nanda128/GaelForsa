from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class FarmBase(BaseModel):
    name: str
    location: Optional[str] = None


class FarmCreate(FarmBase):
    pass


class FarmUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None


class FarmOut(FarmBase):
    id: int
    slug: str
    created_at: datetime

    class Config:
        from_attributes = True


class TurbineBase(BaseModel):
    farm_id: int
    name: str
    turbine_id: int = Field(..., description="User-specified turbine ID")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: Optional[str] = None
    health_score: Optional[float] = None


class TurbineCreate(TurbineBase):
    pass


class TurbineUpdate(BaseModel):
    name: Optional[str] = None
    turbine_id: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: Optional[str] = None
    health_score: Optional[float] = None


class TurbineOut(TurbineBase):
    id: int
    slug: str
    created_at: datetime
    last_csv_upload: Optional[datetime] = None
    row_count: Optional[int] = None
    csv_path: Optional[str] = None
    feature_csv_path: Optional[str] = None
    log_counter: int = 0

    class Config:
        from_attributes = True


class HealthOut(BaseModel):
    status: str
    health_score: Optional[float] = None


class LogRowIn(BaseModel):
    row: str


class LogsOut(BaseModel):
    turbine_id: int
    rows: list[str]
