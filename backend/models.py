from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from db import Base


class Farm(Base):
    __tablename__ = "farms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    location = Column(String, nullable=True)
    slug = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    turbines = relationship("Turbine", back_populates="farm", cascade="all, delete-orphan")


class Turbine(Base):
    __tablename__ = "turbines"

    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(Integer, ForeignKey("farms.id"), nullable=False)
    name = Column(String, nullable=False)
    turbine_id = Column(Integer, nullable=False)
    slug = Column(String, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    status = Column(String, nullable=True)
    health_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_csv_upload = Column(DateTime, nullable=True)
    row_count = Column(Integer, nullable=True)
    csv_path = Column(String, nullable=True)
    feature_csv_path = Column(String, nullable=True)
    log_counter = Column(Integer, nullable=False, default=0)

    farm = relationship("Farm", back_populates="turbines")
