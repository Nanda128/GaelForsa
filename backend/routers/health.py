from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db import get_db
from models import Turbine
from schemas import HealthOut


router = APIRouter()


@router.get("/{turbine_id}/health", response_model=HealthOut)
def get_health(turbine_id: int, db: Session = Depends(get_db)):
    turbine = db.query(Turbine).filter(Turbine.id == turbine_id).first()
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    if turbine.health_score is None:
        return HealthOut(status="pending", health_score=None)
    return HealthOut(status="ready", health_score=turbine.health_score)
