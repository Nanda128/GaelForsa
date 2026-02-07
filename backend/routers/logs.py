from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from config import get_settings
from db import get_db
from models import Farm, Turbine
from schemas import LogRowIn, LogsOut
from services.csv_store import append_row_and_trim, data_csv_path, read_recent_rows
from services.ml_hooks import on_retrain_trigger


router = APIRouter()


@router.post("/{turbine_id}/logs")
def append_log_row(turbine_id: int, payload: LogRowIn, db: Session = Depends(get_db)):
    turbine = db.query(Turbine).filter(Turbine.id == turbine_id).first()
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    if not turbine.row_count:
        raise HTTPException(status_code=400, detail="Initial CSV upload required")
    farm = db.query(Farm).filter(Farm.id == turbine.farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")

    base_dir = get_settings().data_dir
    path = data_csv_path(base_dir, farm.slug, turbine.slug)
    append_row_and_trim(path, payload.row, turbine.row_count)
    turbine.log_counter = (turbine.log_counter or 0) + 1
    db.add(turbine)
    db.commit()
    if turbine.log_counter % 10 == 0:
        on_retrain_trigger(turbine.id, path)
    return {"status": "ok"}


@router.get("/{turbine_id}/logs", response_model=LogsOut)
def get_logs(
    turbine_id: int,
    limit: int = Query(100, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    turbine = db.query(Turbine).filter(Turbine.id == turbine_id).first()
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    farm = db.query(Farm).filter(Farm.id == turbine.farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")

    base_dir = get_settings().data_dir
    path = data_csv_path(base_dir, farm.slug, turbine.slug)
    rows = read_recent_rows(path, limit=limit)
    return LogsOut(turbine_id=turbine.id, rows=rows)
