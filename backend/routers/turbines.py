from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from config import get_settings
from db import get_db
from models import Farm, Turbine
from schemas import TurbineOut, TurbineUpdate
from services.csv_store import (
    data_csv_path,
    feature_csv_path,
    ensure_turbine_dir,
    save_initial_csv,
    save_feature_csv,
    count_rows,
    rename_turbine_dir,
)
from services.utils import slugify
from services.ml_hooks import on_initial_upload


router = APIRouter()


@router.get("", response_model=list[TurbineOut])
def list_turbines(db: Session = Depends(get_db)):
    return db.query(Turbine).all()


@router.post("", response_model=TurbineOut)
def create_turbine(
    farm_id: int = Form(...),
    name: str = Form(...),
    turbine_id: int = Form(...),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    status: str | None = Form(None),
    health_score: float | None = Form(None),
    file: UploadFile = File(...),
    feature_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")
    turbine = Turbine(
        farm_id=farm.id,
        name=name,
        turbine_id=turbine_id,
        slug=slugify(name),
        latitude=latitude,
        longitude=longitude,
        status=status,
        health_score=health_score,
    )
    db.add(turbine)
    db.commit()
    db.refresh(turbine)
    base_dir = get_settings().data_dir
    ensure_turbine_dir(base_dir, farm.slug, turbine.slug)
    dest_path = data_csv_path(base_dir, farm.slug, turbine.slug)
    feature_path = feature_csv_path(base_dir, farm.slug, turbine.slug)
    save_initial_csv(file.file, dest_path)
    save_feature_csv(feature_file.file, feature_path)
    turbine.row_count = count_rows(dest_path)
    turbine.last_csv_upload = datetime.utcnow()
    turbine.csv_path = dest_path
    turbine.feature_csv_path = feature_path
    db.add(turbine)
    db.commit()
    db.refresh(turbine)

    on_initial_upload(turbine.id, dest_path, feature_path)

    return turbine


@router.get("/{turbine_id}", response_model=TurbineOut)
def get_turbine(turbine_id: int, db: Session = Depends(get_db)):
    turbine = db.query(Turbine).filter(Turbine.id == turbine_id).first()
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    return turbine


@router.put("/{turbine_id}", response_model=TurbineOut)
def update_turbine(turbine_id: int, payload: TurbineUpdate, db: Session = Depends(get_db)):
    turbine = db.query(Turbine).filter(Turbine.id == turbine_id).first()
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    old_slug = turbine.slug
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(turbine, field, value)
    if payload.name:
        turbine.slug = slugify(payload.name)
    db.add(turbine)
    db.commit()
    db.refresh(turbine)
    if old_slug != turbine.slug:
        farm = db.query(Farm).filter(Farm.id == turbine.farm_id).first()
        if farm:
            rename_turbine_dir(get_settings().data_dir, farm.slug, old_slug, turbine.slug)
    return turbine


@router.delete("/{turbine_id}")
def delete_turbine(turbine_id: int, db: Session = Depends(get_db)):
    turbine = db.query(Turbine).filter(Turbine.id == turbine_id).first()
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    db.delete(turbine)
    db.commit()
    return {"status": "deleted"}


