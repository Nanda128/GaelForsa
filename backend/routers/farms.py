from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db import get_db
from models import Farm
from schemas import FarmCreate, FarmOut, FarmUpdate
from config import get_settings
from services.csv_store import rename_farm_dir
from services.utils import slugify


router = APIRouter()


@router.get("", response_model=list[FarmOut])
def list_farms(db: Session = Depends(get_db)):
    return db.query(Farm).all()


@router.post("", response_model=FarmOut)
def create_farm(payload: FarmCreate, db: Session = Depends(get_db)):
    existing = db.query(Farm).filter(Farm.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Farm already exists")
    farm = Farm(name=payload.name, location=payload.location, slug=slugify(payload.name))
    db.add(farm)
    db.commit()
    db.refresh(farm)
    return farm


@router.get("/{farm_id}", response_model=FarmOut)
def get_farm(farm_id: int, db: Session = Depends(get_db)):
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")
    return farm


@router.put("/{farm_id}", response_model=FarmOut)
def update_farm(farm_id: int, payload: FarmUpdate, db: Session = Depends(get_db)):
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")
    old_slug = farm.slug
    if payload.name:
        farm.name = payload.name
        farm.slug = slugify(payload.name)
    if payload.location is not None:
        farm.location = payload.location
    db.add(farm)
    db.commit()
    db.refresh(farm)
    if old_slug != farm.slug:
        rename_farm_dir(get_settings().data_dir, old_slug, farm.slug)
    return farm


@router.delete("/{farm_id}")
def delete_farm(farm_id: int, db: Session = Depends(get_db)):
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")
    db.delete(farm)
    db.commit()
    return {"status": "deleted"}
