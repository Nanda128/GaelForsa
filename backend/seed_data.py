#!/usr/bin/env python3
"""
Seed script to populate the database with example farms and turbines.

This script creates sample wind farms located in Ireland with multiple turbines
each. The data is compatible with the frontend application and can be used for
demonstration purposes.

Usage:
    python seed_data.py

Environment Variables:
    DATABASE_URL: PostgreSQL connection URL
                  (default: postgresql+psycopg://postgres:postgres@localhost:5432/gaelforsa)

Note: If using Docker, make sure the database container is running:
    docker-compose up -d db
"""

import os
import re
import sys
from datetime import datetime, timedelta, timezone
import random

# Add parent directory to path for IDE resolution when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db import Base
from models import Farm, Turbine # noqa

# Default DATABASE_URL matching docker-compose.yml credentials
DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/gaelforsa"


def slugify(value: str) -> str:
    """Generate a URL-friendly slug from a string."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "unnamed"


# Example wind farms in Ireland with realistic coordinates
EXAMPLE_FARMS = [
    {
        "name": "Atlantic Coast Wind Farm",
        "location": "County Clare, Ireland",
        "turbines": [
            {"name": "Turbine Alpha", "turbine_id": 1, "lat": 52.8472, "lon": -9.4289, "status": "green", "health": 95.0},
            {"name": "Turbine Bravo", "turbine_id": 2, "lat": 52.8512, "lon": -9.4345, "status": "green", "health": 92.5},
            {"name": "Turbine Charlie", "turbine_id": 3, "lat": 52.8389, "lon": -9.4201, "status": "yellow", "health": 78.0},
            {"name": "Turbine Delta", "turbine_id": 4, "lat": 52.8445, "lon": -9.4156, "status": "green", "health": 88.5},
            {"name": "Turbine Echo", "turbine_id": 5, "lat": 52.8534, "lon": -9.4098, "status": "red", "health": 45.0},
        ]
    },
    {
        "name": "Galway Bay Offshore",
        "location": "Galway Bay, Ireland",
        "turbines": [
            {"name": "GB-T01", "turbine_id": 101, "lat": 53.1523, "lon": -9.0512, "status": "green", "health": 97.5},
            {"name": "GB-T02", "turbine_id": 102, "lat": 53.1478, "lon": -9.0623, "status": "green", "health": 94.0},
            {"name": "GB-T03", "turbine_id": 103, "lat": 53.1589, "lon": -9.0445, "status": "green", "health": 91.5},
            {"name": "GB-T04", "turbine_id": 104, "lat": 53.1412, "lon": -9.0378, "status": "yellow", "health": 72.0},
        ]
    },
    {
        "name": "Donegal Heights",
        "location": "County Donegal, Ireland",
        "turbines": [
            {"name": "DH-North-1", "turbine_id": 201, "lat": 54.9534, "lon": -7.7201, "status": "green", "health": 99.0},
            {"name": "DH-North-2", "turbine_id": 202, "lat": 54.9612, "lon": -7.7345, "status": "green", "health": 96.5},
            {"name": "DH-South-1", "turbine_id": 203, "lat": 54.9423, "lon": -7.7089, "status": "green", "health": 93.0},
            {"name": "DH-South-2", "turbine_id": 204, "lat": 54.9378, "lon": -7.7234, "status": "yellow", "health": 81.0},
            {"name": "DH-Central", "turbine_id": 205, "lat": 54.9501, "lon": -7.7156, "status": "green", "health": 89.5},
            {"name": "DH-East", "turbine_id": 206, "lat": 54.9567, "lon": -7.6978, "status": "red", "health": 35.0},
        ]
    },
    {
        "name": "Cork Harbour Energy",
        "location": "Cork Harbour, Ireland",
        "turbines": [
            {"name": "CH-Alpha", "turbine_id": 301, "lat": 51.8512, "lon": -8.2945, "status": "green", "health": 94.5},
            {"name": "CH-Beta", "turbine_id": 302, "lat": 51.8478, "lon": -8.3012, "status": "green", "health": 91.0},
            {"name": "CH-Gamma", "turbine_id": 303, "lat": 51.8556, "lon": -8.2867, "status": "green", "health": 88.5},
        ]
    },
    {
        "name": "Midlands Wind Park",
        "location": "County Offaly, Ireland",
        "turbines": [
            {"name": "MW-Unit-1", "turbine_id": 401, "lat": 53.2345, "lon": -7.4912, "status": "green", "health": 96.0},
            {"name": "MW-Unit-2", "turbine_id": 402, "lat": 53.2412, "lon": -7.5023, "status": "yellow", "health": 75.5},
            {"name": "MW-Unit-3", "turbine_id": 403, "lat": 53.2289, "lon": -7.4834, "status": "green", "health": 92.0},
            {"name": "MW-Unit-4", "turbine_id": 404, "lat": 53.2378, "lon": -7.4756, "status": "green", "health": 87.5},
            {"name": "MW-Unit-5", "turbine_id": 405, "lat": 53.2456, "lon": -7.4689, "status": "green", "health": 90.0},
            {"name": "MW-Unit-6", "turbine_id": 406, "lat": 53.2523, "lon": -7.4545, "status": "yellow", "health": 68.0},
            {"name": "MW-Unit-7", "turbine_id": 407, "lat": 53.2234, "lon": -7.5101, "status": "green", "health": 85.0},
        ]
    },
]


def create_seed_data(session):
    """Create example farms and turbines in the database."""
    created_farms = 0
    created_turbines = 0

    for farm_data in EXAMPLE_FARMS:
        # Check if farm already exists
        existing_farm = session.query(Farm).filter(Farm.name == farm_data["name"]).first()
        if existing_farm:
            print(f"  Farm '{farm_data['name']}' already exists, skipping...")
            continue

        # Create the farm
        farm = Farm(
            name=farm_data["name"],
            location=farm_data["location"],
            slug=slugify(farm_data["name"]),
            created_at=datetime.now(timezone.utc) - timedelta(days=random.randint(30, 365))
        )
        session.add(farm)
        session.flush()  # Get the farm ID
        created_farms += 1
        print(f"  Created farm: {farm.name} (ID: {farm.id})")

        # Create turbines for this farm
        for turbine_data in farm_data["turbines"]:
            turbine = Turbine(
                farm_id=farm.id,
                name=turbine_data["name"],
                turbine_id=turbine_data["turbine_id"],
                slug=slugify(turbine_data["name"]),
                latitude=turbine_data["lat"],
                longitude=turbine_data["lon"],
                status=turbine_data["status"],
                health_score=turbine_data["health"],
                created_at=datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30)),
                log_counter=random.randint(0, 100)
            )
            session.add(turbine)
            created_turbines += 1

            status_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(turbine_data["status"], "⚪")
            print(f"      {status_icon} Turbine: {turbine.name} (Health: {turbine.health_score}%)")

    session.commit()
    return created_farms, created_turbines


def main():
    """Main entry point for the seed script."""
    print("\n" + "=" * 60)
    print("GaelForsa Wind Farm Seed Data Script")
    print("=" * 60 + "\n")

    # Get database URL from environment or use default
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    print(f"Database URL: {database_url}\n")

    # Create engine with connection timeout
    try:
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5}  # 5 second timeout
        )

        # Test the connection first
        print("Testing database connection...")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("   Connection successful!\n")

        # Create tables if they don't exist
        Base.metadata.create_all(engine)
        print("Database tables verified/created.\n")

        Session = sessionmaker(bind=engine)
        session = Session()

        print("Seeding example data...\n")
        farms_created, turbines_created = create_seed_data(session)

        print("\n" + "-" * 60)
        print(f"Seeding complete!")
        print(f"   Farms created: {farms_created}")
        print(f"   Turbines created: {turbines_created}")
        print("-" * 60 + "\n")

        # Show summary of all data
        total_farms = session.query(Farm).count()
        total_turbines = session.query(Turbine).count()
        print(f"📈 Database now contains:")
        print(f"   • {total_farms} farm(s)")
        print(f"   • {total_turbines} turbine(s)\n")

        session.close()

    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  1. PostgreSQL is running (try: docker-compose up -d db)")
        print("  2. The database 'gaelforsa' exists")
        print("  3. Credentials are correct (default: postgres:postgres)")
        print("  4. DATABASE_URL environment variable is set correctly if not using defaults")
        print(f"\nCurrent DATABASE_URL: {database_url}")
        print(f"\nExpected format: postgresql+psycopg://user:password@host:port/database")
        sys.exit(1)


if __name__ == "__main__":
    main()

