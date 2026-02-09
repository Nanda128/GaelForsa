#!/usr/bin/env python3
"""
Script to clear all farms and turbines from the database.

This script removes all turbine and farm records from the database.
DON'T USE THIS WITH REAL WORLD DATA. IT'S MEANT FOR TESTING PURPOSES ONLY.

Usage:
    python clear_data.py
    python clear_data.py --force  # Skip confirmation prompt

Environment Variables:
    DATABASE_URL: PostgreSQL connection URL
                  (default: postgresql+psycopg://postgres:postgres@localhost:5432/gaelforsa)

Use "docker-compose up -d db" to make sure the database is running before executing this script
"""

import os
import sys

# Add parent directory to path for IDE resolution when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db import Base
from models import Farm, Turbine  # noqa

DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/gaelforsa"


def clear_all_data(session):
    """Delete all turbines and farms from the database."""

    turbine_count = session.query(Turbine).count()
    farm_count = session.query(Farm).count()

    if turbine_count == 0 and farm_count == 0:
        print("   Database is already empty.")
        return 0, 0

    print(f"   Deleting {turbine_count} turbine(s)...")
    session.query(Turbine).delete()

    print(f"   Deleting {farm_count} farm(s)...")
    session.query(Farm).delete()

    session.commit()
    return farm_count, turbine_count


def main():
    """Main entry point for the clear script."""
    print("\n" + "=" * 60)
    print("GaelForsa Database Clear Script")
    print("=" * 60 + "\n")

    # Check for --force flag
    force = "--force" in sys.argv or "-f" in sys.argv

    # Get database URL from environment or use default
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    print(f"Database URL: {database_url}\n")

    # Create engine with connection timeout
    try:
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5}  # 5s timeout
        )

        # Test the connection first
        print("Testing database connection...")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("   Connection successful!\n")

        # Ensure tables exist
        Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        # Show current counts
        total_farms = session.query(Farm).count()
        total_turbines = session.query(Turbine).count()
        print(f"Current database contents:")
        print(f"   • {total_farms} farm(s)")
        print(f"   • {total_turbines} turbine(s)\n")

        if total_farms == 0 and total_turbines == 0:
            print("Database is already empty. Nothing to do.\n")
            session.close()
            return

        # Confirmation prompt (unless --force is used)
        if not force:
            print("⚠️  WARNING: This will permanently delete ALL farms and turbines!")
            response = input("   Are you sure you want to continue? (yes/no): ").strip().lower()
            if response not in ("yes", "y"):
                print("\nOperation cancelled.\n")
                session.close()
                return
            print()

        print("  Clearing database...\n")
        farms_deleted, turbines_deleted = clear_all_data(session)

        print("\n" + "-" * 60)
        print(f" Database cleared successfully!")
        print(f"   Farms deleted: {farms_deleted}")
        print(f"   Turbines deleted: {turbines_deleted}")
        print("-" * 60 + "\n")

        session.close()

    except Exception as e:
        print(f"\n Error: {e}")
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

