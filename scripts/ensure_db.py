"""
Create the amharic_rag database if it does not exist.
Uses DATABASE_URL from .env (connects to 'postgres' DB to run CREATE DATABASE).
"""
import os
import re
import sys

# Load .env from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from pathlib import Path
env_path = Path(project_root) / ".env"
if not env_path.exists():
    print("No .env file found. Copy .env.example to .env and set DATABASE_URL.")
    sys.exit(1)

# Simple .env parse
env = {}
with open(env_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

url = env.get("DATABASE_URL")
if not url:
    print("DATABASE_URL not set in .env")
    sys.exit(1)

# Parse postgresql+psycopg://user:pass@host:port/dbname
m = re.match(r"postgresql\+psycopg://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", url)
if not m:
    m = re.match(r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", url)
if not m:
    print("Could not parse DATABASE_URL")
    sys.exit(1)

user, password, host, port, dbname = m.groups()
# Connect to default 'postgres' database to create our DB
try:
    import psycopg
except ImportError:
    print("Install psycopg: pip install psycopg[binary]")
    sys.exit(1)

conninfo = f"postgresql://{user}:{password}@{host}:{port}/postgres"
try:
    with psycopg.connect(conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (dbname,),
            )
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{dbname}"')
                print(f"Created database: {dbname}")
            else:
                print(f"Database already exists: {dbname}")
except Exception as e:
    print(f"Database check failed: {e}")
    print("Ensure PostgreSQL is running and .env DATABASE_URL is correct.")
    sys.exit(1)
