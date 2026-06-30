from logging.config import fileConfig
from pathlib import Path
import os

from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv

# ----------------------------
# Alembic config object
# ----------------------------
config = context.config

# ----------------------------
# Logging
# ----------------------------
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ----------------------------
# Load .env
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing in .env")

# IMPORTANT: Alembic expects sqlalchemy.url
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# ----------------------------
# Metadata (not used yet)
# ----------------------------
target_metadata = None


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        literal_binds=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()