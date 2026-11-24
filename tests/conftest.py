import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def app():
    from app import create_app
    from config import TestConfig
    return create_app(TestConfig)


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def analytic():
    import pandas as pd
    return pd.read_parquet(ROOT / "data" / "processed" / "orders_analytic.parquet")


# Tests that need the full-rebuild files (raw data -> scripts/run_all.py). The shipped ZIP leaves out the
# large intermediate files and the 119 MB database, so these tests are SKIPPED (not failed) until you rebuild.
_REBUILD_FILES = [ROOT / "data" / "processed" / "orders_all.parquet",
                  ROOT / "data" / "processed" / "orders_analytic.parquet",
                  ROOT / "database" / "ecx.db"]
_REBUILD_TESTS = {"test_data_pipeline.py", "test_leakage.py"}
_NEEDS_ORDERS = ROOT / "data" / "processed" / "orders_with_history.parquet"      # used by the operations tests that run on the real Olist data
_NEEDS_ORDERS_TESTS = {"test_ops_pages_and_api_flow", "test_api_v1_key_flow_scopes_and_rate_limit", "test_digest_route_and_cli", "test_readyz_reports_checks"}
_REBUILD_NAMES = {"test_health_ok", "test_explain_endpoint"}


def pytest_collection_modifyitems(config, items):
    if not _NEEDS_ORDERS.exists():
        sk = pytest.mark.skip(reason="needs data/processed/orders_with_history.parquet")
        for it in items:
            if it.name in _NEEDS_ORDERS_TESTS:
                it.add_marker(sk)
    if all(f.exists() for f in _REBUILD_FILES):
        return
    skip = pytest.mark.skip(reason="needs the full rebuild from the raw data (see README 'Full rebuild')")
    for it in items:
        if it.fspath.basename in _REBUILD_TESTS or it.name in _REBUILD_NAMES:
            it.add_marker(skip)
