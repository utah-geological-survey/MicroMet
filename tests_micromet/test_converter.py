from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
import sys
from datetime import datetime
import tempfile
import os
from unittest.mock import mock_open, patch

sys.path.append("../src")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from micromet.converter import Reformatter, AmerifluxDataProcessor

# tests/test_ameriflux_data_processor.py
import importlib


# --------------------------------------------------------------------------- #
# Helper: provide a deterministic _infer_datetime_col that the class expects.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_infer_datetime_col(monkeypatch):
    """
    Replace _infer_datetime_col with a trivial implementation that always
    returns 'timestamp'.  This keeps the test focused on DataFrame handling
    instead of column-detection logic (which is tested separately).
    """
    # Grab the module where the class is defined *dynamically* so the patch will
    # work even if the class is re-exported.
    mod = importlib.import_module(AmerifluxDataProcessor.__module__)
    monkeypatch.setattr(mod, "_infer_datetime_col", lambda df: "timestamp")


# --------------------------------------------------------------------------- #
# 1.  Header-row detection
# --------------------------------------------------------------------------- #
def test_header_rows_tidy_file(tmp_path: Path):
    """A tidy (non-TOA5) file should yield header_rows == 0."""
    csv = tmp_path / "tidy.csv"
    csv.write_text("timestamp,value\n" "2025-01-01 00:00:00,1.23\n")

    proc = AmerifluxDataProcessor(csv)
    assert proc.header_rows == 0, "Non-TOA5 files should have no header offset"


def test_header_rows_toa5(tmp_path: Path):
    """A TOA5 file must keep the default four header rows."""
    csv = tmp_path / "toa5.csv"
    csv.write_text(
        "TOA5,site,logger\n"  # 0
        "meta line 1\n"  # 1
        "meta line 2\n"  # 2
        "meta line 3\n"  # 3
        "timestamp,value\n"  # 4  ← becomes header after skiprows=4
        "2025-01-01 00:00:00,4.56\n"
    )

    proc = AmerifluxDataProcessor(csv)
    assert proc.header_rows == 4, "TOA5 files must skip the four metadata lines"


# --------------------------------------------------------------------------- #
# 2.  DataFrame parsing & sorting
# --------------------------------------------------------------------------- #
def test_to_dataframe_sorted_and_coerced(tmp_path: Path):
    """
    The returned DataFrame should
    * contain parsed, coerced `datetime64[ns]` values
    * be sorted in ascending time order
    * preserve other columns as-is
    """
    csv = tmp_path / "tidy_unsorted.csv"
    csv.write_text(
        "timestamp,value\n"
        "2025-01-02 00:00:00,2.0\n"  # out of order on purpose
        "2025-01-01 00:00:00,1.0\n"
        "bad-time,3.0\n"  # <- coercible to NaT
    )

    df = AmerifluxDataProcessor(csv).to_dataframe()

    # --- column types ------------------------------------------------------ #
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["value"].tolist() == [1.0, 2.0, 3.0]  # same data
    # The bad timestamp should have become NaT and sorted last
    assert df["timestamp"].iloc[0] == pd.Timestamp("2025-01-01 00:00:00")
    assert df["timestamp"].iloc[1] == pd.Timestamp("2025-01-02 00:00:00")
    assert pd.isna(df["timestamp"].iloc[2])

    # --- order ------------------------------------------------------------- #
    # After sorting, the data should be chronological except NaT at the end.
    assert list(df["timestamp"].sort_values(na_position="last")) == list(
        df["timestamp"]
    )


###############################################################################
# Fixtures
###############################################################################
@pytest.fixture(scope="session")
def mod():
    """Import the module that contains `Reformatter` once for all tests."""
    # ⬇️  Replace **reformatter** with the real module name if different
    return importlib.import_module("micromet.converter")


@pytest.fixture
def cfg_dict():
    """A minimal YAML config that exercises every pipeline branch."""
    return {
        # step-2 renames
        "renames_eddy": {"OldCO2": "CO2"},
        "renames_met": {},
        # soil-handling
        "math_soils_v2": [],
        # housekeeping
        "drop_cols": [],
        "column_order": ["CO2", "Tau", "u_star"],
    }


@pytest.fixture
def tmp_cfg(tmp_path: Path, cfg_dict):
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(cfg_dict))
    return p


@pytest.fixture
def var_limits_csv(tmp_path: Path):
    """Provide hard min/max limits for one variable to test clipping."""
    p = tmp_path / "varlimits.csv"
    pd.DataFrame({"variable": ["CO2"], "min": [380.0], "max": [420.0]}).to_csv(
        p, index=False
    )
    return p


@pytest.fixture
def monkeypatched_reformatter(monkeypatch, mod, tmp_cfg):
    """
    Create a Reformatter instance with external helpers & constants patched so
    the tests do not rely on any other project files.
    """
    # ── helpers ────────────────────────────────────────────────────────────
    monkeypatch.setattr(
        mod, "load_yaml", lambda path: yaml.safe_load(Path(path).read_text())
    )
    monkeypatch.setattr(mod, "_infer_datetime_col", lambda df: "TIMESTAMP")

    # ── constants used inside the class but defined elsewhere ─────────────
    monkeypatch.setattr(mod, "MISSING_VALUE", -9999, raising=False)
    monkeypatch.setattr(
        mod, "SOIL_SENSOR_SKIP_INDEX", 3, raising=False
    )  # allow us to drop idx ≥ 3
    monkeypatch.setattr(
        mod, "DEFAULT_SOIL_DROP_LIMIT", 2, raising=False
    )  # keep tests predictable

    return mod.Reformatter(tmp_cfg, drop_soil=False)  # drop_soil toggled per-test


###############################################################################
# Unit-style tests for individual behaviours
###############################################################################
def _base_frame(ts="202501010000"):
    """Convenience helper to build a frame with the mandatory TIMESTAMP col."""
    return pd.DataFrame({"TIMESTAMP": [ts]})


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------
def test_fix_timestamps_resample_and_interpolate(monkeypatched_reformatter):
    """
    _fix_timestamps should:

    1. Parse TIMESTAMP strings → datetime_start.
    2. Drop rows with bad / future timestamps.
    3. Deduplicate on datetime_start (first wins).
    4. Resample to 30-min regular grid.
    5. Interpolate a single missing step (limit=1).

    Expected grid for our sample: 00:00, 00:30, 01:00 on 1 Jan 2025.
    """
    df = pd.DataFrame(
        {
            "TIMESTAMP": [
                "202501010000",  # valid, will keep first duplicate
                "202501010000",
                "202501010015",  # 15-min mark → falls into 00:00 bin
                "202501010100",  # next full hour
                "invalid",  # should be dropped (coerce→NaT)
                "299901011200",  # far future → should be dropped
            ],
            "Val": [1, 2, 3, 4, 5, 6],
        }
    )

    out = monkeypatched_reformatter._fix_timestamps(df)

    # 1. index should be exactly three 30-min steps
    expected_idx = pd.date_range("2025-01-01 00:00", "2025-01-01 01:00", freq="30min")
    assert out.index.equals(expected_idx)

    # 2. duplicate removal – 00:00 keeps first value (1)
    assert out.loc["2025-01-01 00:00", "Val"] == 1

    # 3. interpolation – 00:30 is linear interp between 1 (00:00) and 4 (01:00)
    assert out.loc["2025-01-01 00:30", "Val"] == pytest.approx(2.5)

    # 4. future/invalid rows gone
    assert out.index.max() < pd.Timestamp("2990-01-01")


def test_prefix_and_legacy_conversion(monkeypatched_reformatter):
    """
    BulkEC_, VWC_, Ka_, and T_*cm_* must be normalised AND converted to the
    modern soil naming scheme.
    """
    df = _base_frame().assign(
        BulkEC_5cm_N_Avg=[1.0],
        VWC_10cm_S_Avg=[0.25],
        Ka_20cm_N_Avg=[7.0],
        T_30cm_S_Avg=[12.0],
    )

    out = monkeypatched_reformatter.prepare(df)

    expected = {"EC_3_1_1", "SWC_4_2_1", "K_3_3_1", "TS_4_4_1"}
    assert expected.issubset(out.columns)


def test_tau_fixer_sets_zero_to_missing(monkeypatched_reformatter):
    df = _base_frame().assign(Tau=[0.0], u_star=[0.2])
    out = monkeypatched_reformatter.prepare(df)

    # after prepare(), NaNs have been replaced by MISSING_VALUE (-9999)
    assert out["Tau"].iloc[0] == -9999, "Tau=0 should become the missing-value flag"


def test_swc_scaled_to_percent(monkeypatched_reformatter):
    df = _base_frame().assign(SWC_3_1_1=[0.30])  # 0–1 volumetric
    out = monkeypatched_reformatter.prepare(df)
    assert out["SWC_3_1_1"].iloc[0] == pytest.approx(30.0), "SWC not scaled to %"


def test_ssitc_rating(monkeypatched_reformatter):
    df = _base_frame().assign(FC_SSITC_TEST=[5])  # falls in 4–6 ➜ rating 1
    out = monkeypatched_reformatter.prepare(df)
    assert out["FC_SSITC_TEST"].iloc[0] == 1, "SSITC column not rated correctly"


def test_varlimits_clipping(monkeypatch, mod, tmp_cfg, var_limits_csv):
    fmt = mod.Reformatter(tmp_cfg, var_limits_csv=var_limits_csv, drop_soil=False)
    # patch helpers/constants again for this instance
    monkeypatch.setattr(mod, "_infer_datetime_col", lambda df: "TIMESTAMP")
    monkeypatch.setattr(mod, "MISSING_VALUE", -9999, raising=False)

    df = _base_frame().assign(CO2=[1000.0])  # outside max=420
    out = fmt.prepare(df)
    assert out["CO2"].iloc[0] == 420.0, "Value not clipped to var-limit max"


def test_ssitc_scale_and_rating(monkeypatched_reformatter):
    """
    FC_SSITC_TEST values should be re-rated:
        0–3   → 0
        4–6   → 1
        ≥7    → 2
    The conversion happens only when the column's max() > 3.
    """
    df = pd.DataFrame(
        {
            "TIMESTAMP": ["202501010000", "202501010030", "202501010100"],
            "FC_SSITC_TEST": [2, 5, 8],  # covers all three rating bands
        }
    )

    out = monkeypatched_reformatter.prepare(df)

    # retrieve the three rows that correspond to our inputs
    rated = out["FC_SSITC_TEST"].iloc[:3].to_numpy()
    assert np.array_equal(rated, np.array([0, 1, 2])), "SSITC rating incorrect"
    assert rated.dtype.kind in {"i", "u"}, "Rated values should be integer-typed"


if __name__ == "__main__":
    pytest.main([__file__])
