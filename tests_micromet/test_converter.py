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


# =============================================================================
# Pytest test suite for uploaded modules
# =============================================================================
# The tests below provide lightweight coverage for the key public functions and
# classes in the user's modules.  They rely only on the Python standard library
# and the packages already required by the modules themselves.  All data used
# are synthetic and small to ensure the test suite runs quickly (\u003c1 s).
# =============================================================================

# -----------------------------------------------------------------------------
# tools.py
# -----------------------------------------------------------------------------
from micromet.tools import (
    polar_to_cartesian_dataframe,
    aggregate_to_daily_centroid,
    # generate_density_raster,  # heavy ‒ skip here, see note below
)


def test_polar_to_cartesian_dataframe_basic():
    df = pd.DataFrame({"WD": [0, 90, 180, 270], "Dist": [1, 1, 1, 1]})
    out = polar_to_cartesian_dataframe(df.copy())

    expected_x = np.array([0, 1, 0, -1], dtype=float)
    expected_y = np.array([1, 0, -1, 0], dtype=float)

    assert np.allclose(out["X_Dist"], expected_x, atol=1e-6)
    assert np.allclose(out["Y_Dist"], expected_y, atol=1e-6)


@pytest.mark.parametrize("weighted", [True, False])
def test_aggregate_to_daily_centroid(weighted):
    base = pd.Timestamp("2025-01-01 00:00:00")
    ts = pd.date_range(base, periods=48, freq="30min", name="Timestamp")
    df = pd.DataFrame(
        {
            "Timestamp": ts,
            "X": np.linspace(0, 10, len(ts)),
            "Y": np.linspace(10, 0, len(ts)),
            "ET": np.ones(len(ts)),
        }
    )

    daily = aggregate_to_daily_centroid(df, weighted=weighted)

    # Exactly one daily centroid expected and no NaNs
    assert len(daily) == 1
    assert daily.isna().sum().sum() == 0


# -----------------------------------------------------------------------------
# ep_footprint.py – lightweight scalar models
# -----------------------------------------------------------------------------
from micromet.ep_footprint import kljun_04, Footprint


def test_footprint_error_helper():
    fp = Footprint.error()
    # All attributes should equal the ERROR flag (−9999)
    assert all(v == -9999 for v in fp.__dict__.values())


def test_kljun_04_valid_case():
    fp = kljun_04(
        std_w=0.5,
        ustar=0.3,
        zL=-1.0,
        sonic_height=3.0,
        disp_height=0.2,
        rough_length=0.05,
    )
    # Basic sanity checks
    assert isinstance(fp, Footprint)
    assert fp.peak > 0
    assert math.isfinite(fp.x50)


# -----------------------------------------------------------------------------
# volk.py – utility helpers (no heavy raster IO tested here)
# -----------------------------------------------------------------------------
from micromet.volk import snap_centroid, norm_minmax_dly_et, norm_dly_et, mask_fp_cutoff


def test_snap_centroid_alignment():
    x_adj, y_adj = snap_centroid(435627.3, 4512322.7)

    # Coordinates should snap to 15 m grid and land on an **odd** multiple
    assert (x_adj / 15) % 2 == 1
    assert (y_adj / 15) % 2 == 1


def test_normalization_helpers():
    arr = pd.Series([2.0, 4.0, 6.0, 8.0])
    mm = norm_minmax_dly_et(arr)
    s = norm_dly_et(arr)

    assert np.isclose(mm.min(), 0)
    assert np.isclose(mm.max(), 1)
    assert np.isclose(s.sum(), 1)


def test_mask_fp_cutoff_retains_core():
    fp = np.ones((10, 10))
    fp[0, 0] = 100  # dominant cell
    masked = mask_fp_cutoff(fp, cutoff=0.5)
    # Some cells should be set to ~0 after masking
    assert np.count_nonzero(masked) < fp.size
    # Dominant cell must remain
    assert masked[0, 0] > 0


# -----------------------------------------------------------------------------
# improved_ffp.py – validate light‑weight internals (heavy calc skipped)
# -----------------------------------------------------------------------------
import logging
from micromet.improved_ffp import FFPModel


def _make_minimal_met_df():
    return pd.DataFrame(
        {
            "V_SIGMA": [1.0],
            "USTAR": [0.5],
            "MO_LENGTH": [-50.0],
            "WD": [180.0],
            "WS": [5.0],
        },
        index=[pd.Timestamp("2025-01-01")],
    )


def test_ffpmodel_basic_validation():
    df = _make_minimal_met_df()
    model = FFPModel(
        df,
        domain=[-100, 100, -100, 100],
        dx=10,
        dy=10,
        nx=10,
        ny=10,
        smooth_data=False,
        verbosity=0,
        logger=logging.getLogger("ffp_test"),
    )
    # Domain should echo back as floats
    assert model.domain == [-100.0, 100.0, -100.0, 100.0]

    # Scaled peak distance is a fixed constant ≈ 0.87
    assert np.isclose(model.calc_scaled_footprint_peak(), 0.87, atol=1e-3)


# -----------------------------------------------------------------------------
# ffp_xr.py – constructor smoke test (skip if xarray heavy deps missing)
# -----------------------------------------------------------------------------
import importlib

ffp_xr_spec = importlib.util.find_spec("ffp_xr")
if ffp_xr_spec is not None:
    from micromet.ffp_xr import ffp_climatology_new

    def test_ffp_xr_smoke():
        df = _make_minimal_met_df().rename(
            columns={
                "V_SIGMA": "V_SIGMA",
                "USTAR": "USTAR",
                "MO_LENGTH": "MO_LENGTH",
                "WD": "WD",
                "WS": "WS",
            }
        )
        model = ffp_climatology_new(
            df,
            domain=[-100, 100, -100, 100],
            dx=10,
            dy=10,
            nx=10,
            ny=10,
            smooth_data=False,
            verbosity=0,
            logger=logging.getLogger("ffp_xr_test"),
        )
        # After define_domain(), grids should have expected shape
        assert model.xv.shape == (len(model.x), len(model.y))

else:
    pytest.skip("ffp_xr module not available", allow_module_level=True)


# -----------------------------------------------------------------------------
# NOTE ON HEAVY TESTS
# -----------------------------------------------------------------------------
# * generate_density_raster() (tools.py) and many volk.py functions rely on heavy
#   geospatial libraries (geopandas, rasterio) or large external datasets.  Those
#   are intentionally **not** exercised here to keep CI fast and dependency‑
#   light.  Add integration tests in a separate, optional test suite if needed.
# =============================================================================
