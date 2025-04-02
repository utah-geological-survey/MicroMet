"""
Improved Flux Footprint Prediction (FFP) Module

This module implements the FFP model described in:
Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015:
A simple two-dimensional parameterisation for Flux Footprint Prediction (FFP).
Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015

This version includes improvements for better alignment with the theoretical framework.
"""

import logging
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import traceback


class FFPModel:
    """
    Improved implementation of the Flux Footprint Prediction model.
    """

    REQUIRED_COLUMNS = [
        "V_SIGMA",  # Standard deviation of lateral velocity fluctuations
        "USTAR",  # Friction velocity
        "MO_LENGTH",  # Obukhov length
        "WD",  # Wind direction
        "WS",  # Wind speed
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        domain: list = [-1000.0, 1000.0, -1000.0, 1000.0],
        dx: float = 10.0,
        dy: float = 10.0,
        nx: int = 1000,
        ny: int = 1000,
        rs: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        crop_height: float = 0.2,
        atm_bound_height: float = 2000.0,
        inst_height: float = 2.0,
        rslayer: bool = False,
        smooth_data: bool = True,
        crop: bool = False,
        verbosity: int = 2,
        logger=None,
        **kwargs,
    ):
        """
        Initialize the FFP model with configuration parameters.

        Args:
            df: Input DataFrame containing required meteorological data
            domain: Physical domain boundaries [xmin, xmax, ymin, ymax]
            dx, dy: Grid spacing
            nx, ny: Number of grid points
            rs: List of relative source area contributions to calculate
            crop_height: Vegetation height
            atm_bound_height: Atmospheric boundary layer height
            inst_height: Instrument height
            rslayer: Consider roughness sublayer
            smooth_data: Apply smoothing to output
            crop: Crop output to significant area
            verbosity: Logging detail level
            logger: Logger instance
        """
        # Validate input DataFrame
        self._validate_input_df(df)

        # Initialize basic attributes
        self.sigma_y = None
        self.sigma_y_correction = None
        self.fclim_2d = None
        self.f_2d = None

        self.df = df.copy()  # Make a copy to avoid modifying original
        self.domain = self._validate_domain(domain)
        self.dx = float(dx)
        self.dy = float(dy)
        self.nx = int(nx)
        self.ny = int(ny)
        self.rs = self._validate_rs(rs)

        # Validate physical parameters
        if crop_height < 0:
            raise ValueError("crop_height must be positive")
        if atm_bound_height <= 10:
            raise ValueError("atm_bound_height must be > 10m")
        if inst_height <= crop_height:
            raise ValueError("inst_height must be greater than crop_height")

        self.crop_height = crop_height
        self.atm_bound_height = atm_bound_height
        self.inst_height = inst_height

        self.smooth_data = bool(smooth_data)
        self.crop = bool(crop)
        self.verbosity = int(verbosity)

        # Set up logger
        self.logger = logger or self._setup_logger()
        self.logger.setLevel(logging.DEBUG if verbosity > 1 else logging.INFO)

        # Model constants
        self.k = 0.4  # von Karman constant
        self.oln = 5000.0  # neutral stability limit

        # Add RSL-specific parameters
        self.n_rsl = 2.75  # RSL height multiplier (2 ≤ n ≤ 5 per paper)
        self.rsl_params = {
            "conv": {"ps1": 0.8},  # p value for convective conditions
            "stab": {"ps1": 0.55},  # p value for stable conditions
        }

        # Initialize model parameters (will be updated based on stability)
        self.initialize_model_parameters()

        # Process input data
        self.prep_df_fields(
            crop_height=crop_height,
            inst_height=inst_height,
            atm_bound_height=atm_bound_height,
        )

        # Set up computational domain
        self.define_domain()

        # Create xarray dataset
        self.create_xr_dataset()

        # Perform validity checks
        self.check_validity_ranges()

        # Handle stability regimes
        self.handle_stability_regimes()

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns."""
        missing_cols = [
            col
            for col in self.REQUIRED_COLUMNS
            if col not in map(str.upper, df.columns)
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in input DataFrame: {missing_cols}"
            )

        # Check for invalid values
        for col in self.REQUIRED_COLUMNS:
            if df[col].isnull().any():
                self.logger.warning(f"Found null values in column {col}")
            if not np.isfinite(df[col]).all():
                self.logger.warning(f"Found non-finite values in column {col}")

    def _validate_domain(self, domain: list) -> list:
        """Validate domain specification."""
        if len(domain) != 4:
            raise ValueError("domain must be a list of [xmin, xmax, ymin, ymax]")
        domain = [float(x) for x in domain]
        if domain[0] >= domain[1] or domain[2] >= domain[3]:
            raise ValueError("Invalid domain bounds")
        return domain

    def _validate_rs(self, rs: list) -> list:
        """Validate relative source area contributions."""
        if not isinstance(rs, (list, np.ndarray)):
            raise ValueError("rs must be a list or array")
        rs = [float(r) for r in rs]
        if any(r <= 0 or r >= 1 for r in rs):
            raise ValueError("all rs values must be between 0 and 1")
        return sorted(rs)

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("FFPModel")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def get_scalar_value(self, arr):
        """
        Safely convert array-like object to scalar value.
        Uses mean for arrays with multiple values.

        Args:
            arr: Value to convert (could be array, DataArray, or scalar)

        Returns:
            float: Scalar value
        """
        if isinstance(arr, xr.DataArray):
            return float(arr.mean())
        elif isinstance(arr, np.ndarray):
            return float(np.mean(arr))
        else:
            return float(arr)

    def calc_crosswind_integrated_footprint(self, x_star):
        """
        Calculate the scaled crosswind-integrated footprint F̂y*(X̂*).

        Implementation of Equation 14 from Kljun et al. (2015):
        F̂y* = a(X̂* - d)^b * exp(-c/(X̂* - d))

        Where a=1.452, b=-1.991, c=1.462, d=0.136 (Equation 17)

        Args:
            x_star (xarray.DataArray): Scaled distance X*

        Returns:
            xarray.DataArray: Scaled crosswind-integrated footprint
        """
        self.logger.debug("Calculating crosswind-integrated footprint...")

        try:
            # Get scalar parameters
            a_scalar = self.get_scalar_value(self.a)
            b_scalar = self.get_scalar_value(self.b)
            c_scalar = self.get_scalar_value(self.c)
            d_scalar = self.get_scalar_value(self.d)

            # Calculate modified x_star
            x_star_modified = x_star - d_scalar

            # Create mask for valid values
            mask = x_star_modified > 0

            # Calculate footprint using xarray's where operation
            f_star = xr.where(
                mask,
                a_scalar
                * (x_star_modified**b_scalar)
                * np.exp(-c_scalar / x_star_modified),
                0.0,
            )

            # Log statistics
            self.logger.debug(f"x_star_modified shape: {x_star_modified.shape}")
            self.logger.debug(
                f"x_star_modified range: {float(x_star_modified.min())} to {float(x_star_modified.max())}"
            )
            self.logger.debug(f"f_star shape: {f_star.shape}")
            self.logger.debug(
                f"f_star range: {float(f_star.min())} to {float(f_star.max())}"
            )

            return f_star

        except Exception as e:
            self.logger.error(f"Error in crosswind integration: {str(e)}")
            raise

    def get_source_area_contour(self, r, x_ru, x_rd, y_r):
        """
        Create contour dataset for the source area with unique coordinates.

        Args:
            r: Relative contribution
            x_ru: Upwind distance from peak
            x_rd: Downwind distance from peak
            y_r: Crosswind extent

        Returns:
            xr.Dataset: Source area contour data with unique coordinates
        """
        # Get scalar values
        x_rd_val = self.get_scalar_value(x_rd)
        x_ru_val = self.get_scalar_value(x_ru)
        y_r_val = self.get_scalar_value(y_r)

        # Create distance arrays with unique, evenly spaced coordinates
        x = np.linspace(x_rd_val, x_ru_val, self.nx)
        y = np.linspace(-y_r_val, y_r_val, self.ny)

        # Ensure coordinates are unique by adding small offset where needed
        eps = np.finfo(float).eps  # Smallest possible float value
        x_unique = x + np.arange(len(x)) * eps
        y_unique = y + np.arange(len(y)) * eps

        # Create grid
        xx, yy = np.meshgrid(x_unique, y_unique)

        # Calculate polar coordinates
        rho = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)

        # Get mean wind direction
        wind_dir_mean = self.get_scalar_value(self.ds["wind_dir"])
        rotated_theta = theta - (wind_dir_mean * np.pi / 180.0)

        # Calculate footprint components using mean values
        x_star = self.calc_scaled_x(rho)
        sigma_y = self.calc_crosswind_spread(rho)
        f_y = self.calc_crosswind_integrated_footprint(x_star)

        # Calculate 2D footprint
        f_2d = (
            f_y / (np.sqrt(2 * np.pi) * sigma_y) * np.exp(-(yy**2) / (2.0 * sigma_y**2))
        )

        # Calculate contour level
        total_flux = np.sum(f_2d) * self.dx * self.dy
        sorted_f = np.sort(f_2d.flatten())[::-1]
        cumsum_f = np.cumsum(sorted_f) * self.dx * self.dy
        idx = np.searchsorted(cumsum_f / total_flux, r)
        if idx >= len(sorted_f):
            idx = len(sorted_f) - 1
        contour_level = sorted_f[idx]

        # Create output dataset with unique coordinates
        return xr.Dataset(
            {
                "contour_level": xr.DataArray(contour_level),
                "x": xr.DataArray(x_unique, dims=["x"]),
                "y": xr.DataArray(y_unique, dims=["y"]),
                "f": xr.DataArray(f_2d, dims=["y", "x"]),
            }
        )

    def calc_scaled_x(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the scaled distance X* using mean wind speed.

        Implementation of Equation 6 from Kljun et al. (2015):
        X* = (x/zm) * (1 - zm/h) / (u(zm)/(u*k))

        Where:
        - x is upwind distance
        - zm is measurement height
        - h is boundary layer height
        - u(zm) is mean wind speed at measurement height
        - u* is friction velocity
        - k is von Karman constant (0.4)

        Args:
            x (float or array): Upwind distance [m]

        Returns:
            Scaled distance X*
        """
        # Get mean values
        zm_mean = float(self.ds["zm"].mean())
        h_mean = float(self.ds["h"].mean())
        u_zm_mean = float(self.ds["umean"].mean())
        ustar_mean = float(self.ds["ustar"].mean())

        # Calculate scaling
        result = (
            x / zm_mean * (1 - zm_mean / h_mean) * (ustar_mean / (u_zm_mean * self.k))
        )

        return result

    def initialize_model_parameters(self):
        """
        Initialize model parameters ensuring scalar values.
        """

        # Convert all parameters to scalar values
        def ensure_scalar(val):
            return self.get_scalar_value(val)

        # Base parameters
        self.a = ensure_scalar(1.4524)
        self.b = ensure_scalar(-1.9914)
        self.c = ensure_scalar(1.4622)
        self.d = ensure_scalar(0.1359)

        # Crosswind dispersion parameters
        self.ac = ensure_scalar(2.17)
        self.bc = ensure_scalar(1.66)
        self.cc = ensure_scalar(20.0)

    def check_rsl_validity(self):
        """
        Check if measurements are within roughness sublayer.
        Based on Section 2 of the paper: z* ≈ n h_rs where h_rs ≈ 10z₀
        """
        # Calculate RSL height
        h_rs = 10 * self.df["z0"]  # Roughness element height
        z_star = self.n_rsl * h_rs  # RSL height

        # Check if measurement height is above RSL
        rsl_valid = self.df["zm"] > z_star

        # Log RSL check results
        invalid_count = np.sum(~rsl_valid)
        if invalid_count > 0:
            self.logger.warning(
                f"{invalid_count} measurements within roughness sublayer "
                f"(zm <= {self.n_rsl:.1f} * h_rs)"
            )

        return rsl_valid

    def apply_rsl_corrections(self):
        """
        Apply corrections for measurements within roughness sublayer.
        Following Section 2 and Appendix C of the paper.
        """
        # Calculate RSL parameters
        h_rs = 10 * self.ds["z0"]
        z_star = self.n_rsl * h_rs

        # Identify measurements within RSL
        in_rsl = self.ds["zm"] <= z_star

        if in_rsl.any():
            # Calculate scaling factors for RSL effects
            stability_param = self.ds["zm"] / self.ds["ol"]

            # Get appropriate ps1 value based on stability
            ps1 = xr.where(
                stability_param <= 0,
                self.rsl_params["conv"]["ps1"],
                self.rsl_params["stab"]["ps1"],
            )

            # Calculate RSL correction based on eq. C1-C3
            T = ps1 * self.ds["zm"] / self.ds["ustar"]
            sigma_y0 = self.ds["sigmav"] * T

            # First calculate the basic crosswind spread
            sigma_y = self.calc_crosswind_spread(self.xv)

            # Then combine with near-source dispersion term where needed
            self.sigma_y = xr.where(in_rsl, np.sqrt(sigma_y0**2 + sigma_y**2), sigma_y)

            # Calculate blind zone correction
            self.x_min = self.ds["ustar"] * T

            # Log RSL corrections
            self.logger.info(
                f"Applied RSL corrections to {in_rsl.sum().values} measurements"
            )

    def prep_df_fields(
        self, crop_height: float, inst_height: float, atm_bound_height: float
    ):
        """
        Prepare and validate input data fields.

        Args:
            crop_height: Vegetation height
            inst_height: Instrument height
            atm_bound_height: Atmospheric boundary layer height
        """
        # Calculate displacement height
        d_h = 10 ** (0.979 * np.log10(crop_height) - 0.154)

        # Add derived fields
        self.df["zm"] = inst_height - d_h  # measurement height above displacement
        self.df["h_c"] = crop_height
        self.df["z0"] = crop_height * 0.123  # roughness length
        self.df["h"] = atm_bound_height

        # Rename fields to standard names
        self.df = self.df.rename(
            columns={
                "V_SIGMA": "sigmav",
                "USTAR": "ustar",
                "wd": "wind_dir",
                "WD": "wind_dir",
                "MO_LENGTH": "ol",
                "ws": "umean",
                "WS": "umean",
            }
        )

        # Apply validity checks
        self._apply_validity_masks()

        # Drop invalid data
        self.df = self.df.dropna(subset=["sigmav", "wind_dir", "h", "ol"])
        self.ts_len = len(self.df)
        self.logger.debug(f"Valid input length: {self.ts_len}")

        # Add RSL validity check
        rsl_valid = self.check_rsl_validity()
        self.df["rsl_valid"] = rsl_valid

        # Additional RSL parameters
        self.df["h_rs"] = 10 * self.df["z0"]
        self.df["z_star"] = self.n_rsl * self.df["h_rs"]

        # Log RSL statistics
        mean_z_star = self.df["z_star"].mean()
        self.logger.info(
            f"Mean RSL height (z*): {mean_z_star:.1f}m, "
            f"Measurement height: {inst_height}m"
        )

    def _apply_validity_masks(self):
        """Apply physical validity constraints to input data."""
        self.df["zm"] = np.where(self.df["zm"] <= 0.0, np.nan, self.df["zm"])
        self.df["h"] = np.where(self.df["h"] <= 10.0, np.nan, self.df["h"])
        self.df["zm"] = np.where(self.df["zm"] > self.df["h"], np.nan, self.df["zm"])
        self.df["sigmav"] = np.where(self.df["sigmav"] < 0.0, np.nan, self.df["sigmav"])
        self.df["ustar"] = np.where(self.df["ustar"] <= 0.1, np.nan, self.df["ustar"])
        self.df["wind_dir"] = np.where(
            (self.df["wind_dir"] > 360.0) | (self.df["wind_dir"] < 0.0),
            np.nan,
            self.df["wind_dir"],
        )

    def define_domain(self):
        """Set up the computational domain and grid."""
        self.logger.info("Setting up computational domain...")

        try:
            # Create coordinate arrays
            if self.dx is None and self.nx is not None:
                self.x = np.linspace(self.domain[0], self.domain[1], self.nx + 1)
                self.y = np.linspace(self.domain[2], self.domain[3], self.ny + 1)
            else:
                self.x = np.arange(self.domain[0], self.domain[1] + self.dx, self.dx)
                self.y = np.arange(self.domain[2], self.domain[3] + self.dy, self.dy)

            self.logger.debug(f"Domain dimensions - x: {len(self.x)}, y: {len(self.y)}")

            # Create 2D grid with explicit dimensions
            self.xv, self.yv = np.meshgrid(self.x, self.y, indexing="ij")

            # Create polar coordinate grids as xarray DataArrays with explicit dimensions
            self.rho = xr.DataArray(
                np.sqrt(self.xv**2 + self.yv**2),
                dims=("x", "y"),
                coords={"x": self.x, "y": self.y},
            )

            self.theta = xr.DataArray(
                np.arctan2(self.yv, self.xv),
                dims=("x", "y"),
                coords={"x": self.x, "y": self.y},
            )

            # Initialize footprint grid with proper dimensions
            self.fclim_2d = xr.zeros_like(self.rho)

            self.logger.debug(
                f"Grid shapes - xv: {self.xv.shape}, rho: {self.rho.shape}"
            )
            self.logger.info("Domain setup completed successfully")

        except Exception as e:
            self.logger.error(f"Error in domain setup: {str(e)}")
            raise

    def create_xr_dataset(self):
        """Create xarray Dataset from input DataFrame."""
        self.df.index.name = "time"
        self.ds = xr.Dataset.from_dataframe(self.df)

    def handle_stability_regimes(self):
        """
        Implement comprehensive stability regime classification and handling.
        Based on Section 2 of Kljun et al. (2015).
        """
        # Calculate stability parameter zm/L
        stability_param = self.ds["zm"] / self.ds["ol"]

        # Define regime boundaries per paper
        regimes = xr.Dataset()

        # Convective regimes
        regimes["strongly_unstable"] = stability_param <= -15.5
        regimes["unstable"] = (stability_param > -15.5) & (stability_param < -0.1)

        # Near-neutral regime
        regimes["neutral"] = (stability_param >= -0.1) & (stability_param <= 0.1)

        # Stable regimes
        regimes["stable"] = (stability_param > 0.1) & (
            stability_param < self.oln / self.ds["zm"]
        )
        regimes["strongly_stable"] = stability_param >= self.oln / self.ds["zm"]

        # Calculate regime-specific parameters
        params = {
            "unstable": {"a": 2.930, "b": -2.285, "c": 2.127, "d": -0.107},
            "neutral": {"a": 1.472, "b": -1.996, "c": 1.480, "d": 0.169},
            "stable": {"a": 1.472, "b": -1.996, "c": 1.480, "d": 0.169},
        }

        # Implement smooth transitions between regimes
        transition_zone = 0.1

        def transition_weight(param, center, width):
            """Calculate smooth transition weight."""
            return 0.5 * (1 + np.tanh((param - center) / width))

        # Calculate regime weights
        neutral_to_unstable = transition_weight(stability_param, -0.1, transition_zone)
        neutral_to_stable = transition_weight(stability_param, 0.1, transition_zone)

        # Apply smooth parameter transitions
        for param in ["a", "b", "c", "d"]:
            self.__dict__[param] = (
                neutral_to_unstable * params["unstable"][param]
                + (1 - neutral_to_unstable)
                * (1 - neutral_to_stable)
                * params["neutral"][param]
                + neutral_to_stable * params["stable"][param]
            )

        # Log regime statistics
        for regime in regimes:
            count = regimes[regime].sum().values
            percentage = (count / self.ts_len) * 100
            self.logger.info(f"{regime}: {count} points ({percentage:.1f}%)")

        # Add stability flags to output dataset
        self.ds["stability_regime"] = xr.where(
            regimes["strongly_unstable"],
            -2,
            xr.where(
                regimes["unstable"],
                -1,
                xr.where(regimes["neutral"], 0, xr.where(regimes["stable"], 1, 2)),
            ),
        )

        return regimes

    def check_validity_ranges(self):
        """
        Check validity ranges according to equation 27 and other constraints from Kljun et al. (2015).

        Implements the following validity checks:
        1. Height validity (Eq. 27): 20z₀ < zm < he
           where he ≈ 0.8h is the entrainment height
        2. Stability validity (Eq. 27): -15.5 ≤ zm/L
        3. Turbulence validity: u* > 0.1 m/s, σv > 0
        4. Roughness sublayer validity: zm > z* ≈ n*hrs
           where hrs ≈ 10z₀ and 2 ≤ n ≤ 5 (Section 2)

        Returns:
            xr.Dataset: Dataset containing boolean masks for:
                - height_valid: True where 20z₀ < zm < he
                - stability_valid: True where -15.5 ≤ zm/L
                - turbulence_valid: True where u* > 0.1 and σv > 0
                - rsl_valid: True where measurement height above RSL
                Combined validity is stored in self.valid_footprint
        """
        validity_mask = xr.Dataset()

        # Height validity: 20z₀ < zm < he
        validity_mask["height_valid"] = xr.where(
            (self.ds["zm"] > 20 * self.ds["z0"]) & (self.ds["zm"] < 0.8 * self.ds["h"]),
            True,
            False,
        )

        # Stability validity: -15.5 ≤ zm/L
        validity_mask["stability_valid"] = xr.where(
            self.ds["zm"] / self.ds["ol"] >= -15.5, True, False
        )

        # Turbulence validity
        validity_mask["turbulence_valid"] = xr.where(
            (self.ds["ustar"] > 0.1) & (self.ds["sigmav"] > 0), True, False
        )

        # Combined validity
        self.valid_footprint = validity_mask.all()

        # Add RSL-specific checks
        rsl_valid = self.check_rsl_validity()
        validity_mask["rsl_valid"] = rsl_valid

        # Update combined validity to include RSL
        self.valid_footprint = self.valid_footprint & rsl_valid

        return validity_mask

    def calc_pi_4(self):
        """
        Calculate Π4 with comprehensive stability function implementation including neutral conditions.

        Returns:
            xr.DataArray: Calculated Π4 values with proper neutral condition handling
        """
        stability_param = self.ds["zm"] / self.ds["ol"]

        # Initialize psi_m array
        psi_m = xr.zeros_like(stability_param)

        # Determine stability regimes
        stable_mask = self.ds["ol"] > 0
        unstable_mask = self.ds["ol"] < -self.oln
        neutral_mask = (self.ds["ol"] <= 0) & (self.ds["ol"] >= -self.oln)

        # Calculate psi_m for each stability regime
        # Stable conditions
        psi_m = xr.where(stable_mask, -5.3 * stability_param, psi_m)

        # Unstable conditions
        psi_m = xr.where(unstable_mask, self.calc_unstable_psi(stability_param), psi_m)

        # Neutral conditions - psi_m remains zero
        self.logger.debug(
            f"Stability regime counts - Stable: {stable_mask.sum().values}, "
            f"Unstable: {unstable_mask.sum().values}, "
            f"Neutral: {neutral_mask.sum().values}"
        )

        return np.log(self.ds["zm"] / self.ds["z0"]) - psi_m

    def calc_unstable_psi(self, stability_param):
        """
        Calculate ψM for unstable conditions with improved numerical stability.

        Args:
            stability_param: Stability parameter zm/L

        Returns:
            xr.DataArray: Calculated ψM values for unstable conditions
        """
        # Limit stability parameter to prevent numerical issues
        stability_param = xr.where(stability_param < -15.5, -15.5, stability_param)

        chi = (1 - 19 * stability_param) ** 0.25

        # Implement the full equation with better numerical handling
        return (
            np.log((1 + chi**2) / 2)
            + 2 * np.log((1 + chi) / 2)
            - 2 * np.arctan(chi)
            + np.pi / 2
        )

    def apply_paper_smoothing(self):
        """
        Apply the specific smoothing kernel from Section 4.1 of the paper.
        Returns smoothed DataArray.
        """
        self.logger.debug("Starting paper smoothing...")

        try:
            # Define smoothing kernel
            skernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])

            # Apply convolution twice as specified in paper
            smoothed = self.fclim_2d.copy()

            # Track original stats
            self.logger.debug(
                f"Pre-smoothing stats - Min: {float(smoothed.min()):.2e}, Max: {float(smoothed.max()):.2e}"
            )

            for i in range(2):
                smoothed = xr.apply_ufunc(
                    lambda x: signal.convolve2d(
                        x, skernel, mode="same", boundary="fill", fillvalue=0
                    ),
                    smoothed,
                    input_core_dims=[["x", "y"]],
                    output_core_dims=[["x", "y"]],
                    dask="allowed",
                )

                self.logger.debug(f"Smoothing iteration {i + 1} complete")
                self.logger.debug(
                    f"Post-iteration stats - Min: {float(smoothed.min()):.2e}, Max: {float(smoothed.max()):.2e}"
                )

            if np.isnan(smoothed).any() or not np.any(smoothed):
                self.logger.warning(
                    "Smoothing produced invalid results, reverting to unsmoothed data"
                )
                return self.fclim_2d

            return smoothed

        except Exception as e:
            self.logger.error(f"Error in smoothing: {str(e)}")
            self.logger.debug("Returning unsmoothed data")
            return self.fclim_2d

    def verify_results(self, data, name="data"):
        """
        Safely verify results for NaN and infinite values with proper type checking.

        Args:
            data: Data to verify (can be xarray.DataArray, numpy array, or other)
            name: Name of the data for logging purposes
        """
        try:
            if isinstance(data, xr.DataArray):
                if data.isnull().any():
                    self.logger.warning(f"NaN values detected in {name}")
                if not xr.DataArray.all(data.where(data.notnull())):
                    self.logger.warning(f"Non-finite values detected in {name}")

            elif isinstance(data, np.ndarray):
                if np.any(np.isnan(data)):
                    self.logger.warning(f"NaN values detected in {name}")
                if not np.all(np.isfinite(data)):
                    self.logger.warning(f"Non-finite values detected in {name}")

            else:
                # Try to convert to numpy array first
                try:
                    arr = np.asarray(data)
                    if np.any(np.isnan(arr)):
                        self.logger.warning(f"NaN values detected in {name}")
                    if not np.all(np.isfinite(arr)):
                        self.logger.warning(f"Non-finite values detected in {name}")
                except:
                    self.logger.warning(
                        f"Could not verify {name} for NaN/infinite values"
                    )

        except Exception as e:
            self.logger.warning(f"Error verifying {name}: {str(e)}")

    def verify_data(self, data: xr.DataArray, name: str) -> bool:
        """
        Verify data array validity with detailed logging.

        Args:
            data: Data array to verify
            name: Name of the data for logging

        Returns:
            bool: True if data is valid, False otherwise
        """
        if data is None:
            self.logger.error(f"{name} is None")
            return False

        if not isinstance(data, xr.DataArray):
            self.logger.error(f"{name} is not a DataArray, got {type(data)}")
            return False

        if data.size == 0:
            self.logger.error(f"{name} is empty")
            return False

        if np.all(np.isnan(data)):
            self.logger.error(f"All values in {name} are NaN")
            return False

        # Log statistics for debugging
        self.logger.debug(f"{name} statistics:")
        self.logger.debug(f"  Shape: {data.shape}")
        self.logger.debug(f"  NaN count: {np.sum(np.isnan(data))}")
        self.logger.debug(f"  Min: {float(data.min())}")
        self.logger.debug(f"  Max: {float(data.max())}")
        self.logger.debug(f"  Mean: {float(data.mean())}")

        return True

    def calc_xr_footprint(self):
        """
        Calculate the flux footprint prediction using xarray operations.
        """
        self.logger.info("Starting footprint calculation...")

        try:
            # Calculate wind direction rotation using numpy functions
            self.rotated_theta = self.theta - (self.ds["wind_dir"] * np.pi / 180.0)
            self.logger.debug(f"Rotated theta shape: {self.rotated_theta.shape}")

            # Calculate stability parameter and Monin-Obukhov stability function
            psi_f = self.calc_pi_4()

            # Initialize arrays
            self.logger.debug(
                f"Initializing arrays with shape: x={len(self.x)}, y={len(self.y)}"
            )

            # Calculate scaled distance using numpy functions with xarray
            xstar_ci_dummy = (
                self.rho
                * np.cos(self.rotated_theta)
                / self.ds["zm"]
                * (1.0 - self.ds["zm"] / self.ds["h"])
                / (np.log(self.ds["zm"] / self.ds["z0"]) - psi_f)
            )

            self.logger.debug(f"xstar_ci_dummy shape: {xstar_ci_dummy.shape}")
            self.logger.debug(
                f"xstar_ci_dummy range: {float(xstar_ci_dummy.min())} to {float(xstar_ci_dummy.max())}"
            )

            # Calculate footprint components
            f_ci = self.calc_crosswind_integrated_footprint(xstar_ci_dummy)
            self._log_array_stats("crosswind_integrated_footprint", f_ci)

            # Handle crosswind dispersion
            sigy = self.calc_crosswind_spread_xr(xstar_ci_dummy)
            self._log_array_stats("crosswind_dispersion", sigy)

            # Calculate 2D footprint using numpy functions
            self.f_2d = xr.where(
                sigy > 0,
                f_ci
                / (np.sqrt(2 * np.pi) * sigy)
                * np.exp(
                    -((self.rho * np.sin(self.rotated_theta)) ** 2) / (2.0 * sigy**2)
                ),
                0.0,
            )

            self._log_array_stats("2d_footprint", self.f_2d)

            # Calculate footprint climatology
            footprint_sum = self.f_2d.sum(dim="time")
            total_sum = float(footprint_sum.sum())

            if total_sum < 1e-10:
                self.logger.warning(
                    "Near-zero sum in footprint climatology, initializing with uniform distribution"
                )
                self.fclim_2d = xr.ones_like(footprint_sum) / (
                    len(self.x) * len(self.y)
                )
            else:
                self.fclim_2d = footprint_sum / self.ts_len

            self._log_array_stats("climatology", self.fclim_2d)

            # Apply smoothing if requested
            if self.smooth_data:
                self.fclim_2d = self.apply_paper_smoothing()

            self.logger.info("Footprint calculation completed successfully")
            return self.fclim_2d

        except Exception as e:
            self.logger.error(f"Error in footprint calculation: {str(e)}")
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise

    def _log_array_stats(self, name, arr):
        """Helper method to log array statistics"""
        self.logger.debug(f"{name} statistics:")
        self.logger.debug(f"  Shape: {arr.shape}")
        self.logger.debug(f"  NaN count: {xr.where(np.isnan(arr), 1, 0).sum()}")
        self.logger.debug(f"  Min: {float(arr.min())}")
        self.logger.debug(f"  Max: {float(arr.max())}")
        self.logger.debug(f"  Mean: {float(arr.mean())}")

    def calc_scaled_distance_rsl(self):
        """
        Calculate scaled distance with RSL considerations.
        Based on Appendix C of the paper.
        """
        # Basic scaled distance calculation
        xstar_base = (
            self.rho
            * np.cos(self.rotated_theta)
            / self.ds["zm"]
            * (1.0 - self.ds["zm"] / self.ds["h"])
            / (np.log(self.ds["zm"] / self.ds["z0"]) - self.calc_pi_4())
        )

        # Apply RSL correction where needed
        in_rsl = self.ds["zm"] <= self.ds["z_star"]

        if in_rsl.any():
            # Calculate blind zone adjustment
            x_adj = xr.where(in_rsl, self.x_min * np.cos(self.rotated_theta), 0.0)

            # Adjust scaled distance
            xstar_adj = xr.where(in_rsl, xstar_base + x_adj / self.ds["zm"], xstar_base)

            return xstar_adj

        return xstar_base

    def calculate_pi_groups(self):
        """
        Calculate dimensionless Π groups according to the paper.

        Returns:
            tuple: The four dimensionless groups
        """
        if not hasattr(self, "f_2d"):
            raise AttributeError(
                "f_2d must be initialized before calculating pi groups"
            )

        # Π1 = fy*zm
        pi_1 = self.f_2d * self.ds["zm"]

        # Π2 = x/zm
        pi_2 = self.rho * np.cos(self.rotated_theta) / self.ds["zm"]

        # Π3 = (h - zm)/h = 1 - zm/h
        pi_3 = 1 - self.ds["zm"] / self.ds["h"]

        # Π4 = u(zm)/(u*k) = ln(zm/z0) - ψM
        pi_4 = self.calc_pi_4()

        return pi_1, pi_2, pi_3, pi_4

    def calc_crosswind_dispersion(self, xstar_ci_dummy, px):
        """
        Calculate the scaled crosswind dispersion with numerical safety checks.
        """
        try:
            # Ensure positive input for sqrt
            x_abs = np.abs(xstar_ci_dummy)
            denom = np.maximum(1.0 + self.cc * x_abs, 1e-10)

            sqrt_term = np.clip(
                self.bc * x_abs**2 / denom,
                0.0,  # Ensure non-negative input to sqrt
                1e6,  # Upper limit to prevent overflow
            )

            result = xr.where(px, self.ac * np.sqrt(sqrt_term), 0.0)

            # Safety check for invalid values
            result = xr.where(np.isfinite(result), result, 0.0)

            return result

        except Exception as e:
            self.logger.error(f"Error in crosswind dispersion calculation: {str(e)}")
            return xr.zeros_like(xstar_ci_dummy)

    def scale_crosswind_dispersion(self, sigystar_dummy):
        """
        Scale the dimensionless crosswind dispersion σy* to real-scale σy.

        Implements the real-scale conversion described in Equations 12-13 and surrounding text
        of Kljun et al. (2015):

        σy = σy*/(scale_const) * zm * σv/u*

        where:
        - σy* is dimensionless crosswind dispersion
        - scale_const is stability-dependent scaling factor:
            For L ≤ 0 (convective): scale_const = min(1, 10⁻⁵|zm/L|⁻¹ + 0.80)
            For L > 0 (stable): scale_const = min(1, 10⁻⁵|zm/L|⁻¹ + 0.55)
        - zm is measurement height
        - σv is standard deviation of lateral velocity fluctuations
        - u* is friction velocity

        Args:
            sigystar_dummy (xarray.DataArray): Dimensionless crosswind dispersion σy*

        Returns:
            xarray.DataArray: Real-scale crosswind dispersion σy [m]

        Note:
            Includes numerical safety checks to prevent division by zero and handle
            potential non-finite values in intermediate calculations.
        """
        try:
            self.logger.debug("Starting crosswind dispersion scaling...")
            self.logger.debug(f"Input σy* shape: {sigystar_dummy.shape}")
            self.logger.debug(
                f"σy* range: [{float(sigystar_dummy.min()):.2e}, {float(sigystar_dummy.max()):.2e}]"
            )

            # Calculate stability parameter
            stability_param = self.ds["zm"] / self.ds["ol"]
            self.logger.debug(
                f"Stability parameter range: [{float(stability_param.min()):.2e}, {float(stability_param.max()):.2e}]"
            )

            # Calculate stability-dependent scaling constant with safety
            scale_const = xr.where(
                self.ds["ol"] <= 0,
                1e-5 * np.maximum(np.abs(stability_param), 1e-10) ** (-1)
                + 0.80,  # Convective
                1e-5 * np.maximum(np.abs(stability_param), 1e-10) ** (-1)
                + 0.55,  # Stable
            )

            self.logger.debug(
                f"Scale constant range: [{float(scale_const.min()):.2e}, {float(scale_const.max()):.2e}]"
            )

            # Limit scaling constant
            scale_const = xr.where(
                np.isfinite(scale_const), np.minimum(scale_const, 1.0), 1.0
            )

            self.logger.debug("Applied upper limit of 1.0 to scale constant")
            self.logger.debug(
                f"Final scale constant range: [{float(scale_const.min()):.2e}, {float(scale_const.max()):.2e}]"
            )

            # Calculate scaled dispersion with safety checks
            result = (
                sigystar_dummy
                / np.maximum(scale_const, 1e-10)
                * self.ds["zm"]
                * np.maximum(self.ds["sigmav"], 1e-10)
                / np.maximum(self.ds["ustar"], 1e-10)
            )

            # Log intermediate values
            self.logger.debug(
                f"zm range: [{float(self.ds['zm'].min()):.2e}, {float(self.ds['zm'].max()):.2e}]"
            )
            self.logger.debug(
                f"sigmav range: [{float(self.ds['sigmav'].min()):.2e}, {float(self.ds['sigmav'].max()):.2e}]"
            )
            self.logger.debug(
                f"ustar range: [{float(self.ds['ustar'].min()):.2e}, {float(self.ds['ustar'].max()):.2e}]"
            )

            # Final safety check
            result = xr.where(np.isfinite(result), result, 0.0)

            self.logger.debug(
                f"Final σy range: [{float(result.min()):.2e}, {float(result.max()):.2e}]"
            )
            self.logger.info("Crosswind dispersion scaling completed successfully")

            return result

        except Exception as e:
            self.logger.error(f"Error in scaling crosswind dispersion: {str(e)}")
            self.logger.debug(f"Detailed error trace:", exc_info=True)
            raise

    def calc_2d_footprint(self, f_ci_dummy, sigy_dummy):
        """
        Calculate the two-dimensional footprint distribution f(x,y).

        Implementation of Equation 10 from Kljun et al. (2015):
        f(x,y) = fy(x) * (1/(sqrt(2π)σy)) * exp(-y^2/(2σy^2))

        Where:
        - fy(x) is the crosswind-integrated footprint
        - σy is the standard deviation of crosswind spread
        - y is the crosswind distance from the centerline

        Args:
            f_ci_dummy: Crosswind-integrated footprint
            sigy_dummy: Scaled crosswind dispersion

        Returns:
            xr.DataArray: Two-dimensional footprint [m^-2]
        """
        return xr.where(
            sigy_dummy > 0,
            f_ci_dummy
            / (np.sqrt(2 * np.pi) * sigy_dummy)
            * np.exp(
                -((self.rho * np.sin(self.rotated_theta)) ** 2) / (2.0 * sigy_dummy**2)
            ),
            0.0,
        )

    def calculate_source_areas(self):
        """
        Calculate source areas for different relative contributions.
        Returns dictionary of xarray-compatible objects.
        """
        source_areas = {}

        # Calculate for each requested relative contribution
        for r in self.rs:
            if not 0.1 <= r <= 0.9:
                self.logger.warning(f"Skipping r={r}, outside valid range 0.1-0.9")
                continue

            # Calculate extents
            x_r = self.calc_crosswind_integrated_extent(r)
            x_ru, x_rd = self.calc_peak_based_limits(r)
            y_r = self.calc_crosswind_extent(r, x_ru, x_rd)

            # Store results as xarray objects
            source_areas[f"r_{int(r * 100)}"] = {
                "x_r": xr.DataArray(x_r, name="extent"),
                "x_ru": xr.DataArray(x_ru, name="upwind_extent"),
                "x_rd": xr.DataArray(x_rd, name="downwind_extent"),
                "y_r": xr.DataArray(y_r, name="crosswind_extent"),
                "contour": self.get_source_area_contour(r, x_ru, x_rd, y_r),
            }

        return source_areas

    def calc_crosswind_integrated_extent(self, r: float) -> xr.DataArray:
        """
        Calculate crosswind-integrated footprint extent for relative contribution r.
        Following Equations 23-25 of the paper.

        Args:
            r: Relative contribution (between 0.1 and 0.9)

        Returns:
            xr.DataArray: Distance from receptor containing fraction r of footprint
        """
        # Parameters from Eq. 17
        c = 1.462
        d = 0.136

        # Calculate scaled extent from Eq. 24
        x_star_r = -c / np.log(r) + d

        # Convert to real scale using Eq. 25
        x_r = (
            x_star_r
            * self.ds["zm"]
            * (1 - self.ds["zm"] / self.ds["h"]) ** -1
            * (np.log(self.ds["zm"] / self.ds["z0"]) - self.calc_pi_4())
        )

        return x_r

    def calc_peak_based_limits(self, r: float) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate upwind and downwind distances from the peak location for fraction r.
        Following Eq. 26 of the paper.

        Args:
            r: Relative contribution (between 0.1 and 0.9)

        Returns:
            Tuple containing upwind and downwind distances from peak
        """
        # Calculate peak location (Eq. 20)
        x_star_max = -self.c / self.b + self.d

        # Get x_star_r from crosswind integration
        x_star_r = -self.c / np.log(r) + self.d

        # Calculate downwind limit (Eq. 26 with first set of parameters)
        x_star_rd = 0.44 * x_star_r**-0.77 + 0.24

        # New fixed condition:
        condition = (x_star_max < x_star_r) & (x_star_r <= 1.5)

        # Calculate upwind limit with split conditions (Eq. 26)
        x_star_ru = xr.where(
            condition,
            0.60 * x_star_r**1.32 + 0.61,
            0.96 * x_star_r**1.01 + 0.19,
        )

        # Convert to real scale
        x_rd = self.scale_to_real_distance(x_star_rd)
        x_ru = self.scale_to_real_distance(x_star_ru)

        return x_ru, x_rd

    def calc_real_footprint_peak(
        self,
        zm: Union[float, xr.DataArray],
        h: Union[float, xr.DataArray],
        u_zm: Union[float, xr.DataArray] = None,
        ustar: Union[float, xr.DataArray] = None,
        z0: Union[float, xr.DataArray] = None,
        k: float = 0.4,
    ) -> Union[float, xr.DataArray]:
        """
        Calculate the real-scale footprint peak location.
        Based on Equations 20-22 of Kljun et al. (2015).
        Works with both scalar values and xarray DataArrays.

        Args:
            zm: Measurement height [m]
            h: Boundary layer height [m]
            u_zm: Mean wind speed at measurement height [m/s], optional
            ustar: Friction velocity [m/s], optional
            z0: Roughness length [m], optional
            k: von Karman constant (default=0.4)

        Returns:
            Real distance to footprint peak [m]
            Returns DataArray if inputs are DataArrays, float if inputs are scalars
        """
        # Calculate scaled peak location (Eq. 20)
        x_star_max = -self.c / self.b + self.d  # Should evaluate to 0.87

        # Check which conversion method to use based on available inputs
        if u_zm is not None and ustar is not None:
            # Use Eq. 21 with wind speed and friction velocity
            x_max = x_star_max * zm * (1 - zm / h) ** -1 * u_zm / (ustar * k)

        elif z0 is not None:
            # Use Eq. 22 with roughness length and stability correction
            psi_m = self.calc_pi_4()
            x_max = x_star_max * zm * (1 - zm / h) ** -1 * (np.log(zm / z0) - psi_m)

        else:
            raise ValueError("Must provide either (u_zm, ustar) or z0")

        # Log output differently for scalar vs DataArray
        if isinstance(x_max, xr.DataArray):
            self.logger.debug(
                f"Calculated footprint peak with mean at x = {float(x_max.mean()):.2f} m"
            )
        else:
            self.logger.debug(f"Calculated footprint peak at x = {x_max:.2f} m")

        return x_max

    def calc_scaled_footprint_peak(self) -> float:
        """
        Calculate the scaled footprint peak location (X*max).
        Based on Equation 20 of Kljun et al. (2015).

        Returns:
            float: Scaled distance to footprint peak (constant)
        """
        return -self.c / self.b + self.d  # Should evaluate to 0.87

    def scale_to_real_distance(self, x_star: xr.DataArray) -> xr.DataArray:
        """
        Convert scaled distance to real distance using Eq. 6/7.

        Args:
            x_star: Scaled distance X*

        Returns:
            xr.DataArray: Real-scale distance
        """
        return (
            x_star
            * self.ds["zm"]
            * (1 - self.ds["zm"] / self.ds["h"]) ** -1
            * (np.log(self.ds["zm"] / self.ds["z0"]) - self.calc_pi_4())
        )

    def calc_crosswind_extent(
        self, r: float, x_ru: xr.DataArray, x_rd: xr.DataArray
    ) -> xr.DataArray:
        """
        Calculate crosswind extent of source area using parameterized relationships.

        Args:
            r: Relative contribution
            x_ru: Upwind distance from peak
            x_rd: Downwind distance from peak

        Returns:
            xr.DataArray: Crosswind extent at each distance
        """
        # Calculate sigma_y at peak location
        x_peak = self.calc_real_footprint_peak(
            self.ds["zm"], self.ds["h"], self.ds["umean"], self.ds["ustar"]
        )
        sigma_y_peak = self.calc_crosswind_spread(x_peak)

        # Scale crosswind extent based on distance from peak
        def scale_sigma(x_dist):
            return sigma_y_peak * np.sqrt(x_dist / x_peak)

        y_r = xr.where(x_rd <= x_peak, scale_sigma(x_rd), scale_sigma(x_ru))

        return y_r

    def calc_crosswind_spread_xr(self, x_star):
        """
        Calculate the standard deviation of crosswind spread σy.

        Implements Equations 18-19 from Kljun et al. (2015):
        σy* = ac * sqrt((bc * |X*|^2)/(1 + cc|X*|))

        Where ac=2.17, bc=1.66, cc=20.0 (Equation 19)

        Real-scale σy is then obtained through:
        σy = σy*/(scale_const) * zm * σv/u*

        Where scale_const depends on stability (Eq. not numbered in paper):
        - For unstable: 1e-5|zm/L|^(-1) + 0.80
        - For stable: 1e-5|zm/L|^(-1) + 0.55

        Args:
            x_star (float or array): Distance from receptor [m]

        Returns:
            Standard deviation of crosswind spread [m]
        """
        try:
            # Calculate scaled crosswind spread σy* using Eq. 18
            sigma_y_star = self.ac * np.sqrt(
                (self.bc * x_star**2) / (1 + self.cc * x_star)
            )

            scale_const = self.scale_crosswind_dispersion(sigma_y_star)

            # Convert to real scale
            sigma_y = (
                sigma_y_star
                / scale_const
                * self.ds["zm"]
                * self.ds["sigmav"]
                / self.ds["ustar"]
            )

            return sigma_y

        except Exception as e:
            self.logger.error(f"Error calculating crosswind spread: {str(e)}")
            raise

    def calc_crosswind_spread(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate the standard deviation of crosswind spread σy.

        Implements Equations 18-19 from Kljun et al. (2015):
        σy* = ac * sqrt((bc * |X*|^2)/(1 + cc|X*|))

        Where ac=2.17, bc=1.66, cc=20.0 (Equation 19)

        Real-scale σy is then obtained through:
        σy = σy*/(scale_const) * zm * σv/u*

        Where scale_const depends on stability (Eq. not numbered in paper):
        - For unstable: 1e-5|zm/L|^(-1) + 0.80
        - For stable: 1e-5|zm/L|^(-1) + 0.55

        Args:
            x (float or array): Distance from receptor [m]

        Returns:
            Standard deviation of crosswind spread [m]
        """
        # Get mean values of stability parameters for consistent calculation
        zm_mean = float(self.ds["zm"].mean())
        h_mean = float(self.ds["h"].mean())
        u_zm_mean = float(self.ds["umean"].mean())
        ustar_mean = float(self.ds["ustar"].mean())
        sigmav_mean = float(self.ds["sigmav"].mean())
        ol_mean = float(self.ds["ol"].mean())

        # Calculate mean stability parameter
        stability_param_mean = zm_mean / ol_mean

        # Calculate scaled distance X*
        x_star = self.calc_scaled_x(x)

        # Calculate scaled crosswind spread σy* using Eq. 18
        # Parameters from Eq. 19: ac = 2.17, bc = 1.66, cc = 20.0
        sigma_y_star = self.ac * np.sqrt((self.bc * x_star**2) / (1 + self.cc * x_star))

        # Calculate stability-dependent scaling constant
        if stability_param_mean <= 0:  # Convective
            scale_const = 1e-5 * abs(stability_param_mean) ** (-1) + 0.80
        else:  # Stable
            scale_const = 1e-5 * abs(stability_param_mean) ** (-1) + 0.55

        # Limit scaling constant to maximum of 1.0
        scale_const = min(scale_const, 1.0)

        # Convert to real scale (Eq. 18)
        sigma_y = sigma_y_star / scale_const * zm_mean * sigmav_mean / ustar_mean

        return sigma_y

    def plot_footprint(self, config=None, ax=None, show_contours=True, levels=10):
        """
        Plot the footprint climatology with optional contours.

        Args:
            config: Configuration object containing metadata (optional)
            ax: Matplotlib axis to plot on (optional)
            show_contours: Whether to show contour lines
            levels: Number of contour levels or list of levels

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 8))
            else:
                fig = ax.figure

            # Create meshgrid for plotting if needed
            if not hasattr(self, "X") or not hasattr(self, "Y"):
                self.X, self.Y = np.meshgrid(self.x, self.y)

            self.logger.debug(f"Domain shapes - x: {len(self.x)}, y: {len(self.y)}")
            self.logger.debug(f"Footprint climatology shape: {self.fclim_2d.shape}")
            self.logger.debug(
                f"Footprint climatology stats - min: {float(self.fclim_2d.min())}, max: {float(self.fclim_2d.max())}"
            )

            # Plot filled contours
            contourf = ax.contourf(
                self.X, self.Y, self.fclim_2d, levels=levels, cmap="YlOrRd"
            )
            plt.colorbar(contourf, ax=ax, label="Flux footprint")

            # Add contour lines if requested
            if show_contours:
                contour = ax.contour(
                    self.X,
                    self.Y,
                    self.fclim_2d,
                    levels=levels,
                    colors="k",
                    alpha=0.5,
                    linewidths=0.5,
                )
                ax.clabel(contour, inline=True, fontsize=8, fmt="%.1e")

            # Add tower location
            ax.plot(0, 0, "k^", markersize=10, label="Tower")

            # Set labels and title
            ax.set_xlabel("Distance [m]")
            ax.set_ylabel("Distance [m]")

            # Try to get site name from config if available
            title = "Flux Footprint"
            if config is not None:
                try:
                    site_name = config.get("METADATA", "site_name", fallback="").strip()
                    if site_name:
                        title = f"Flux Footprint - {site_name}"
                except Exception as e:
                    self.logger.warning(
                        f"Could not get site name from config: {str(e)}"
                    )

            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.axis("equal")
            ax.legend()

            return fig, ax

        except Exception as e:
            self.logger.error(f"Error plotting footprint: {str(e)}")
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise

    def run(self, return_result: bool = True) -> Optional[xr.Dataset]:
        """
        Execute the FFP model calculations.

        Args:
            return_result: If True, returns complete results dataset

        Returns:
            Optional[xr.Dataset]: Dataset containing footprint results
        """
        self.logger.info("Starting FFP model calculations...")

        try:
            # Ensure domain is properly initialized
            if not hasattr(self, "x") or not hasattr(self, "y"):
                self.define_domain()

            # Initialize fclim_2d with zeros before calculations
            self.fclim_2d = xr.DataArray(
                np.zeros((len(self.x), len(self.y))),
                dims=("x", "y"),
                coords={"x": self.x, "y": self.y},
            )

            # Perform main calculations
            self.fclim_2d = self.calc_xr_footprint()  # Now returns fclim_2d directly

            # Add validation checks
            if self.fclim_2d is None:
                self.logger.error("fclim_2d is None after calculations")
                raise ValueError("Footprint climatology calculation failed")

            if not isinstance(self.fclim_2d, xr.DataArray):
                self.logger.error(
                    f"fclim_2d is not a DataArray, got {type(self.fclim_2d)}"
                )
                raise ValueError("Invalid footprint climatology type")

            if np.all(np.isnan(self.fclim_2d)):
                self.logger.error("All values in fclim_2d are NaN")
                raise ValueError("Invalid footprint climatology values")

            if return_result:
                self.logger.debug(f"Domain shapes - x: {len(self.x)}, y: {len(self.y)}")
                self.logger.debug(f"Footprint climatology shape: {self.fclim_2d.shape}")
                self.logger.debug(
                    f"Footprint climatology stats - min: {float(self.fclim_2d.min())}, max: {float(self.fclim_2d.max())}"
                )

                try:
                    # Create base results dataset with explicit dimensions
                    results = xr.Dataset(
                        {
                            "footprint_climatology": self.fclim_2d,  # Use DataArray directly
                            "domain_x": ("x", self.x),
                            "domain_y": ("y", self.y),
                        }
                    )

                    if hasattr(self, "f_2d") and self.f_2d is not None:
                        if "time" in self.f_2d.dims:
                            results["footprint_2d"] = self.f_2d
                        else:
                            self.logger.warning("f_2d missing time dimension")

                    # Add source areas if calculated
                    if hasattr(self, "source_areas") and self.source_areas is not None:
                        for r_level, area_dict in self.source_areas.items():
                            for key, value in area_dict.items():
                                if isinstance(value, xr.DataArray):
                                    results[f"{r_level}_{key}"] = value

                    self.logger.info("FFP model calculations completed successfully")
                    return results

                except Exception as e:
                    self.logger.error(f"Error creating results dataset: {str(e)}")
                    raise

        except Exception as e:
            self.logger.error(f"Error in FFP calculations: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)  # Add full traceback
            raise

        self.logger.info("FFP model calculations completed")
        return None

    def save_results(self, filename: str):
        """
        Save model results to a netCDF file.

        Args:
            filename: Path where the results should be saved
        """
        results = xr.Dataset(
            {
                "footprint_2d": self.f_2d,
                "footprint_climatology": self.fclim_2d,
                "domain_x": xr.DataArray(self.x, dims=["x"]),
                "domain_y": xr.DataArray(self.y, dims=["y"]),
                "parameters": xr.DataArray(
                    {
                        "crop_height": float(self.df["h_c"].iloc[0]),
                        "inst_height": float(self.df["zm"].iloc[0])
                        + float(self.df["h_c"].iloc[0]),
                        "atm_bound_height": float(self.df["h"].iloc[0]),
                    }
                ),
            }
        )

        # Add source areas as individual variables
        if hasattr(self, "source_areas"):
            for key, value in self.source_areas.items():
                results[key] = value

        results.to_netcdf(filename)
        self.logger.info(f"Results saved to {filename}")
