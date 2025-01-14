import logging
import numbers

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from affine import Affine
from fluxdataqaqc import Data
from matplotlib.colors import LogNorm
from numpy import ma
from scipy import signal as sg

###############################################################################
# Configure logging
###############################################################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# You can send logs to stdout, a file, or elsewhere. Here we just use StreamHandler:
stream_handler = logging.StreamHandler()
# Customize the log format
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s\n"
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# Define the file path (absolute or relative). For instance:
log_file_path = "../logs/volk.log"

# Create a FileHandler and set the level
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Create a Formatter
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s\n"
)

# Set the formatter for the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

###############################################################################


def mask_fp_cutoff(f_array, cutoff=0.9):
    """
    Masks all values outside of the cutoff value

    Args:
        f_array (float) : 2D numpy array of point footprint contribution values (no units)
        cutoff (float) : Cutoff value for the cumulative sum of footprint values

    Returns:
        f_array (float) : 2D numpy array of footprint values, with nan == 0
    """
    val_array = f_array.flatten()
    sort_df = pd.DataFrame({"f": val_array}).sort_values(by="f").iloc[::-1]
    sort_df["cumsum_f"] = sort_df["f"].cumsum()

    sort_group = sort_df.groupby("f", as_index=True).mean()
    diff = abs(sort_group["cumsum_f"] - cutoff)
    sum_cutoff = diff.idxmin()
    f_array = np.where(f_array >= sum_cutoff, f_array, np.nan)
    f_array[~np.isfinite(f_array)] = 0.00000000e000

    logger.debug(f"mask_fp_cutoff: applied cutoff={cutoff}, sum_cutoff={sum_cutoff}")
    return f_array


def find_transform(xs, ys):
    """
    Returns the affine transform for 2d arrays xs and ys

    Args:
        xs (float) : 2D numpy array of x-coordinates
        ys (float) : 2D numpy array of y-coordinates

    Returns:
        aff_transform : affine.Affine object
    """

    shape = xs.shape

    # Choose points to calculate affine transform
    y_points = [0, 0, shape[0] - 1]
    x_points = [0, shape[0] - 1, shape[1] - 1]
    in_xy = np.float32([[i, j] for i, j in zip(x_points, y_points)])
    out_xy = np.float32([[xs[i, j], ys[i, j]] for i, j in zip(y_points, x_points)])

    # Calculate affine transform
    aff_transform = Affine(*cv2.getAffineTransform(in_xy, out_xy).flatten())
    logger.debug("Affine transform calculated.")
    return aff_transform


def ffp_climatology(
    zm=None,
    z0=None,
    umean=None,
    h=None,
    ol=None,
    sigmav=None,
    ustar=None,
    wind_dir=None,
    domain=None,
    dx=None,
    dy=None,
    nx=None,
    ny=None,
    rs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    rslayer=0,
    smooth_data=1,
    crop=False,
    pulse=None,
    verbosity=2,
    fig=False,
    **kwargs,
):
    """
    Derive a flux footprint estimate based on the simple parameterisation FFP
    See Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015:
    The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
    Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
    contact: n.kljun@swansea.ac.uk

    This function calculates footprints within a fixed physical domain for a series of
    time steps, rotates footprints into the corresponding wind direction and aggregates
    all footprints to a footprint climatology. The percentage of source area is
    calculated for the footprint climatology.
    For determining the optimal extent of the domain (large enough to include footprints)
    use calc_footprint_FFP.py.

    FFP Input
        All vectors need to be of equal length (one value for each time step)
        zm       = Measurement height above displacement height (i.e. z-d) [m]
                   usually a scalar, but can also be a vector
        z0       = Roughness length [m] - enter [None] if not known
                   usually a scalar, but can also be a vector
        umean    = Vector of mean wind speed at zm [ms-1] - enter [None] if not known
                   Either z0 or umean is required. If both are given,
                   z0 is selected to calculate the footprint
        h        = Vector of boundary layer height [m]
        ol       = Vector of Obukhov length [m]
        sigmav   = Vector of standard deviation of lateral velocity fluctuations [ms-1]
        ustar    = Vector of friction velocity [ms-1]
        wind_dir = Vector of wind direction in degrees (of 360) for rotation of the footprint

        Optional input:
        domain       = Domain size as an array of [xmin xmax ymin ymax] [m].
                       Footprint will be calculated for a measurement at [0 0 zm] m
                       Default is smallest area including the r% footprint or [-1000 1000 -1000 1000]m,
                       whichever smallest (80% footprint if r not given).
        dx, dy       = Cell size of domain [m]
                       Small dx, dy results in higher spatial resolution and higher computing time
                       Default is dx = dy = 2 m. If only dx is given, dx=dy.
        nx, ny       = Two integer scalars defining the number of grid elements in x and y
                       Large nx/ny result in higher spatial resolution and higher computing time
                       Default is nx = ny = 1000. If only nx is given, nx=ny.
                       If both dx/dy and nx/ny are given, dx/dy is given priority if the domain is also specified.
        rs           = Percentage of source area for which to provide contours, must be between 10% and 90%.
                       Can be either a single value (e.g., "80") or a list of values (e.g., "[10, 20, 30]")
                       Expressed either in percentages ("80") or as fractions of 1 ("0.8").
                       Default is [10:10:80]. Set to "None" for no output of percentages
        rslayer      = Calculate footprint even if zm within roughness sublayer: set rslayer = 1
                       Note that this only gives a rough estimate of the footprint as the model is not
                       valid within the roughness sublayer. Default is 0 (i.e. no footprint for within RS).
                       z0 is needed for estimation of the RS.
        smooth_data  = Apply convolution filter to smooth footprint climatology if smooth_data=1 (default)
        crop         = Crop output area to size of the 80% footprint or the largest r given if crop=1
        pulse        = Display progress of footprint calculations every pulse-th footprint (e.g., "100")
        verbosity    = Level of verbosity at run time: 0 = completely silent, 1 = notify only of fatal errors,
                       2 = all notifications
        fig          = Plot an example figure of the resulting footprint (on the screen): set fig = 1.
                       Default is 0 (i.e. no figure).

    FFP output
        FFP      = Structure array with footprint climatology data for measurement at [0 0 zm] m
        x_2d	    = x-grid of 2-dimensional footprint [m]
        y_2d	    = y-grid of 2-dimensional footprint [m]
        fclim_2d = Normalised footprint function values of footprint climatology [m-2]
        rs       = Percentage of footprint as in input, if provided
        fr       = Footprint value at r, if r is provided
        xr       = x-array for contour line of r, if r is provided
        yr       = y-array for contour line of r, if r is provided
        n        = Number of footprints calculated and included in footprint climatology
        flag_err = 0 if no error, 1 in case of error, 2 if not all contour plots (rs%) within specified domain,
                   3 if single data points had to be removed (outside validity)

    Created: 19 May 2016 natascha kljun
    Converted from matlab to python, together with Gerardo Fratini, LI-COR Biosciences Inc.
    version: 1.4
    last change: 11/12/2019 Gerardo Fratini, ported to Python 3.x
    Copyright (C) 2015,2016,2017,2018,2019,2020 Natascha Kljun
    """

    # ===========================================================================
    # Get kwargs
    show_heatmap = kwargs.get("show_heatmap", True)

    # ===========================================================================
    # Input check
    flag_err = 0

    # Check existence of required input pars
    if None in [zm, h, ol, sigmav, ustar] or (z0 is None and umean is None):
        raise_ffp_exception(1, verbosity)

    # Convert all input items to lists
    if not isinstance(zm, list):
        zm = [zm]
    if not isinstance(h, list):
        h = [h]
    if not isinstance(ol, list):
        ol = [ol]
    if not isinstance(sigmav, list):
        sigmav = [sigmav]
    if not isinstance(ustar, list):
        ustar = [ustar]
    if not isinstance(wind_dir, list):
        wind_dir = [wind_dir]
    if not isinstance(z0, list):
        z0 = [z0]
    if not isinstance(umean, list):
        umean = [umean]

    # Check that all lists have same length, if not raise an error and exit
    ts_len = len(ustar)
    logger.debug(f"input len is {ts_len}")
    if any(len(lst) != ts_len for lst in [sigmav, wind_dir, h, ol]):
        # at least one list has a different length, exit with error message
        raise_ffp_exception(11, verbosity)

    # Special treatment for zm, which is allowed to have length 1 for any
    # length >= 1 of all other parameters
    if all(val is None for val in zm):
        raise_ffp_exception(12, verbosity)
    if len(zm) == 1:
        raise_ffp_exception(17, verbosity)
        zm = [zm[0] for i in range(ts_len)]

    # Resolve ambiguity if both z0 and umean are passed (defaults to using z0)
    # If at least one value of z0 is passed, use z0 (by setting umean to None)
    if not all(val is None for val in z0):
        raise_ffp_exception(13, verbosity)
        umean = [None for i in range(ts_len)]
        # If only one value of z0 was passed, use that value for all footprints
        if len(z0) == 1:
            z0 = [z0[0] for i in range(ts_len)]
    elif len(umean) == ts_len and not all(val is None for val in umean):
        raise_ffp_exception(14, verbosity)
        z0 = [None for i in range(ts_len)]
    else:
        raise_ffp_exception(15, verbosity)

    # Rename lists as now the function expects time series of inputs
    ustars, sigmavs, hs, ols, wind_dirs, zms, z0s, umeans = (
        ustar,
        sigmav,
        h,
        ol,
        wind_dir,
        zm,
        z0,
        umean,
    )

    logger.debug(
        f"variables ustars, sigmavs, hs, ols, wind_dirs, zms, z0s, umeans input: {ustars}, {sigmavs}, {hs}, {ols}, {wind_dirs}, {zms}, {z0s}, {umeans}"
    )

    # ===========================================================================
    # Handle rs
    if rs is not None:
        # Check that rs is a list, otherwise make it a list
        if isinstance(rs, numbers.Number):
            if 0.9 < rs <= 1 or 90 < rs <= 100:
                rs = 0.9
            rs = [rs]
        if not isinstance(rs, list):
            raise_ffp_exception(18, verbosity)

        # If rs is passed as percentages, normalize to fractions of one
        if np.max(rs) >= 1:
            rs = [x / 100.0 for x in rs]

        # Eliminate any values beyond 0.9 (90%) and inform user
        if np.max(rs) > 0.9:
            raise_ffp_exception(19, verbosity)
            rs = [item for item in rs if item <= 0.9]

        # Sort levels in ascending order
        rs = list(np.sort(rs))

    # ===========================================================================
    # Define computational domain
    # Check passed values and make some smart assumptions
    if isinstance(dx, numbers.Number) and dy is None:
        dy = dx
    if isinstance(dy, numbers.Number) and dx is None:
        dx = dy
    if not all(isinstance(item, numbers.Number) for item in [dx, dy]):
        dx = dy = None
    if isinstance(nx, int) and ny is None:
        ny = nx
    if isinstance(ny, int) and nx is None:
        nx = ny
    if not all(isinstance(item, int) for item in [nx, ny]):
        nx = ny = None
    if not isinstance(domain, list) or len(domain) != 4:
        domain = None

    if all(item is None for item in [dx, nx, domain]):
        # If nothing is passed, default domain is a square of 2 Km size centered
        # at the tower with pizel size of 2 meters (hence a 1000x1000 grid)
        domain = [-1000.0, 1000.0, -1000.0, 1000.0]
        dx = dy = 2.0
        nx = ny = 1000
    elif domain is not None:
        # If domain is passed, it takes the precendence over anything else
        if dx is not None:
            # If dx/dy is passed, takes precendence over nx/ny
            nx = int((domain[1] - domain[0]) / dx)
            ny = int((domain[3] - domain[2]) / dy)
        else:
            # If dx/dy is not passed, use nx/ny (set to 1000 if not passed)
            if nx is None:
                nx = ny = 1000
            # If dx/dy is not passed, use nx/ny
            dx = (domain[1] - domain[0]) / float(nx)
            dy = (domain[3] - domain[2]) / float(ny)
    elif dx is not None and nx is not None:
        # If domain is not passed but dx/dy and nx/ny are, define domain
        domain = [-nx * dx / 2, nx * dx / 2, -ny * dy / 2, ny * dy / 2]
    elif dx is not None:
        # If domain is not passed but dx/dy is, define domain and nx/ny
        domain = [-1000, 1000, -1000, 1000]
        nx = int((domain[1] - domain[0]) / dx)
        ny = int((domain[3] - domain[2]) / dy)
    elif nx is not None:
        # If domain and dx/dy are not passed but nx/ny is, define domain and dx/dy
        domain = [-1000, 1000, -1000, 1000]
        dx = (domain[1] - domain[0]) / float(nx)
        dy = (domain[3] - domain[2]) / float(nx)

    # Put domain into more convenient vars
    xmin, xmax, ymin, ymax = domain
    logger.info(f"Domain: {domain}")
    # Define rslayer if not passed
    if rslayer is None:
        rslayer = 0

    # Define smooth_data if not passed
    if smooth_data is None:
        smooth_data = 1

    # Define crop if not passed
    if crop is None:
        crop = 0

    # Define pulse if not passed
    if pulse is None:
        if ts_len <= 20:
            pulse = 1
        else:
            pulse = int(ts_len / 20)

    # Define fig if not passed
    if fig is None:
        fig = 0

    logger.debug(
        f"parameters rslayer, smooth_data, crop, pulse, fig: {rslayer}, {smooth_data}, {crop}, {pulse}, {fig}"
    )
    # ===========================================================================
    # Model parameters
    a = 1.4524
    b = -1.9914
    c = 1.4622
    d = 0.1359
    ac = 2.17
    bc = 1.66
    cc = 20.0

    oln = 5000  # limit to L for neutral scaling
    k = 0.4  # von Karman

    # ===========================================================================
    # Define physical domain in cartesian and polar coordinates
    # Cartesian coordinates
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    x_2d, y_2d = np.meshgrid(x, y)
    logger.debug(f"x_2d: {x_2d}, y_2d: {y_2d}")
    # Polar coordinates
    # Set theta such that North is pointing upwards and angles increase clockwise
    rho = np.sqrt(x_2d**2 + y_2d**2)
    theta = np.arctan2(x_2d, y_2d)
    logger.debug(f"rho: {rho}, theta: {theta}")
    # initialize raster for footprint climatology
    fclim_2d = np.zeros(x_2d.shape)

    # ===========================================================================
    # Loop on time series

    # Initialize logic array valids to those 'timestamps' for which all inputs are
    # at least present (but not necessarily phisically plausible)
    valids = [
        True if not any([val is None for val in vals]) else False
        for vals in zip(ustars, sigmavs, hs, ols, wind_dirs, zms)
    ]

    logger.debug(f"List of valids {valids}")

    if verbosity > 1:
        logger.info("Beginning footprint calculations...")

    for ix, (ustar, sigmav, h, ol, wind_dir, zm, z0, umean) in enumerate(
        zip(ustars, sigmavs, hs, ols, wind_dirs, zms, z0s, umeans)
    ):
        # Counter
        if verbosity > 1 and ix % pulse == 0:
            print("Calculating footprint ", ix + 1, " of ", ts_len)
            logger.info(f"Calculating footprint {ix + 1} of {ts_len}")

        valids[ix] = check_ffp_inputs(
            ustar, sigmav, h, ol, wind_dir, zm, z0, umean, rslayer, verbosity
        )

        logger.debug(f"valids of {ix} are {valids[ix]}")

        # If inputs are not valid, skip current footprint
        if not valids[ix]:
            raise_ffp_exception(16, verbosity)
        else:
            # ===========================================================================
            # Rotate coordinates into wind direction
            if wind_dir is not None:
                rotated_theta = theta - wind_dir * np.pi / 180.0

                logger.debug(f"rotated_theta: {rotated_theta}")
            # ===========================================================================
            # Create real scale crosswind integrated footprint and dummy for
            # rotated scaled footprint
            fstar_ci_dummy = np.zeros(x_2d.shape)
            f_ci_dummy = np.zeros(x_2d.shape)
            xstar_ci_dummy = np.zeros(x_2d.shape)
            px = np.ones(x_2d.shape)

            if z0 is not None:
                # Use z0
                if ol <= 0 or ol >= oln:
                    xx = (1 - 19.0 * zm / ol) ** 0.25
                    psi_f = (
                        np.log((1 + xx**2) / 2.0)
                        + 2.0 * np.log((1 + xx) / 2.0)
                        - 2.0 * np.arctan(xx)
                        + np.pi / 2
                    )
                    logger.debug(f"psi_f = {psi_f}, xx = {xx}")
                elif ol > 0 and ol < oln:
                    psi_f = -5.3 * zm / ol
                    # print(psi_f, zm, ol)
                    logger.debug(f"psi_f = {psi_f}, zm = {zm}, ol = {ol}")

                if (np.log(zm / z0) - psi_f) > 0:
                    logger.debug("Calculating xstar_ci_dummy...")
                    xstar_ci_dummy = (
                        rho
                        * np.cos(rotated_theta)
                        / zm
                        * (1.0 - (zm / h))
                        / (np.log(zm / z0) - psi_f)
                    )
                    px = np.where(xstar_ci_dummy > d)
                    fstar_ci_dummy[px] = (
                        a
                        * (xstar_ci_dummy[px] - d) ** b
                        * np.exp(-c / (xstar_ci_dummy[px] - d))
                    )
                    f_ci_dummy[px] = (
                        fstar_ci_dummy[px]
                        / zm
                        * (1.0 - (zm / h))
                        / (np.log(zm / z0) - psi_f)
                    )

                else:
                    flag_err = 3
                    valids[ix] = 0
                    logger.debug("flag err 3")
            else:
                # Use umean if z0 not available
                xstar_ci_dummy = (
                    rho
                    * np.cos(rotated_theta)
                    / zm
                    * (1.0 - (zm / h))
                    / (umean / ustar * k)
                )
                px = np.where(xstar_ci_dummy > d)
                fstar_ci_dummy[px] = (
                    a
                    * (xstar_ci_dummy[px] - d) ** b
                    * np.exp(-c / (xstar_ci_dummy[px] - d))
                )
                f_ci_dummy[px] = (
                    fstar_ci_dummy[px] / zm * (1.0 - (zm / h)) / (umean / ustar * k)
                )

            # ===========================================================================
            # Calculate dummy for scaled sig_y* and real scale sig_y
            sigystar_dummy = np.zeros(x_2d.shape)
            sigystar_dummy[px] = ac * np.sqrt(
                bc
                * np.abs(xstar_ci_dummy[px]) ** 2
                / (1 + cc * np.abs(xstar_ci_dummy[px]))
            )

            if abs(ol) > oln:
                ol = -1e6
            if ol <= 0:  # convective
                scale_const = 1e-5 * abs(zm / ol) ** (-1) + 0.80
            elif ol > 0:  # stable
                scale_const = 1e-5 * abs(zm / ol) ** (-1) + 0.55
            if scale_const > 1:
                scale_const = 1.0

            sigy_dummy = np.zeros(x_2d.shape)
            sigy_dummy[px] = sigystar_dummy[px] / scale_const * zm * sigmav / ustar
            sigy_dummy[sigy_dummy < 0] = np.nan

            # ===========================================================================
            # Calculate real scale f(x,y)
            f_2d = np.zeros(x_2d.shape)
            f_2d[px] = (
                f_ci_dummy[px]
                / (np.sqrt(2 * np.pi) * sigy_dummy[px])
                * np.exp(
                    -((rho[px] * np.sin(rotated_theta[px])) ** 2)
                    / (2.0 * sigy_dummy[px] ** 2)
                )
            )

            # ===========================================================================
            # Add to footprint climatology raster
            fclim_2d = fclim_2d + f_2d
            logger.debug(f"fclim_2d: {fclim_2d}, f_2d: {f_2d}")
    # ===========================================================================
    # Continue if at least one valid footprint was calculated
    n = sum(valids)
    logger.debug(f"n: {n}")
    vs = None
    clevs = None
    if n == 0:
        logger.warning("No valid footprints were calculated.")
        print("No footprint calculated")
        flag_err = 1
    else:

        # ===========================================================================
        # Normalize and smooth footprint climatology
        fclim_2d = fclim_2d / n

        if smooth_data is not None:
            skernel = np.matrix("0.05 0.1 0.05; 0.1 0.4 0.1; 0.05 0.1 0.05")
            fclim_2d = sg.convolve2d(fclim_2d, skernel, mode="same")
            fclim_2d = sg.convolve2d(fclim_2d, skernel, mode="same")

        # ===========================================================================
        # Derive footprint ellipsoid incorporating R% of the flux, if requested,
        # starting at peak value.
        if rs is not None:
            clevs = get_contour_levels(fclim_2d, dx, dy, rs)
            frs = [item[2] for item in clevs]
            xrs = []
            yrs = []
            for ix, fr in enumerate(frs):
                xr, yr = get_contour_vertices(x_2d, y_2d, fclim_2d, fr)
                if xr is None:
                    frs[ix] = None
                    flag_err = 2
                xrs.append(xr)
                yrs.append(yr)
        else:
            if crop:
                rs_dummy = 0.8  # crop to 80%
                clevs = get_contour_levels(fclim_2d, dx, dy, rs_dummy)
                xrs = []
                yrs = []
                xrs, yrs = get_contour_vertices(x_2d, y_2d, fclim_2d, clevs[0][2])

        # ===========================================================================
        # Crop domain and footprint to the largest rs value
        if crop:
            xrs_crop = [x for x in xrs if x is not None]
            yrs_crop = [x for x in yrs if x is not None]
            if rs is not None:
                dminx = np.floor(min(xrs_crop[-1]))
                dmaxx = np.ceil(max(xrs_crop[-1]))
                dminy = np.floor(min(yrs_crop[-1]))
                dmaxy = np.ceil(max(yrs_crop[-1]))
            else:
                dminx = np.floor(min(xrs_crop))
                dmaxx = np.ceil(max(xrs_crop))
                dminy = np.floor(min(yrs_crop))
                dmaxy = np.ceil(max(yrs_crop))

            if dminy >= ymin and dmaxy <= ymax:
                jrange = np.where((y_2d[:, 0] >= dminy) & (y_2d[:, 0] <= dmaxy))[0]
                jrange = np.concatenate(([jrange[0] - 1], jrange, [jrange[-1] + 1]))
                jrange = jrange[np.where((jrange >= 0) & (jrange <= y_2d.shape[0]))[0]]
            else:
                jrange = np.linspace(0, 1, y_2d.shape[0] - 1)

            if dminx >= xmin and dmaxx <= xmax:
                irange = np.where((x_2d[0, :] >= dminx) & (x_2d[0, :] <= dmaxx))[0]
                irange = np.concatenate(([irange[0] - 1], irange, [irange[-1] + 1]))
                irange = irange[np.where((irange >= 0) & (irange <= x_2d.shape[1]))[0]]
            else:
                irange = np.linspace(0, 1, x_2d.shape[1] - 1)

            jrange = [[it] for it in jrange]
            x_2d = x_2d[jrange, irange]
            y_2d = y_2d[jrange, irange]
            fclim_2d = fclim_2d[jrange, irange]

        # ===========================================================================
        # Plot footprint
        if fig:
            fig_out, ax = plot_footprint(
                x_2d=x_2d, y_2d=y_2d, fs=fclim_2d, show_heatmap=show_heatmap, clevs=frs
            )
        else:
            fig_out = None

    # ===========================================================================
    # Fill output structure
    if rs is not None:
        return {
            "x_2d": x_2d,
            "y_2d": y_2d,
            "fclim_2d": fclim_2d,
            "rs": rs,
            "fr": frs,
            "xr": xrs,
            "yr": yrs,
            "n": n,
            "flag_err": flag_err,
            "fig": fig_out,
        }
    else:
        return {
            "x_2d": x_2d,
            "y_2d": y_2d,
            "fclim_2d": fclim_2d,
            "n": n,
            "flag_err": flag_err,
        }  # 'fig': fig_out


def check_ffp_inputs(ustar, sigmav, h, ol, wind_dir, zm, z0, umean, rslayer, verbosity):
    """
    Validates input parameters for physical plausibility and consistency for a footprint model.

    This function checks the validity of the input parameters provided for a footprint
    model. It ensures that the parameters adhere to the required physical constraints
    and consistency rules. The function raises exceptions when specific conditions
    are violated and can also operate with a specified verbosity level for raising
    exceptions.

    Args:
        ustar (float or array-like): Friction velocity. The value must be greater than 0.1
            for all elements.
        sigmav (float or array-like): Standard deviation of vertical wind speed. The value
            must be greater than 0 for all elements.
        h (float): Boundary layer height. The value must be greater than 10.
        ol (float or array-like): Obukhov length. The computed ratio of `zm / ol` must
            not be less than -15.5.
        wind_dir (float or array-like): Wind direction in degrees. Must be in the
            range [0, 360] for all elements.
        zm (float): Measurement height above ground. Must be positive and less than the
            boundary layer height (`h`).
        z0 (float, optional): Surface roughness length. If specified, must be greater
            than 0.
        umean (float, optional): Mean wind speed. Required if `z0` is specified.
        rslayer (int): Stability regime flag. Can modify the criteria for how `zm` and `z0`
            are validated.
        verbosity (int): Verbosity level for exception handling. Higher values may allow
            more detailed error reporting.

    Returns:
        bool: True if all inputs pass the validity checks, otherwise raises an exception.

    Raises:
        Exception: Raised with specific error codes for violations of physical validity:
            - Code 2: Measurement height (`zm`) is non-positive.
            - Code 3: Surface roughness length (`z0`) is non-positive without `umean`.
            - Code 4: Boundary layer height (`h`) is too small (≤ 10 m).
            - Code 5: Measurement height (`zm`) exceeds boundary layer height (`h`).
            - Code 6 or 20: Measurement height (`zm`) is inconsistent with surface
              roughness length (`z0`) given the stability layer regime.
            - Code 7: Ratio of measurement height to Obukhov length (`zm / ol`) is
              less than -15.5.
            - Code 8: Standard deviation of vertical wind speed (`sigmav`) is
              non-positive.
            - Code 9: Friction velocity (`ustar`) is too small (≤ 0.1).
            - Code 10: Wind direction values are out of the valid range ([0, 360]).
    """
    # Check passed values for physical plausibility and consistency
    if zm <= 0.0:
        raise_ffp_exception(2, verbosity)
        logger.debug(f"zm <= 0.0   zm={zm}")
        return False
    if z0 is not None and umean is None and z0 <= 0.0:
        raise_ffp_exception(3, verbosity)
        logger.debug("z0 is not None and umean is None and z0 <= 0.0")
        return False
    if h <= 10.0:
        raise_ffp_exception(4, verbosity)
        logger.debug(f"h <= 10.0  h={h}")
        return False
    if zm > h:
        raise_ffp_exception(5, verbosity)
        logger.debug(f"zm > h  zm={zm}, h={h}")
        return False
    if z0 is not None and umean is None and zm <= 12.5 * z0:
        logger.debug(f"zm <= 12.5 * z0   zm={zm}, z0={z0}")
        if rslayer == 1:
            raise_ffp_exception(6, verbosity)
            logger.debug("rslayer == 1")
        else:
            raise_ffp_exception(20, verbosity)
            logger.debug("rslayer != 1")
            return False
    if (float(zm) / ol).any() <= -15.5:
        raise_ffp_exception(7, verbosity)
        logger.debug(f"float(zm) / ol).any() <= -15.5  zm={zm}, ol={ol}")
        return False
    if sigmav.any() <= 0:
        raise_ffp_exception(8, verbosity)
        logger.debug(f"sigmav.any() <= 0  sigmav={sigmav}")
        return False
    if ustar.any() <= 0.1:
        raise_ffp_exception(9, verbosity)
        logger.debug(f"ustar.any() <= 0.1 {ustar}")
        return False
    if wind_dir.any() > 360:
        raise_ffp_exception(10, verbosity)
        logger.debug(f"wind_dir.any() > 360   wind_dir={wind_dir}")
        return False
    if wind_dir.any() < 0:
        logger.debug(f"wind_dir.any() < 0  wind_dir={wind_dir}")
        raise_ffp_exception(10, verbosity)
        return False
    return True


def get_contour_levels(f, dx, dy, rs=None):
    """Contour levels of f at percentages of f-integral given by rs"""

    # Check input and resolve to default levels in needed
    if not isinstance(rs, (int, float, list)):
        rs = list(np.linspace(0.10, 0.90, 9))
    if isinstance(rs, (int, float)):
        rs = [rs]

    # Levels
    pclevs = np.empty(len(rs))
    pclevs[:] = np.nan
    ars = np.empty(len(rs))
    ars[:] = np.nan
    logger.debug(pclevs)

    sf = np.sort(f, axis=None)[::-1]
    msf = ma.masked_array(
        sf, mask=(np.isnan(sf) | np.isinf(sf))
    )  # Masked array for handling potential nan
    csf = msf.cumsum().filled(np.nan) * dx * dy
    for ix, r in enumerate(rs):
        dcsf = np.abs(csf - r)
        pclevs[ix] = sf[np.nanargmin(dcsf)]
        ars[ix] = csf[np.nanargmin(dcsf)]

    return [(round(r, 3), ar, pclev) for r, ar, pclev in zip(rs, ars, pclevs)]


def get_contour_vertices(x, y, f, lev):
    # import matplotlib._contour as cntr
    import matplotlib.pyplot as plt

    cs = plt.contour(x, y, f, [lev])
    plt.close()
    segs = cs.allsegs[0][0]
    logger.debug(segs)
    xr = [vert[0] for vert in segs]
    yr = [vert[1] for vert in segs]
    # Set contour to None if it's found to reach the physical domain
    if (
        x.min() >= min(segs[:, 0])
        or max(segs[:, 0]) >= x.max()
        or y.min() >= min(segs[:, 1])
        or max(segs[:, 1]) >= y.max()
    ):
        return [None, None]

    return [xr, yr]  # x,y coords of contour points.


def plot_footprint(
    x_2d,
    y_2d,
    fs,
    clevs=None,
    show_heatmap=True,
    normalize=None,
    colormap=None,
    line_width=0.5,
    iso_labels=None,
):
    """
    Plots footprint data and optionally overlays contours, isopleths, and heatmaps for one
    or more footprints over a 2D grid.

    This function visualizes footprint data by creating either a heatmap, contour plot, or
    both, depending on the provided parameters. It supports multiple footprints, each
    represented by a unique contour color when overlaid together. Customization options
    are available for contour levels, colormap, line width, isopleth labels, and normalization.

    Args:
        x_2d (numpy.ndarray): 2D array representing the x-coordinates of the grid.
        y_2d (numpy.ndarray): 2D array representing the y-coordinates of the grid.
        fs (numpy.ndarray or list of numpy.ndarray): Footprint data as a 2D array for single
            footprint or list of 2D arrays for multiple footprints.
        clevs (list[float] or None): Contour levels for the plot. If None, no contours are
            drawn. Defaults to None.
        show_heatmap (bool): If True, displays a heatmap for the footprint. Defaults to True.
        normalize (str or None): Normalization method for the heatmap. Specify "log" for
            logarithmic normalization, or None for no normalization. Defaults to None.
        colormap (matplotlib.colors.Colormap or None): Colormap to use for plotting the
            heatmap or contours. Defaults to None, which uses `cm.jet`.
        line_width (float): Line width for contour plotting. Defaults to 0.5.
        iso_labels (list[tuple[float]] or None): Labels for the isopleths as percentages.
            If None, no isopleth labels are added. Defaults to None.

    Returns:
        tuple: Contains the following:
            - fig (matplotlib.figure.Figure): The figure object containing the plot.
            - ax (matplotlib.axes.Axes): The axes object of the plot.
    """

    # If input is a list of footprints, don't show footprint but only contours,
    # with different colors
    if isinstance(fs, list):
        show_heatmap = False
    else:
        fs = [fs]

    if colormap is None:
        colormap = cm.jet
    # Define colors for each contour set
    cs = [colormap(ix) for ix in np.linspace(0, 1, len(fs))]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 8))
    # fig.patch.set_facecolor('none')
    # ax.patch.set_facecolor('none')

    if clevs is not None:
        # Temporary patch for pyplot.contour requiring contours to be in ascending orders
        clevs = clevs[::-1]

        # Eliminate contour levels that were set to None
        # (e.g. because they extend beyond the defined domain)
        clevs = [clev for clev in clevs if clev is not None]

        # Plot contour levels of all passed footprints
        # Plot isopleth
        levs = [clev for clev in clevs]
        for f, c in zip(fs, cs):
            cc = [c] * len(levs)
            if show_heatmap:
                cp = ax.contour(x_2d, y_2d, f, levs, colors="w", linewidths=line_width)
            else:
                cp = ax.contour(x_2d, y_2d, f, levs, colors=cc, linewidths=line_width)
            # Isopleth Labels
            if iso_labels is not None:
                pers = [str(int(clev[0] * 100)) + "%" for clev in clevs]
                fmt = {}
                for l, s in zip(cp.levels, pers):
                    fmt[l] = s
                plt.clabel(cp, cp.levels[:], inline=1, fmt=fmt, fontsize=7)

    # plot footprint heatmap if requested and if only one footprint is passed
    if show_heatmap:
        if normalize == "log":
            norm = LogNorm()
        else:
            norm = None

        xmin = np.nanmin(x_2d)
        xmax = np.nanmax(x_2d)
        ymin = np.nanmin(y_2d)
        ymax = np.nanmax(y_2d)
        for f in fs:
            im = ax.imshow(
                f[:, :],
                cmap=colormap,
                extent=(xmin, xmax, ymin, ymax),
                norm=norm,
                origin="lower",
                aspect=1,
            )
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        # Colorbar
        cbar = fig.colorbar(im, shrink=1.0, format="%.3e")
        # cbar.set_label('Flux contribution', color = 'k')
    plt.show()

    return fig, ax


exTypes = {
    "message": "Message",
    "alert": "Alert",
    "error": "Error",
    "fatal": "Fatal error",
}

exceptions = [
    {
        "code": 1,
        "type": exTypes["fatal"],
        "msg": "At least one required parameter is missing. Please enter all "
        "required inputs. Check documentation for details.",
    },
    {
        "code": 2,
        "type": exTypes["error"],
        "msg": "zm (measurement height) must be larger than zero.",
    },
    {
        "code": 3,
        "type": exTypes["error"],
        "msg": "z0 (roughness length) must be larger than zero.",
    },
    {
        "code": 4,
        "type": exTypes["error"],
        "msg": "h (BPL height) must be larger than 10 m.",
    },
    {
        "code": 5,
        "type": exTypes["error"],
        "msg": "zm (measurement height) must be smaller than h (PBL height).",
    },
    {
        "code": 6,
        "type": exTypes["alert"],
        "msg": "zm (measurement height) should be above roughness sub-layer (12.5*z0).",
    },
    {
        "code": 7,
        "type": exTypes["error"],
        "msg": "zm/ol (measurement height to Obukhov length ratio) must be equal or larger than -15.5.",
    },
    {
        "code": 8,
        "type": exTypes["error"],
        "msg": "sigmav (standard deviation of crosswind) must be larger than zero.",
    },
    {
        "code": 9,
        "type": exTypes["error"],
        "msg": "ustar (friction velocity) must be >=0.1.",
    },
    {
        "code": 10,
        "type": exTypes["error"],
        "msg": "wind_dir (wind direction) must be >=0 and <=360.",
    },
    {
        "code": 11,
        "type": exTypes["fatal"],
        "msg": "Passed data arrays (ustar, zm, h, ol) don't all have the same length.",
    },
    {
        "code": 12,
        "type": exTypes["fatal"],
        "msg": "No valid zm (measurement height above displacement height) passed.",
    },
    {
        "code": 13,
        "type": exTypes["alert"],
        "msg": "Using z0, ignoring umean if passed.",
    },
    {"code": 14, "type": exTypes["alert"], "msg": "No valid z0 passed, using umean."},
    {"code": 15, "type": exTypes["fatal"], "msg": "No valid z0 or umean array passed."},
    {
        "code": 16,
        "type": exTypes["error"],
        "msg": "At least one required input is invalid. Skipping current footprint.",
    },
    {
        "code": 17,
        "type": exTypes["alert"],
        "msg": "Only one value of zm passed. Using it for all footprints.",
    },
    {
        "code": 18,
        "type": exTypes["fatal"],
        "msg": "if provided, rs must be in the form of a number or a list of numbers.",
    },
    {
        "code": 19,
        "type": exTypes["alert"],
        "msg": "rs value(s) larger than 90% were found and eliminated.",
    },
    {
        "code": 20,
        "type": exTypes["error"],
        "msg": "zm (measurement height) must be above roughness sub-layer (12.5*z0).",
    },
]


def raise_ffp_exception(code, verbosity):
    """
    Raises exceptions based on provided error code and verbosity level, with appropriate logging
    and messaging defined by the exception type.

    The function utilizes an external `exceptions` list to locate the matching exception type
    and message using the provided error code. Depending on the verbosity level and exception
    type, it either logs or prints warnings and errors or raises a critical exception, signaling
    an immediate program halt. The exception messaging can be customized based on execution
    needs and conditions.

    Args:
        code: An integer representing the error code used to fetch details about a specific
            exception type and its corresponding message.
        verbosity: An integer controlling the level of detail in exception logging and messaging.
            Higher verbosity levels include more detailed output, with level 0 suppressing most
            message outputs.

    Raises:
        Exception: Raised when the exception type corresponds to a fatal error. The execution of
            the program is forcibly aborted in such cases.
    """

    ex = [it for it in exceptions if it["code"] == code][0]
    string = ex["type"] + "(" + str(ex["code"]).zfill(4) + "):\n " + ex["msg"]

    if verbosity > 0:
        logger.warning("")

    if ex["type"] == exTypes["fatal"]:
        if verbosity > 0:
            string = string + "\n FFP_fixed_domain execution aborted."
            logger.error(string)
        else:
            string = ""
        raise Exception(string)
    elif ex["type"] == exTypes["alert"]:
        string = string + "\n Execution continues."
        if verbosity > 1:
            print(string)
            logger.warning(string)
    elif ex["type"] == exTypes["error"]:
        string = string + "\n Execution continues."
        if verbosity > 1:
            print(string)
            logger.error(string)
    else:
        if verbosity > 1:
            print(string)
            logger.warning(string)


if __name__ == "__main__":

    # load initial flux data
    d = Data("US-CRT_config.ini")
    # adding variable names to Data instance name list for resampling
    d.variables["MO_LENGTH"] = "MO_LENGTH"
    d.variables["USTAR"] = "USTAR"
    d.variables["V_SIGMA"] = "V_SIGMA"
    # renaming columns (optional and only affects windspeed and wind direction names)
    df = d.df.rename(columns=d.inv_map)
    df = df.resample("h").mean()
    # get coords info from Data instance
    latitude = d.latitude
    longitude = d.longitude
    station_coord = (latitude, longitude)
    station = d.site_id
    # get EPSG code from lat,long, convert to UTM https://epsg.io/32617
    EPSG = int(
        32700
        - np.round((45 + latitude) / 90.0) * 100
        + np.round((183 + longitude) / 6.0)
    )
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{int(EPSG)}")
    (station_x, station_y) = transformer.transform(*station_coord)
    # check results, EPSG code should be 32617, lon should be near 304485 and lat 4611191
    h_c = 0.2  # Height of canopy [m]
    # Estimated displacement height [m]
    d = 10 ** (0.979 * np.log10(h_c) - 0.154)
    # Other model parameters
    zm_s = 2.0  # Measurement height [m] from AMF metadata
    h_s = 2000.0  # Height of atmos. boundary layer [m] - assumed
    dx = 3.0  # Model resolution [m]
    origin_d = 200.0  # Model bounds distance from origin [m]
    # from 7 AM to 8 PM only, modify if needed
    start_hr = 7
    end_hr = 20
    hours = np.arange(start_hr, end_hr + 1)

    # Loop through each day in the dataframe
    for date in df.index.date:

        # Subset dataframe to only values in day of year
        print(f"Date: {date}")
        temp_df = df[df.index.date == date]

        new_dat = None

        for indx, t in enumerate(hours):

            band = indx + 1
            print(f"Hour: {t}")

            try:
                temp_line = temp_df.loc[temp_df.index.hour == t, :]

                # Calculate footprint
                temp_ffp = ffp_climatology(
                    domain=[-origin_d, origin_d, -origin_d, origin_d],
                    dx=dx,
                    dy=dx,
                    zm=zm_s - d,
                    h=h_s,
                    rs=None,
                    z0=h_c * 0.123,
                    ol=temp_line["MO_LENGTH"].values,
                    sigmav=temp_line["V_SIGMA"].values,
                    ustar=temp_line["USTAR"].values,  # umean=temp_line['ws'].values,
                    wind_dir=temp_line["wd"].values,
                    crop=0,
                    fig=0,
                    verbosity=0,
                )
                ####verbosoity=2 prints out errors; if z0 triggers errors, use umean
                #    print(zm_s-d)

                f_2d = np.array(temp_ffp["fclim_2d"])
                x_2d = np.array(temp_ffp["x_2d"]) + station_x
                y_2d = np.array(temp_ffp["y_2d"]) + station_y
                f_2d = f_2d * dx**2

                # Calculate affine transform for given x_2d and y_2d
                affine_transform = find_transform(x_2d, y_2d)

                # Create data file if not already created
                if new_dat is None:
                    out_f = f"./{date}_{station}.tif"
                    print(f_2d.shape)
                    new_dat = rasterio.open(
                        out_f,
                        "w",
                        driver="GTiff",
                        dtype=rasterio.float64,
                        count=len(hours),
                        height=f_2d.shape[0],
                        width=f_2d.shape[1],
                        transform=affine_transform,
                        crs=pyproj.crs.CRS.from_epsg(int(EPSG)),
                        nodata=0.00000000e000,
                    )

            except Exception as e:

                print(f"Hour {t} footprint failed, band {band} not written.")

                temp_ffp = None

                continue

            # Mask out points that are below a % threshold (defaults to 90%)
            f_2d = mask_fp_cutoff(f_2d)

            # Write the new band
            new_dat.write(f_2d, indx + 1)

            # Update tags with metadata
            tag_dict = {
                "hour": f"{t * 100:04}",
                "wind_dir": temp_line["wd"].values,
                "total_footprint": np.nansum(f_2d),
            }

            new_dat.update_tags(indx + 1, **tag_dict)

        # Close dataset if it exists
        try:
            new_dat.close()
        except:
            continue

        print()

        # for esample just create a single day and exit
        break
