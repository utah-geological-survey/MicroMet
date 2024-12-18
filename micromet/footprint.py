import numbers
import matplotlib
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
import sys


class ffp_climatology(object):
    """
    Derive a flux footprint estimate based on the simple parameterisation FFP
    See Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015:
    The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
    Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
    contact: natascha.kljun@cec.lu.se

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
    version: 1.42
    last change: 11/12/2019 Gerardo Fratini, ported to Python 3.x
    Copyright (C) 2015 - 2024 Natascha Kljun
    """
    exTypes = {'message': 'Message',
               'alert': 'Alert',
               'error': 'Error',
               'fatal': 'Fatal error'}

    exceptions = [
        {'code': 1,
         'type': exTypes['fatal'],
         'msg': 'At least one required parameter is missing. Please enter all '
                'required inputs. Check documentation for details.'},
        {'code': 2,
         'type': exTypes['error'],
         'msg': 'zm (measurement height) must be larger than zero.'},
        {'code': 3,
         'type': exTypes['error'],
         'msg': 'z0 (roughness length) must be larger than zero.'},
        {'code': 4,
         'type': exTypes['error'],
         'msg': 'h (BPL height) must be larger than 10 m.'},
        {'code': 5,
         'type': exTypes['error'],
         'msg': 'zm (measurement height) must be smaller than h (PBL height).'},
        {'code': 6,
         'type': exTypes['alert'],
         'msg': 'zm (measurement height) should be above roughness sub-layer (12.5*z0).'},
        {'code': 7,
         'type': exTypes['error'],
         'msg': 'zm/ol (measurement height to Obukhov length ratio) must be equal or larger than -15.5.'},
        {'code': 8,
         'type': exTypes['error'],
         'msg': 'sigmav (standard deviation of crosswind) must be larger than zero.'},
        {'code': 9,
         'type': exTypes['error'],
         'msg': 'ustar (friction velocity) must be >=0.1.'},
        {'code': 10,
         'type': exTypes['error'],
         'msg': 'wind_dir (wind direction) must be >=0 and <=360.'},
        {'code': 11,
         'type': exTypes['fatal'],
         'msg': 'Passed data arrays (ustar, zm, h, ol) don\'t all have the same length.'},
        {'code': 12,
         'type': exTypes['fatal'],
         'msg': 'No valid zm (measurement height above displacement height) passed.'},
        {'code': 13,
         'type': exTypes['alert'],
         'msg': 'Using z0, ignoring umean if passed.'},
        {'code': 14,
         'type': exTypes['alert'],
         'msg': 'No valid z0 passed, using umean.'},
        {'code': 15,
         'type': exTypes['fatal'],
         'msg': 'No valid z0 or umean array passed.'},
        {'code': 16,
         'type': exTypes['error'],
         'msg': 'At least one required input is invalid. Skipping current footprint.'},
        {'code': 17,
         'type': exTypes['alert'],
         'msg': 'Only one value of zm passed. Using it for all footprints.'},
        {'code': 18,
         'type': exTypes['fatal'],
         'msg': 'if provided, rs must be in the form of a number or a list of numbers.'},
        {'code': 19,
         'type': exTypes['alert'],
         'msg': 'rs value(s) larger than 90% were found and eliminated.'},
        {'code': 20,
         'type': exTypes['error'],
         'msg': 'zm (measurement height) must be above roughness sub-layer (12.5*z0).'},
    ]


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



    def __init__(self, zm=None,
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
                    figure=False,
                    **kwargs):

        # Check existence of required input pars
        if None in [zm, h, ol, sigmav, ustar] or (z0 is None and umean is None):
            self.raise_ffp_exception(1, verbosity)

        self.zm = zm
        self.z0 = z0
        self.umean = umean
        self.h = h
        self.ol = ol
        self.sigmav = sigmav
        self.ustar = ustar
        self.wind_dir = wind_dir
        self.domain = domain
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.rs = rs
        self.rslayer = rslayer
        self.smooth_data = smooth_data
        self.crop = crop
        self.pulse = pulse
        self.verbosity = verbosity
        self.fig = figure
        self.kwargs = kwargs

        # Get kwargs
        self.show_heatmap = kwargs.get('show_heatmap', True)

        # Input check
        self.flag_err = 0

        self.makelists()
        self.handle_rs()

        self.computational_domain()
        self.check_inputs()
        self.define_domain()
        self.loop_over_time()
        self.results = self.get_results()
        if self.fig:
            self.plot_footprint()


    def makelists(self):
        # Convert all input items to lists

        if not isinstance(self.zm, list):
            self.zms = [self.zm]
        else:
            self.zms = self.zm

        if not isinstance(self.h, list):
            self.hs = [self.h]
        else:
            self.hs = self.h

        if not isinstance(self.ol, list):
            self.ols = [self.ol]
        else:
            self.ols = self.ol

        if not isinstance(self.sigmav, list):
            self.sigmavs = [self.sigmav]
        else:
            self.sigmavs = self.sigmav

        if not isinstance(self.ustar, list):
            self.ustars = [self.ustar]
        else:
            self.ustars = self.ustar

        if not isinstance(self.wind_dir, list):
            self.wind_dirs = [self.wind_dir]
        else:
            self.wind_dirs = self.wind_dir

        if not isinstance(self.z0, list):
            self.z0s = [self.z0]
        else:
            self.z0s = self.z0

        if not isinstance(self.umean, list):
            self.umeans = [self.umean]
        else:
            self.umeans = self.umean

        if not isinstance(self.ustars, list):
            self.ts_len = 1
        else:
            self.ts_len = len(self.ustars)

        if any(len(lst) != self.ts_len for lst in [self.sigmavs, self.wind_dirs, self.hs, self.ols]):
            # at least one list has a different length, exit with error message
            self.raise_ffp_exception(11, self.verbosity)

        # Special treatment for zm, which is allowed to have length 1 for any
        # length >= 1 of all other parameters
        if all(val is None for val in self.zms):
            self.zms = [self.zms[0]] * self.ts_len
            self.raise_ffp_exception(12, self.verbosity)

        if len(self.zms) == 1:
            self.zms = [self.zms[0]] * self.ts_len
            self.raise_ffp_exception(17, self.verbosity)
            self.zms = [self.zms[0] for i in range(self.ts_len)]

        # Resolve ambiguity if both z0 and umean are passed (defaults to using z0)
        # If at least one value of z0 is passed, use z0 (by setting umean to None)
        if not all(val is None for val in self.z0s):
            self.z0s = [self.z0s[0]] * self.ts_len
            self.raise_ffp_exception(13, self.verbosity)
            self.umeans = [None for i in range(self.ts_len)]
            # If only one value of z0 was passed, use that value for all footprints
            if len(self.z0s) == 1:
                self.z0s = [self.z0s[0] for i in range(self.ts_len)]

        elif len(self.umeans) == self.ts_len and not all(val is None for val in self.umeans):
            self.raise_ffp_exception(14, self.verbosity)
            self.z0s = [None for i in range(self.ts_len)]

        else:
            self.raise_ffp_exception(15, self.verbosity)

    def handle_rs(self):
        # Handle rs
        if self.rs is not None:

            # Check that rs is a list, otherwise make it a list
            if isinstance(self.rs, numbers.Number):
                if 0.9 < self.rs <= 1 or 90 < self.rs <= 100:
                    self.rs = 0.9
                self.rs = [self.rs]

            if not isinstance(self.rs, list):
                self.raise_ffp_exception(18, self.verbosity)

            # If rs is passed as percentages, normalize to fractions of one
            if np.max(self.rs) >= 1:
                self.rs = [x / 100. for x in self.rs]

            # Eliminate any values beyond 0.9 (90%) and inform user
            if np.max(self.rs) > 0.9:
                self.raise_ffp_exception(19, self.verbosity)
                self.rs = [item for item in self.rs if item <= 0.9]

            # Sort levels in ascending order
            self.rs = list(np.sort(self.rs))

    def computational_domain(self):
        # Define computational domain
        # Check passed values and make some smart assumptions
        if isinstance(self.dx, numbers.Number) and self.dy is None:
            self.dy = self.dx

        if isinstance(self.dy, numbers.Number) and self.dx is None:
            self.dx = self.dy

        if not all(isinstance(item, numbers.Number) for item in [self.dx, self.dy]):
            self.dx = self.dy = None

        if isinstance(self.nx, int) and self.ny is None:
            self.ny = self.nx

        if isinstance(self.ny, int) and self.nx is None:
            self.nx = self.ny

        if not all(isinstance(item, int) for item in [self.nx, self.ny]):
            self.nx = self.ny = None

        if not isinstance(self.domain, list) or len(self.domain) != 4:
            self.domain = None

        if all(item is None for item in [self.dx, self.nx, self.domain]):
            # If nothing is passed, default domain is a square of 2 Km size centered
            # at the tower with pizel size of 2 meters (hence a 1000x1000 grid)
            self.domain = [-1000., 1000., -1000., 1000.]
            self.dx = self.dy = 2.
            self.nx = self.ny = 1000
        elif self.domain is not None:
            # If domain is passed, it takes the precendence over anything else
            if self.dx is not None:
                # If dx/dy is passed, takes precendence over nx/ny
                self.nx = int((self.domain[1] - self.domain[0]) / self.dx)
                self.ny = int((self.domain[3] - self.domain[2]) / self.dy)
            else:
                # If dx/dy is not passed, use nx/ny (set to 1000 if not passed)
                if self.nx is None:
                    self.nx = self.ny = 1000
                # If dx/dy is not passed, use nx/ny
                self.dx = (self.domain[1] - self.domain[0]) / float(self.nx)
                self.dy = (self.domain[3] - self.domain[2]) / float(self.ny)
        elif self.dx is not None and self.nx is not None:
            # If domain is not passed but dx/dy and nx/ny are, define domain
            self.domain = [-self.nx * self.dx / 2,
                           self.nx * self.dx / 2,
                           -self.ny * self.dy / 2,
                           self.ny * self.dy / 2]
        elif self.dx is not None:
            # If domain is not passed but dx/dy is, define domain and nx/ny
            self.domain = [-1000, 1000, -1000, 1000]
            self.nx = int((self.domain[1] - self.domain[0]) / self.dx)
            self.ny = int((self.domain[3] - self.domain[2]) / self.dy)
        elif self.nx is not None:
            # If domain and dx/dy are not passed but nx/ny is, define domain and dx/dy
            self.domain = [-1000, 1000, -1000, 1000]
            self.dx = (self.domain[1] - self.domain[0]) / float(self.nx)
            self.dy = (self.domain[3] - self.domain[2]) / float(self.nx)

        # Put domain into more convenient vars
        self.xmin, self.xmax, self.ymin, self.ymax = self.domain

    def check_inputs(self):
        # Define rslayer if not passed
        if self.rslayer == None:
            self.rslayer == 0

        # Define smooth_data if not passed
        if self.smooth_data == None:
            self.smooth_data == 1

        # Define crop if not passed
        if self.crop == None:
            self.crop = 0

        # Define pulse if not passed
        if self.pulse == None:
            if self.ts_len <= 20:
                self.pulse = 1
            else:
                self.pulse = int(self.ts_len / 20)

        # Define fig if not passed
        if self.fig == None:
            self.fig = 0


    def define_domain(self):
        # Define physical domain in cartesian and polar coordinates
        # Cartesian coordinates
        self.x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        self.y = np.linspace(self.ymin, self.ymax, self.ny + 1)
        self.x_2d, self.y_2d = np.meshgrid(self.x, self.y)

        # Polar coordinates
        # Set theta such that North is pointing upwards and angles increase clockwise
        self.rho = np.sqrt(self.x_2d ** 2 + self.y_2d ** 2)
        self.theta = np.arctan2(self.x_2d, self.y_2d)

        # initialize raster for footprint climatology
        self.fclim_2d = np.zeros(self.x_2d.shape)

    def loop_over_time(self):
        # Loop on time series
        # Initialize logic array valids to those 'timestamps' for which all inputs are
        # at least present (but not necessarily phisically plausible)
        self.valids = [True if not any([val is None for val in vals]) else False \
                  for vals in zip(self.ustars, self.sigmavs, self.hs, self.ols, self.wind_dirs, self.zms)]

        if self.verbosity > 1:
            print('')

        for ix, (ustar, sigmav, h, ol, wind_dir, zm, z0, umean) \
                in enumerate(zip(self.ustars, self.sigmavs, self.hs, self.ols, self.wind_dirs, self.zms, self.z0s, self.umeans)):

            # Counter
            if self.verbosity > 1 and ix % self.pulse == 0:
                print('Calculating footprint ', ix + 1, ' of ', self.ts_len)

            self.valids[ix] = self.check_ffp_inputs(ustar, sigmav, h, ol, wind_dir, zm, z0, umean, self.rslayer, self.verbosity)

            # If inputs are not valid, skip current footprint
            if not self.valids[ix]:
                self.raise_ffp_exception(16, self.verbosity)
            else:
                # Rotate coordinates into wind direction
                if wind_dir is not None:
                    rotated_theta = self.theta - wind_dir * np.pi / 180.

                # Create real scale crosswind integrated footprint and dummy for
                # rotated scaled footprint
                fstar_ci_dummy = np.zeros(self.x_2d.shape)
                f_ci_dummy = np.zeros(self.x_2d.shape)
                xstar_ci_dummy = np.zeros(self.x_2d.shape)
                px = np.ones(self.x_2d.shape)
                if z0 is not None:
                    # Use z0
                    if ol <= 0 or ol >= self.oln:
                        xx = (1 - 19.0 * zm / ol) ** 0.25
                        psi_f = (np.log((1 + xx ** 2) / 2.) + 2. * np.log((1 + xx) / 2.) - 2. * np.arctan(xx) + np.pi / 2)
                    elif ol > 0 and ol < self.oln:
                        psi_f = -5.3 * zm / ol
                    if (np.log(zm / z0) - psi_f) > 0:
                        xstar_ci_dummy = (self.rho * np.cos(rotated_theta) / zm * (1. - (zm / h)) / (np.log(zm / z0) - psi_f))
                        px = np.where(xstar_ci_dummy > self.d)
                        fstar_ci_dummy[px] = self.a * (xstar_ci_dummy[px] - self.d) ** self.b * np.exp(-self.c / (xstar_ci_dummy[px] - self.d))
                        f_ci_dummy[px] = (fstar_ci_dummy[px] / zm * (1. - (zm / h)) / (np.log(zm / z0) - psi_f))
                    else:
                        flag_err = 3
                        self.valids[ix] = 0
                else:
                    # Use umean if z0 not available
                    xstar_ci_dummy = (self.rho * np.cos(rotated_theta) / zm * (1. - (zm / h)) / (umean / ustar * self.k))
                    px = np.where(xstar_ci_dummy > self.d)
                    fstar_ci_dummy[px] = self.a * (xstar_ci_dummy[px] - self.d) ** self.b * np.exp(-self.c / (xstar_ci_dummy[px] - self.d))
                    f_ci_dummy[px] = (fstar_ci_dummy[px] / zm * (1. - (zm / h)) / (umean / ustar * self.k))

                # Calculate dummy for scaled sig_y* and real scale sig_y
                sigystar_dummy = np.zeros(self.x_2d.shape)
                sigystar_dummy[px] = (self.ac * np.sqrt(self.bc * np.abs(xstar_ci_dummy[px]) ** 2 / (1 +
                                                                                           self.cc * np.abs(
                            xstar_ci_dummy[px]))))

                if abs(ol) > self.oln:
                    ol = -1E6

                if ol <= 0:  # convective
                    scale_const = 1E-5 * abs(zm / ol) ** (-1) + 0.80
                elif ol > 0:  # stable
                    scale_const = 1E-5 * abs(zm / ol) ** (-1) + 0.55

                if scale_const > 1:
                    scale_const = 1.0

                sigy_dummy = np.zeros(self.x_2d.shape)
                sigy_dummy[px] = (sigystar_dummy[px] / scale_const * zm * sigmav / ustar)
                sigy_dummy[sigy_dummy < 0] = np.nan

                # Calculate real scale f(x,y)
                f_2d = np.zeros(self.x_2d.shape)
                f_2d[px] = (f_ci_dummy[px] / (np.sqrt(2 * np.pi) * sigy_dummy[px]) *
                            np.exp(-(self.rho[px] * np.sin(rotated_theta[px])) ** 2 / (2. * sigy_dummy[px] ** 2)))

                # Add to footprint climatology raster
                self.fclim_2d = self.fclim_2d + f_2d

    def get_results(self):
        # Continue if at least one valid footprint was calculated
        n = sum(self.valids)
        vs = None
        clevs = None

        if n == 0:
            print("No footprint calculated")
            flag_err = 1
        else:

            # Normalize and smooth footprint climatology
            self.fclim_2d = self.fclim_2d / n

            if self.smooth_data is not None:
                self.skernel = np.matrix('0.05 0.1 0.05; 0.1 0.4 0.1; 0.05 0.1 0.05')
                self.fclim_2d = sg.convolve2d(self.fclim_2d, self.skernel, mode='same')
                self.fclim_2d = sg.convolve2d(self.fclim_2d, self.skernel, mode='same')

            # Derive footprint ellipsoid incorporating R% of the flux, if requested,
            # starting at peak value.
            self.xrs = []
            self.yrs = []

            if self.rs is not None:
                clevs = self.get_contour_levels(self.fclim_2d, rs=self.rs)
                frs = [item[2] for item in clevs]

                for ix, fr in enumerate(frs):
                    xrr, yrr = self.get_contour_vertices(self.x_2d,
                                                         self.y_2d,
                                                         self.fclim_2d,
                                                         fr)
                    if xrr is None:
                        frs[ix] = None
                        flag_err = 2
                    self.xrs.append(xrr)
                    self.yrs.append(yrr)
            else:
                if self.crop:
                    rs_dummy = 0.8  # crop to 80%
                    clevs = self.get_contour_levels(self.fclim_2d,
                                                    rs=rs_dummy)
                    frs = [item[2] for item in clevs]
                    self.xrs, self.yrs = self.get_contour_vertices(self.x_2d,
                                                                   self.y_2d,
                                                                   self.fclim_2d,
                                                                   clevs[0][2])

            # Crop domain and footprint to the largest rs value
            if self.crop:
                self.xrs_crop = [x for x in self.xrs if x is not None]
                self.yrs_crop = [x for x in self.yrs if x is not None]
                if self.rs is not None:
                    self.dminx = np.floor(min(self.xrs_crop[-1]))
                    self.dmaxx = np.ceil(max(self.xrs_crop[-1]))
                    self.dminy = np.floor(min(self.yrs_crop[-1]))
                    self.dmaxy = np.ceil(max(self.yrs_crop[-1]))
                else:
                    self.dminx = np.floor(min(self.xrs_crop))
                    self.dmaxx = np.ceil(max(self.xrs_crop))
                    self.dminy = np.floor(min(self.yrs_crop))
                    self.dmaxy = np.ceil(max(self.yrs_crop))

                if self.dminy >= self.ymin and self.dmaxy <= self.ymax:
                    jrange = np.where((self.y_2d[:, 0] >= self.dminy) & (self.y_2d[:, 0] <= self.dmaxy))[0]
                    jrange = np.concatenate(([jrange[0] - 1], jrange, [jrange[-1] + 1]))
                    jrange = jrange[np.where((jrange >= 0) & (jrange <= self.y_2d.shape[0]))[0]]
                else:
                    jrange = np.linspace(0, 1, self.y_2d.shape[0] - 1)

                if self.dminx >= self.xmin and self.dmaxx <= self.xmax:
                    irange = np.where((self.x_2d[0, :] >= self.dminx) & (self.x_2d[0, :] <= self.dmaxx))[0]
                    irange = np.concatenate(([irange[0] - 1], irange, [irange[-1] + 1]))
                    irange = irange[np.where((irange >= 0) & (irange <= self.x_2d.shape[1]))[0]]
                else:
                    irange = np.linspace(0, 1, self.x_2d.shape[1] - 1)

                jrange = [[it] for it in jrange]
                self.x_2d = self.x_2d[jrange, irange]
                self.y_2d = self.y_2d[jrange, irange]
                self.fclim_2d = self.fclim_2d[jrange, irange]

        # Plot footprint
        if self.fig:
            fig_out,ax = self.plot_footprint(fs=self.fclim_2d,
                                             show_heatmap=self.show_heatmap,
                                             clevs=frs)
        else:
            fig_out = None

        # Fill output structure
        if self.rs is not None:
            return {'x_2d': self.x_2d,
                    'y_2d': self.y_2d,
                    'fclim_2d': self.fclim_2d,
                    'fig': fig_out,
                    'rs': self.rs,
                    'fr': frs,
                    'xr': self.xrs,
                    'yr': self.yrs,
                    'n': n,
                    'flag_err': self.flag_err}
        else:
            return {'x_2d': self.x_2d,
                    'y_2d': self.y_2d,
                    'fclim_2d': self.fclim_2d,
                    'n': n,
                    'flag_err': self.flag_err}


    def check_ffp_inputs(self,
                         ustar,
                         sigmav,
                         h,
                         ol,
                         wind_dir,
                         zm,
                         z0,
                         umean,
                         rslayer,
                         verbosity):
        """
        Validates the input parameters ensuring they meet physical plausibility and consistency
        requirements for further atmospheric boundary layer calculations. This function performs
        checks on heights, wind speeds, stability parameters, and other inputs.

        Args:
            ustar (float): Friction velocity [m/s], must be greater than 0.1.
            sigmav (float): Standard deviation of the lateral velocity fluctuations, must be
                greater than 0.
            h (float): Boundary layer height [m], must be greater than 10 and should be
                greater than the measurement height `zm`.
            ol (float): Obukhov length [m], used for stability determination, `zm/ol`
                must not be less than -15.5.
            wind_dir (float): Wind direction [degrees], must be between 0 and 360 inclusive.
            zm (float): Measurement height [m], must be greater than 0 and less than or
                equal to `h`.
            z0 (Optional[float]): Roughness length [m], if provided, must be greater than 0.
            umean (Optional[float]): Mean horizontal wind speed [m/s], used in some checks
                involving `z0`.
            rslayer (int): Flag for reference sublayer validation, used when comparing
                `zm` and `z0`, must be an integer where specific conditions apply.
            verbosity (int): Level of verbosity for error output, controls the exception
                messages raised during validation.

        Returns:
            bool: Returns True if all inputs are valid, otherwise False after raising
            an exception.

        Raises:
            Exception: Raised with specific codes representing the failed condition:
                Code 2: If `zm` is less than or equal to 0.
                Code 3: If `z0` is invalid (not None and less than or equal to 0, with
                `umean` unspecified).
                Code 4: If `h` is less than or equal to 10.
                Code 5: If `zm` exceeds `h`.
                Code 6: If `zm` is less than or equal to 12.5 * z0 when `rslayer` equals 1.
                Code 20: If `zm` is less than or equal to 12.5 * z0 and `rslayer` is
                not 1.
                Code 7: If `zm/ol` is less than or equal to -15.5.
                Code 8: If `sigmav` is less than or equal to 0.
                Code 9: If `ustar` is less than or equal to 0.1.
                Code 10: If `wind_dir` is not within the range [0, 360].
        """
        # Check passed values for physical plausibility and consistency
        if zm <= 0.:
            self.raise_ffp_exception(2, verbosity)
            return False
        if z0 is not None and umean is None and z0 <= 0.:
            self.raise_ffp_exception(3, verbosity)
            return False
        if h <= 10.:
            self.raise_ffp_exception(4, verbosity)
            return False
        if zm > h:
            self.raise_ffp_exception(5, self.verbosity)
            return False
        if z0 is not None and umean is None and zm <= 12.5 * z0:
            if rslayer == 1:
                self.raise_ffp_exception(6, self.verbosity)
            else:
                self.raise_ffp_exception(20, self.verbosity)
                return False
        if float(zm) / ol <= -15.5:
            self.raise_ffp_exception(7, self.verbosity)
            return False
        if sigmav <= 0:
            self.raise_ffp_exception(8, self.verbosity)
            return False
        if ustar <= 0.1:
            self.raise_ffp_exception(9, self.verbosity)
            return False
        if wind_dir > 360:
            self.raise_ffp_exception(10, self.verbosity)
            return False
        if wind_dir < 0:
            self.raise_ffp_exception(10, self.verbosity)
            return False
        return True


    def get_contour_levels(self, f, rs=None):
        """
        Calculates contour levels based on the input 2D field `f`. The contour levels
        are determined by computing cumulative sums of the sorted values of `f` and
        comparing them to a set of specified relative thresholds `rs`. Each threshold
        represents a fraction of the total sum. The function identifies the contour
        levels for these threshold values and computes the corresponding areas.

        Args:
            f (numpy.ndarray): A 2D field (e.g., data array) for which to compute the
                contour levels.
            dx (float): Grid spacing in the x-direction, used for scaling the areas.
            dy (float): Grid spacing in the y-direction, used for scaling the areas.
            rs (Union[int, float, List[float], optional]): A single relative threshold
                value or a list of thresholds (default: evenly spaced values between
                0.10 and 0.90). Each threshold is a fraction of the cumulative sum
                used to determine the contour levels.

        Returns:
            List[Tuple[float, float, float]]: A list of tuples, where each tuple
                contains:
                - The relative threshold (rounded to three decimal places).
                - The calculated area corresponding to the threshold.
                - The contour level in the field `f` associated with the threshold.

        Raises:
            TypeError: If `rs` is not an integer, float, or list of floats.
        """

        # Check input and resolve to default levels in needed
        if not isinstance(rs, (int, float, list)):
            rs = list(np.linspace(0.10, 0.90, 9))
        if isinstance(rs, (int, float)): rs = [rs]

        # Levels
        pclevs = np.empty(len(rs))
        pclevs[:] = np.nan
        ars = np.empty(len(rs))
        ars[:] = np.nan

        sf = np.sort(f, axis=None)[::-1]
        msf = np.ma.masked_array(sf, mask=(np.isnan(sf) | np.isinf(sf)))  # Masked array for handling potential nan
        csf = msf.cumsum().filled(np.nan) * self.dx * self.dy
        for ix, r in enumerate(rs):
            dcsf = np.abs(csf - r)
            pclevs[ix] = sf[np.nanargmin(dcsf)]
            ars[ix] = csf[np.nanargmin(dcsf)]

        return [(round(r, 3), ar, pclev) for r, ar, pclev in zip(rs, ars, pclevs)]

    def get_contour_vertices(self, x, y, f, lev):
        """
        Computes the x and y coordinates of vertices of a contour curve at a specified level.

        Uses Matplotlib to calculate the contour at a given level for a 2D field. The function
        checks whether the contour lies entirely within the domain defined by x and y. If any
        part of the contour exits the domain, it returns None for both x and y.

        Args:
            x (numpy.ndarray): A 1D array representing the x-coordinates of the grid.
            y (numpy.ndarray): A 1D array representing the y-coordinates of the grid.
            f (numpy.ndarray): A 2D array representing the function or field values on the
                grid defined by x and y.
            lev (float): The level at which the contour is to be computed.

        Returns:
            list: A list containing two elements: a list of x-coordinates and a list of
            y-coordinates of contour vertices. Returns [None, None] if the contour exits
            the physical domain as defined by the x and y bounds.
        """
        # import matplotlib._contour as cntr
        cs = plt.contour(x, y, f, [lev])
        plt.close()
        segs = cs.allsegs[0][0]
        xr = [vert[0] for vert in segs]
        yr = [vert[1] for vert in segs]
        # Set contour to None if it's found to reach the physical domain
        if x.min() >= min(segs[:, 0]) or max(segs[:, 0]) >= x.max() or \
                y.min() >= min(segs[:, 1]) or max(segs[:, 1]) >= y.max():
            return [None, None]

        return [xr, yr]  # x,y coords of contour points.

    def plot_footprint(self, clevs=None, show_heatmap=True, normalize=None,
                       colormap=None, line_width=0.5, iso_labels=None):
        '''Plot footprint function and contours if request'''

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import LogNorm

        # If input is a list of footprints, don't show footprint but only contours,
        # with different colors
        if isinstance(self.fs, list):
            show_heatmap = False
        else:
            self.fs = [self.fs]

        if colormap is None: colormap = cm.jet
        # Define colors for each contour set
        cs = [colormap(ix) for ix in np.linspace(0, 1, len(self.fs))]

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
            for f, c in zip(self.fs, cs):
                cc = [c] * len(levs)
                if show_heatmap:
                    cp = ax.contour(self.x_2d, self.y_2d, f, levs, colors='w', linewidths=line_width)
                else:
                    cp = ax.contour(self.x_2d, self.y_2d, f, levs, colors=cc, linewidths=line_width)
                # Isopleth Labels
                if iso_labels is not None:
                    pers = [str(int(clev[0] * 100)) + '%' for clev in clevs]
                    fmt = {}
                    for l, s in zip(cp.levels, pers):
                        fmt[l] = s
                    plt.clabel(cp, cp.levels[:], inline=1, fmt=fmt, fontsize=7)

        # plot footprint heatmap if requested and if only one footprint is passed
        if show_heatmap:
            if normalize == 'log':
                norm = LogNorm()
            else:
                norm = None

            xmin = np.nanmin(self.x_2d)
            xmax = np.nanmax(self.x_2d)
            ymin = np.nanmin(self.y_2d)
            ymax = np.nanmax(self.y_2d)
            for f in self.fs:
                im = ax.imshow(f[:, :], cmap=colormap, extent=(self.xmin, self.xmax, self.ymin, self.ymax),
                               norm=norm, origin='lower', aspect=1)
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')

            # Colorbar
            cbar = fig.colorbar(im, shrink=1.0, format='%.3e')
            # cbar.set_label('Flux contribution', color = 'k')
        plt.show()

        return fig, ax

    def raise_ffp_exception(self, code, verbosity):
        """
        Raises an exception or logs a message based on exception type and verbosity level.

        This function processes a given exception code, matches it with a predefined list
        of exceptions, and either logs detailed outputs or raises exceptions depending
        on the type of the exception and the verbosity level provided as inputs.

        Args:
            code (str): The code of the exception to handle. This is matched against a
                predefined list of exceptions to determine the exception's type and
                message.
            verbosity (int): The level of verbosity that controls the output. A higher
                verbosity level produces more detailed logs.

        Raises:
            Exception: Raised when the exception type is specified as `fatal` in the
                predefined list.
        """

        ex = [it for it in self.exceptions if it['code'] == code][0]
        string = ex['type'] + '(' + str(ex['code']).zfill(4) + '):\n ' + ex['msg']

        if verbosity > 0: print('')

        if ex['type'] == self.exTypes['fatal']:
            if verbosity > 0:
                string = string + '\n FFP_fixed_domain execution aborted.'
            else:
                string = ''
            raise Exception(string)
        elif ex['type'] == self.exTypes['alert']:
            string = string + '\n Execution continues.'
            if verbosity > 1: print(string)
        elif ex['type'] == self.exTypes['error']:
            string = string + '\n Execution continues.'
            if verbosity > 1: print(string)
        else:
            if verbosity > 1: print(string)
