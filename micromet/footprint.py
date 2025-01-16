import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import signal as sg
from affine import Affine
import cv2
import logging


class FFPClimatology:
    def __init__(
        self,
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
    ):
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
        self.fig = fig

    def raise_ffp_exception(self, code):
        exceptions = {
            1: "At least one required parameter is missing. Check the inputs.",
            2: "zm (measurement height) must be larger than zero.",
            3: "z0 (roughness length) must be larger than zero.",
            4: "h (boundary layer height) must be larger than 10 m.",
            5: "zm (measurement height) must be smaller than h (boundary layer height).",
            6: "zm (measurement height) should be above the roughness sub-layer.",
            7: "zm/ol (measurement height to Obukhov length ratio) must be >= -15.5.",
            8: "sigmav (standard deviation of crosswind) must be larger than zero.",
            9: "ustar (friction velocity) must be >= 0.1.",
            10: "wind_dir (wind direction) must be in the range [0, 360].",
        }

        message = exceptions.get(code, "Unknown error code.")

        if self.verbosity > 0:
            print(f"Error {code}: {message}")

        if code in [1, 4, 5, 7, 9, 10]:  # Fatal errors
            raise ValueError(f"FFP Exception {code}: {message}")

    def get_contour_levels(self, f, dx, dy, rs=None):
        if rs is None:
            rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        sf = np.sort(f.ravel())[::-1]
        sf_cumsum = np.cumsum(sf) * dx * dy

        levels = []
        for r in rs:
            target = r * sf_cumsum[-1]
            idx = np.searchsorted(sf_cumsum, target)
            levels.append(sf[idx] if idx < len(sf) else None)

        return levels

    def mask_fp_cutoff(self, f_array, cutoff=0.9):
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

        self.logger.debug(
            f"mask_fp_cutoff: applied cutoff={cutoff}, sum_cutoff={sum_cutoff}"
        )
        return f_array

    def find_transform(self, xs, ys):
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
        self.logger.debug("Affine transform calculated.")
        return aff_transform

    def validate_inputs(self):
        if any(
            inp is None for inp in [self.zm, self.h, self.ol, self.sigmav, self.ustar]
        ):
            self.raise_ffp_exception(1)
        if self.z0 is None and self.umean is None:
            self.raise_ffp_exception(1)
        inputs = [
            self.zm,
            self.h,
            self.ol,
            self.sigmav,
            self.ustar,
            self.wind_dir,
            self.z0,
            self.umean,
        ]
        return [inp if isinstance(inp, list) else [inp] for inp in inputs]

    def configure_domain(self):
        # Ensure at least one of dx or nx is defined
        if self.dx is None:
            self.dx = 2  # Default cell size

        if self.nx is None:
            self.nx = 1000

        # If dy or ny is missing, match with dx or nx
        if self.dy is None:
            self.dy = self.dx
        if self.ny is None:
            self.ny = self.nx

        # Define domain if not provided
        if self.domain is None:
            half_size = self.nx * self.dx / 2
            self.domain = [-half_size, half_size, -half_size, half_size]

        # Calculate the grid
        xmin, xmax, ymin, ymax = self.domain
        x = np.linspace(xmin, xmax, self.nx + 1)
        y = np.linspace(ymin, ymax, self.ny + 1)
        return np.meshgrid(x, y)

    def crop_domain(self, x_2d, y_2d, fclim_2d):
        if not self.crop:
            return x_2d, y_2d, fclim_2d

        indices = np.where(fclim_2d > 0)
        if indices[0].size == 0:
            return x_2d, y_2d, fclim_2d

        xmin, xmax = np.min(x_2d[indices]), np.max(x_2d[indices])
        ymin, ymax = np.min(y_2d[indices]), np.max(y_2d[indices])

        cropped_x = x_2d[(x_2d >= xmin) & (x_2d <= xmax)]
        cropped_y = y_2d[(y_2d >= ymin) & (y_2d <= ymax)]

        return cropped_x, cropped_y, fclim_2d

    def plot_footprint(self, x_2d, y_2d, fclim_2d, contours):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            fclim_2d,
            extent=(np.min(x_2d), np.max(x_2d), np.min(y_2d), np.max(y_2d)),
            origin="lower",
            norm=LogNorm() if self.smooth_data else None,
            aspect="auto",
            cmap="jet",
        )

        if contours:
            for contour in contours:
                ax.contour(x_2d, y_2d, fclim_2d, levels=[contour], colors="white")

        plt.colorbar(im, ax=ax, label="Flux Contribution")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Flux Footprint Climatology")
        plt.show()

    def generate_results(self, fclim_2d, n, x_2d, y_2d):
        fclim_2d /= n
        if self.smooth_data:
            kernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
            fclim_2d = sg.convolve2d(fclim_2d, kernel, mode="same")

        x_2d, y_2d, fclim_2d = self.crop_domain(x_2d, y_2d, fclim_2d)
        contour_levels = self.get_contour_levels(fclim_2d, self.dx, self.dy, self.rs)

        if self.fig:
            self.plot_footprint(x_2d, y_2d, fclim_2d, contour_levels)

        return {
            "x_2d": x_2d,
            "y_2d": y_2d,
            "fclim_2d": fclim_2d,
            "rs": self.rs,
            "contours": contour_levels,
            "valid_count": n,
            "error_flag": 0 if n > 0 else 1,
        }

    def run(self):
        inputs = self.validate_inputs()
        x_2d, y_2d = self.configure_domain()
        rho = np.sqrt(x_2d**2 + y_2d**2)
        theta = np.arctan2(x_2d, y_2d)
        fclim_2d = np.zeros(x_2d.shape)
        n = 0

        for ix in range(len(inputs[0])):
            f_2d = self.compute_footprint(ix, inputs, x_2d, y_2d, rho, theta)
            fclim_2d += f_2d
            n += 1 if np.any(f_2d) else 0

        return self.generate_results(fclim_2d, n, x_2d, y_2d)
