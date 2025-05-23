import h5py
import logging
import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import cho_solve, cho_factor
import script.dataset as ds

"""
TODO:
1. select bandwidth?

"""


def traz(f, g):
    """
    Numeric integration using trapezoid rule
    
    """
    if len(f) != len(g):
        raise ValueError('f and g have different lengths')
    if not all(f[i] < f[i+1] for i in range(len(f)-1)):
        raise ValueError('f must be strictly increasing')
    res = 0
    for i in range(len(f) - 1):
        res += 0.5 * (f[i+1] - f[i]) * (g[i] + g[i+1])
    return res


def interp2lin(xin: np.ndarray, yin: np.ndarray, zin: np.ndarray, xou: np.ndarray, you: np.ndarray) -> np.ndarray:
    """
    Performs bilinear interpolation based on the logic of the provided Rcpp code.
    The Rcpp code fits f(x,y) = c0 + c1*x + c2*y + c3*x*y to the four cell corners.

    Args:
        xin (np.ndarray): 1D array of x-coordinates of the grid. Must be sorted.
        yin (np.ndarray): 1D array of y-coordinates of the grid. Must be sorted.
        zin (np.ndarray): 1D array of z-values at grid points (flattened, row-major).
                          zin corresponds to a grid of shape (len(yin), len(xin)).
        xou (np.ndarray): 1D array of x-coordinates for output points.
        you (np.ndarray): 1D array of y-coordinates for output points.

    Returns:
        np.ndarray: 1D array of interpolated z-values for each (xou, you) point.
    """
    nXGrid = xin.shape[0]
    nYGrid = yin.shape[0]
    nUnknownPoints = xou.shape[0]
    
    # Initialize result array with NaNs
    result = np.full(nUnknownPoints, np.nan)
    
    # Handle empty or invalid grid inputs minimally, as per "no input checks"
    if nXGrid == 0 or nYGrid == 0:
        return result 
        
    # Grid boundaries from the input grid coordinates
    xmin, xmax = xin[0], xin[-1]
    ymin, ymax = yin[0], yin[-1]
    
    for i in range(nUnknownPoints):
        xo, yo = xou[i], you[i]
        
        # Boundary check from Rcpp: if point is outside the input grid, result is NaN.
        # The Rcpp condition is: (xmax < xo) || (ymax < yo) || (xmin > xo) || (ymin > yo)
        # This is equivalent to: not (xmin <= xo <= xmax and ymin <= yo <= ymax)
        if not (xmin <= xo <= xmax and ymin <= yo <= ymax):
            result[i] = np.nan
            continue
            
        # Find indices for the interpolation cell corners.
        # x_idx_upper: index of the smallest xin value that is >= xo
        # y_idx_upper: index of the smallest yin value that is >= yo
        # np.searchsorted(side='left') behaves like std::lower_bound.
        # Given that xmin <= xo <= xmax, x_idx_upper will be in [0, nXGrid-1].
        # Similarly for y_idx_upper in [0, nYGrid-1].
        x_idx_upper = np.searchsorted(xin, xo, side='left')
        y_idx_upper = np.searchsorted(yin, yo, side='left')

        # Ensure x_idx_upper is not out of bounds if xo == xmax and xin has duplicates
        # (though typically grid vectors are unique and sorted).
        # If xo == xin[nXGrid-1], searchsorted returns nXGrid-1. This is correct.
        # This clamping is mostly defensive for unusual xin/yin.
        x_idx_upper = np.clip(x_idx_upper, 0, nXGrid - 1)
        y_idx_upper = np.clip(y_idx_upper, 0, nYGrid - 1)
        
        # Determine the actual grid coordinate values for the cell's "upper" corner
        # This corresponds to xa(1) and ya(1) in the Rcpp code.
        val_xU_coord = xin[x_idx_upper] 
        val_yU_coord = yin[y_idx_upper]

        # Determine the grid coordinate values for the cell's "lower" corner.
        # This corresponds to xa(0) and ya(0) in the Rcpp code.
        if x_idx_upper == 0:
            x_idx_lower = 0
        else:
            # If xo is exactly on xin[x_idx_upper], the Rcpp logic uses xin[x_idx_upper-1] as the lower bound for the cell.
            x_idx_lower = x_idx_upper - 1
        val_xL_coord = xin[x_idx_lower]
        
        if y_idx_upper == 0:
            y_idx_lower = 0
        else:
            y_idx_lower = y_idx_upper - 1
        val_yL_coord = yin[y_idx_lower]

        # Retrieve Z values for the four corners of the cell
        # These are indexed using x_idx_lower, x_idx_upper, y_idx_lower, y_idx_upper.
        # z00: Z at (val_xL_coord, val_yL_coord)
        z00 = zin[y_idx_lower * nXGrid + x_idx_lower]
        # z01: Z at (val_xL_coord, val_yU_coord) - Corresponds to Rcpp's za(1)
        z01 = zin[y_idx_upper * nXGrid + x_idx_lower]
        # z10: Z at (val_xU_coord, val_yL_coord) - Corresponds to Rcpp's za(2)
        z10 = zin[y_idx_lower * nXGrid + x_idx_upper]
        # z11: Z at (val_xU_coord, val_yU_coord) - Corresponds to Rcpp's za(3)
        z11 = zin[y_idx_upper * nXGrid + x_idx_upper]
        
        # This array `corners_z` matches the order of `za` in the Rcpp code
        # after mapping its construction.
        corners_z = np.array([z00, z01, z10, z11])
        
        # Construct matrix A for solving coefficients c of f(x,y) = c0 + c1*x + c2*y + c3*x*y
        # The rows of A correspond to the four corner points:
        # (xL, yL), (xL, yU), (xU, yL), (xU, yU)
        A_matrix = np.array([
            [1, val_xL_coord, val_yL_coord, val_xL_coord * val_yL_coord], # Eq for z00
            [1, val_xL_coord, val_yU_coord, val_xL_coord * val_yU_coord], # Eq for z01
            [1, val_xU_coord, val_yL_coord, val_xU_coord * val_yL_coord], # Eq for z10
            [1, val_xU_coord, val_yU_coord, val_xU_coord * val_yU_coord]  # Eq for z11
        ])
        
        # Solve A_matrix * coeffs = corners_z for coeffs.
        # np.linalg.lstsq is used for robustness, similar to Rcpp's QR decomposition solver,
        # especially if the cell is degenerate (e.g., val_xL_coord == val_xU_coord).
        coeffs, _, _, _ = np.linalg.lstsq(A_matrix, corners_z, rcond=None)
        
        # Interpolate the value at (xo, yo) using the solved coefficients
        # fq = [1, xo, yo, xo*yo]
        fq_vector = np.array([1, xo, yo, xo * yo])
        result[i] = fq_vector @ coeffs
        
    return result


class LocalLinear(ABC):
    """
    Abstract class for local linear estimator.
    
    """
    GAUSSIAN_CONST = 1 / np.sqrt(2 * np.pi)

    def __init__(self, pheno):
        """
        pheno:

        Attributes:
        ------------
        
        """
        self.pheno = pheno.pheno
        self.time = pheno.time
        self.n_obs = pheno.sub_n_obs
        self.unique_time = pheno.unique_time
        self.grid_size = pheno.grid_size
        self.time_grid = pheno.time_grid
        self.time_idx = pheno.time_idx
        self.n_time = len(self.unique_time)
        self.n_grids = len(self.time_grid)
        self.logger = logging.getLogger(__name__)

    def _gau_kernel(self, x):
        """
        Calculating the Gaussian density

        """
        gau_k = self.GAUSSIAN_CONST * np.exp(-0.5 * x**2)
        if len(gau_k.shape) == 2:
            gau_k = np.prod(gau_k, axis=1).reshape(-1, 1)
        return gau_k.astype(np.float32)
    
    @staticmethod
    def _wls(x, y, weights):
        """
        Weighted least squares
        
        """
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        xw = x * weights
        xtx = np.dot(xw.T, x)
        xtx += np.eye(xtx.shape[0]) * 1e-8
        xty = np.dot(xw.T, y)
        c, lower = cho_factor(xtx)
        beta = cho_solve((c, lower), xty).astype(np.float32)
        return beta[0]
    
    @abstractmethod
    def _get_design_matrix(self):
        """
        Get design matrix for regression
        
        """
        pass

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def _get_bw_candidates(self):
        pass
    

class Mean(LocalLinear):
    def _get_design_matrix(self, t, bw):
        time_diff = self.time - t
        weights = self._gau_kernel(time_diff / bw)
        time_diff = time_diff.reshape(-1, 1)
        x = np.hstack([np.ones_like(time_diff), time_diff])
        return x, weights
    
    def estimate(self, bw, grid=True):
        mean_function = np.zeros(self.n_time, dtype=np.float32)
        grid_mean_function = np.zeros(self.n_grids, dtype=np.float32)
        
        for i, t in enumerate(self.unique_time):
            x, weights = self._get_design_matrix(t, bw)
            mean_function[i] = self._wls(x, self.pheno, weights)

        if grid:
            for i, t in enumerate(self.time_grid):
                x, weights = self._get_design_matrix(t, bw)
                grid_mean_function[i] = self._wls(x, self.pheno, weights)

        return mean_function, grid_mean_function

    def _get_bw_candidates(self, time_window):
        bw_raw = self.n_time ** (-1 / 5) * 0.05 * time_window
        bw_cand = np.round(bw_raw * np.array([0.5, 1, 2, 3, 5, 10]), 6)
        return bw_cand
    
    def gcv(self):
        time_window = self.unique_time[-1] - self.unique_time[0]
        bw_cand = self._get_bw_candidates(time_window)
        gcv_score = np.zeros(6, dtype=np.float32)
        const = time_window * self.GAUSSIAN_CONST / len(self.pheno)
        self.logger.info(f"Selecting bandwidth for mean function from {bw_cand}")

        for i, bw in enumerate(bw_cand):
            if const / bw >= 1:
                gcv_score[i] = np.nan
                self.logger.info(f"The bandwidth {bw} is too small.")
                continue

            mean_bw, _ = self.estimate(bw, grid=False)
            gcv_score[i] = np.sum((self.pheno - mean_bw[self.time_idx]) ** 2)
            
            if gcv_score[i] == 0:
                gcv_score[i] = np.nan
                self.logger.info(f"The bandwidth is too small.")
                continue

            gcv_score[i] = gcv_score[i] / (1 - const / bw) ** 2
            self.logger.info(
                f"The GCV score for bandwidth {bw} is {gcv_score[i]:.3e}."
            )

        which_min = np.nanargmin(gcv_score)
        if which_min == 0 or which_min == len(bw_cand) - 1:
            self.logger.info(
                (
                    "WARNING: the optimal bandwidth obtained at the boundary "
                    "may not be the best one."
                )
            )
        bw_opt = bw_cand[which_min]
        min_mse = gcv_score[which_min]
        self.logger.info(
            f"The optimal bandwidth is {bw_opt} with GCV score {min_mse:.3e}."
        )

        return bw_opt
    

class Covariance(LocalLinear):
    def __init__(self, pheno, mean, time_idx):
        super().__init__(pheno)
        self.two_way_pheno = np.zeros(
            np.sum(self.n_obs ** 2 - self.n_obs), 
            dtype=np.float32
        )
        self.two_way_time = np.zeros(
            (np.sum(self.n_obs ** 2 - self.n_obs), 2), 
            dtype=np.float32
        )

        start1, end1 = 0, 0
        start2, end2 = 0, 0
        for n_obs in self.n_obs:
            end1 += n_obs
            end2 += n_obs ** 2 - n_obs
            off_diag = ~np.eye(n_obs, dtype=bool)

            # time
            time_stack_by_col = np.tile(self.time[start1: end1].reshape(-1, 1), n_obs)
            time_stack_by_row = time_stack_by_col.T
            self.two_way_time[start2: end2, 0] = time_stack_by_row[off_diag]
            self.two_way_time[start2: end2, 1] = time_stack_by_col[off_diag]

            # pheno
            sub_pheno = self.pheno[start1: end1]
            sub_pheno = sub_pheno - mean[time_idx[start1: end1]]
            sub_pheno = sub_pheno.reshape(-1, 1)
            outer_prod_sub_pheno = np.dot(sub_pheno, sub_pheno.T)
            self.two_way_pheno[start2: end2] = outer_prod_sub_pheno[off_diag]

            start1 = end1
            start2 = end2

        # self.unique_time_comb = np.zeros((self.n_time ** 2 - self.n_time, 2), dtype=np.float32)
        # self.mean_pheno = np.zeros(self.n_time ** 2 - self.n_time, dtype=np.float32)
        # self.time_comb_count = np.zeros((self.n_time ** 2 - self.n_time, 1), dtype=np.int32)
        # k = 0
        # for i in range(self.n_time):
        #     for j in range(self.n_time):
        #         if i != j:
        #             t1 = self.unique_time[i]
        #             t2 = self.unique_time[j]
        #             self.unique_time_comb[k] = np.array([t1, t2])
        #             time_comb_idx = (self.two_way_time == self.unique_time_comb[k]).all(axis=1)
        #             self.time_comb_count[k] = np.sum(time_comb_idx)
        #             self.mean_pheno[k] = np.mean(self.two_way_pheno[time_comb_idx], axis=0)
        #             k += 1

    def _get_design_matrix(self, t1, t2, bw):
        """
        TODO: set large distances as 0.
        
        """
        # time_diff = self.unique_time_comb - np.array([t1, t2])
        time_diff = self.two_way_time - np.array([t1, t2])
        # weights = self._gau_kernel(time_diff / bw) * self.time_comb_count
        weights = self._gau_kernel(time_diff / bw)
        x = np.hstack([np.ones(time_diff.shape[0], dtype=np.float32).reshape(-1, 1), time_diff])
        return x, weights
    
    def _get_design_matrix2(self, time, t, bw):
        """
        local quadratic
        
        """
        time_diff = time - t
        # weights = self._gau_kernel(time_diff / bw) * self.time_comb_count
        weights = self._gau_kernel(time_diff / bw)
        time_diff[:, 1] = time_diff[:, 1] ** 2
        x = np.hstack([np.ones(time_diff.shape[0], dtype=np.float32).reshape(-1, 1), time_diff])
        return x, weights
    
    def estimate(self, bw):
        grid_cov_function = np.zeros((self.n_grids, self.n_grids), dtype=np.float32)

        for t1 in range(self.n_grids):
            for t2 in range(t1, self.n_grids):
                x, weights = self._get_design_matrix(self.time_grid[t1], self.time_grid[t2], bw)
                # grid_cov_function[t1, t2] = self._wls(x, self.mean_pheno, weights)
                grid_cov_function[t1, t2] = self._wls(x, self.two_way_pheno, weights)
        
        iu_rows, iu_cols = np.triu_indices(self.n_grids, k=1)
        grid_cov_function[(iu_cols, iu_rows)] = grid_cov_function[(iu_rows, iu_cols)]

        cut_time_grid = self.time_grid[
            (self.time_grid > np.quantile(self.time_grid, 0.25)) & 
            (self.time_grid < np.quantile(self.time_grid, 0.75))
        ]
        cut_time_grid = np.tile(cut_time_grid.reshape(-1, 1), 2)
        n_cut_time_grid = cut_time_grid.shape[0]
        rotation_matrix = np.array([[1, 1], [-1, 1]]).T * (np.sqrt(2) / 2)
        # rotated_time_comb = np.dot(self.unique_time_comb, rotation_matrix)
        rotated_two_way_time = np.dot(self.two_way_time, rotation_matrix)
        rotated_cut_time_grid = np.dot(cut_time_grid, rotation_matrix)
        cut_time_grid_diag = np.zeros(n_cut_time_grid, dtype=np.float32)
        for t in range(n_cut_time_grid):
            x, weights = self._get_design_matrix2(
                # rotated_time_comb, 
                rotated_two_way_time,
                rotated_cut_time_grid[t],
                0.1
            )
            # cut_time_grid_diag[t] = self._wls(x, self.mean_pheno, weights)
            cut_time_grid_diag[t] = self._wls(x, self.two_way_pheno, weights)

        return grid_cov_function, cut_time_grid_diag
    

class ResidualVariance(LocalLinear):
    def _get_design_matrix(self, t, bw):
        time_diff = self.time - t
        weights = self._gau_kernel(time_diff / bw)
        time_diff = time_diff.reshape(-1, 1)
        x = np.hstack([np.ones_like(time_diff), time_diff])
        return x, weights
    
    def estimate(self, mean, diag, time_idx, bw):
        one_way_mean = mean[time_idx]
        grid_resid_var = np.zeros(self.n_grids, dtype=np.float32)

        for i, t in enumerate(self.time_grid):
            x, weights = self._get_design_matrix(t, bw)
            grid_resid_var[i] = self._wls(x, (self.pheno - one_way_mean)**2, weights)

        cut_time_grid = self.time_grid[
            (self.time_grid > np.quantile(self.time_grid, 0.25)) & 
            (self.time_grid < np.quantile(self.time_grid, 0.75))
        ]
        grid_resid_var = grid_resid_var[
            (self.time_grid > np.quantile(self.time_grid, 0.25)) & 
            (self.time_grid < np.quantile(self.time_grid, 0.75))
        ]
        resid_var = traz(cut_time_grid, (grid_resid_var - diag)) / 0.5 # TODO: check it

        return resid_var
    

def eigen(grid_mean, grid_cov, grid_size, time_grid):
    """
    Eigen decomposition

    Parameters:
    ------------
    pheno: 
    grid_mean (n_ldrs, n_grids): a np.array of mean estimate
    grid_cov (n_ldrs, n_grids, n_grids): a np.array of cov estimate
    grid_size:
    time_grid:

    Returns:
    ---------
    eg_values:
    eg_vectors: 
    
    """
    eg_values, eg_vectors = np.linalg.eigh(grid_cov)
    eg_values = np.flip(eg_values) # (n_time, )
    eg_vectors = np.flip(eg_vectors, axis=1) # (n_time, n_time)
    eg_vectors = eg_vectors[:, eg_values > 0]
    eg_values = eg_values[eg_values > 0]
    
    fve = np.cumsum(eg_values) / np.sum(eg_values)
    n_opt = np.argmax(fve > 0.98) + 1
    eg_values = eg_values[:n_opt] * grid_size
    eg_vectors = eg_vectors[:, :n_opt]

    for j in range(n_opt):
        eg_vectors[:, j] = eg_vectors[:, j] / np.sqrt(traz(time_grid, eg_vectors[:, j] ** 2))
        if np.sum(eg_vectors[:, j] * grid_mean) < 0:
            eg_vectors[:, j] = -eg_vectors[:, j]
        
    return eg_values, eg_vectors
    

class FPCAres:
    def __init__(self, file_path):
        with h5py.File(file_path, "r") as file:
            self.eg_values = file["eg_values"][:]
            self.eg_vectors = file["eg_vectors"][:]
            self.resid_var = file["resid_var"][()]
            self.time_grid = file["time_grid"][:]
            self.max_time = file["max_time"][()]
            self.grid_mean = file["grid_mean"][()]

        self.n_bases = len(self.eg_values)
            
    def select_ldrs(self, n_ldrs):
        if n_ldrs is not None:
            if n_ldrs <= self.n_bases:
                self.eg_values = self.eg_values[:n_ldrs]
                self.eg_vectors = self.eg_vectors[:, :n_ldrs]
                self.n_bases = n_ldrs
            else:
                raise ValueError("the number of bases is less than --n-ldrs")
            
    def interpolate(self, new_time):
        new_time = new_time / np.max(new_time)
        interp_eg_vectors = np.zeros_like(self.eg_vectors)
        for j in range(self.n_bases):
            interp_eg_vectors[:, j] = np.interp(new_time, self.time_grid, self.eg_vectors[:, j])
        return interp_eg_vectors
    

def check_input(args):
    # required arguments
    if args.pheno is None:
        raise ValueError("--pheno is required")


def run(args, log):
    check_input(args)

    # read phenotype
    log.info(f"Read a longitudinal phenotype from {args.pheno}")
    pheno = ds.LongiPheno(args.pheno)
    log.info(f"{pheno.n_sub} unique subjects and {pheno.n_obs} observations.")

    # keep common subjects
    common_idxs = ds.get_common_idxs(pheno.data.index, args.keep)
    common_idxs = ds.remove_idxs(common_idxs, args.remove)
    log.info(f"{len(common_idxs.unique())} subjects after filtering (if applicable).")
    pheno.keep_and_remove(common_idxs)

    # estimation
    log.info("\nDoing FPCA ...")
    pheno.generate_time_features()

    mean_estimator = Mean(pheno)
    if args.mean_bw is None:
        mean_bw = mean_estimator.gcv()
    else:
        mean_bw = args.mean_bw
        log.info(f"Using the user provided bandwidth {mean_bw} for estimating mean function.")

    mean, grid_mean = mean_estimator.estimate(mean_bw)
    
    cov_estimator = Covariance(pheno, mean, pheno.time_idx)
    grid_cov, cut_time_grid_diag = cov_estimator.estimate(0.1)

    resid_var_estimator = ResidualVariance(pheno)
    resid_var = resid_var_estimator.estimate(
        mean, cut_time_grid_diag, pheno.time_idx, 0.1
    )

    # eigen
    eg_values, eg_vectors = eigen(grid_mean, grid_cov, pheno.grid_size, pheno.time_grid)

    # save
    with h5py.File(f"{args.out}_fpca.h5", "w") as file:
        file.create_dataset("eg_values", data=eg_values, dtype="float32")
        file.create_dataset("eg_vectors", data=eg_vectors, dtype="float32")
        file.create_dataset("resid_var", data=resid_var, dtype="float32")
        file.create_dataset("time_grid", data=pheno.time_grid, dtype="float32")
        file.create_dataset("max_time", data=pheno.max_time, dtype="float32")
        file.create_dataset("grid_mean", data=grid_mean, dtype="float32")

    log.info(f"\nSaved FPCA results to {args.out}_fpca.h5")