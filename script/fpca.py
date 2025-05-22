import h5py
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


class LocalLinear(ABC):
    """
    Abstract class for local linear estimator.
    
    """
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
        self.n_time = len(self.unique_time)
        self.n_grids = len(self.time_grid)

    @staticmethod
    def _gau_kernel(x):
        """
        Calculating the Gaussian density

        """
        gau_k = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
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
    

class Mean(LocalLinear):
    def _get_design_matrix(self, t, bw):
        time_diff = self.time - t
        weights = self._gau_kernel(time_diff / bw)
        time_diff = time_diff.reshape(-1, 1)
        x = np.hstack([np.ones_like(time_diff), time_diff])
        return x, weights
    
    def estimate(self, bw):
        mean_function = np.zeros(self.n_time, dtype=np.float32)
        grid_mean_function = np.zeros(self.n_grids, dtype=np.float32)
        
        for i, t in enumerate(self.unique_time):
            x, weights = self._get_design_matrix(t, bw)
            mean_function[i] = self._wls(x, self.pheno, weights)

        for i, t in enumerate(self.time_grid):
            x, weights = self._get_design_matrix(t, bw)
            grid_mean_function[i] = self._wls(x, self.pheno, weights)

        return mean_function, grid_mean_function   
    

class Covariance(LocalLinear):
    def __init__(self, pheno, mean, time_idx):
        super().__init__(pheno)
        self.n_obs_adj = np.repeat(
            1 / (self.n_obs * (self.n_obs - 1)), self.n_obs * (self.n_obs - 1)
        ).reshape(-1, 1)
        
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

        self.unique_time_comb = np.zeros((self.n_time ** 2 - self.n_time, 2), dtype=np.float32)
        self.mean_pheno = np.zeros(self.n_time ** 2 - self.n_time, dtype=np.float32)
        self.time_comb_count = np.zeros((self.n_time ** 2 - self.n_time, 1), dtype=np.int32)
        k = 0
        for i in range(self.n_time):
            for j in range(self.n_time):
                if i != j:
                    t1 = self.unique_time[i]
                    t2 = self.unique_time[j]
                    self.unique_time_comb[k] = np.array([t1, t2])
                    time_comb_idx = (self.two_way_time == self.unique_time_comb[k]).all(axis=1)
                    self.time_comb_count[k] = np.sum(time_comb_idx)
                    self.mean_pheno[k] = np.mean(self.two_way_pheno[time_comb_idx], axis=0)
                    k += 1

    def _get_design_matrix(self, t1, t2, bw):
        """
        TODO: set large distances as 0.
        
        """
        time_diff = self.unique_time_comb - np.array([t1, t2])
        weights = self._gau_kernel(time_diff / bw) * self.time_comb_count
        x = np.hstack([np.ones(time_diff.shape[0], dtype=np.float32).reshape(-1, 1), time_diff])
        return x, weights
    
    def _get_design_matrix2(self, time, t, bw):
        """
        local quadratic
        
        """
        time_diff = time - t
        weights = self._gau_kernel(time_diff / bw) * self.time_comb_count
        time_diff[:, 1] = time_diff[:, 1] ** 2
        x = np.hstack([np.ones(time_diff.shape[0], dtype=np.float32).reshape(-1, 1), time_diff])
        return x, weights
    
    def estimate(self, bw):
        grid_cov_function = np.zeros((self.n_grids, self.n_grids), dtype=np.float32)

        for t1 in range(self.n_grids):
            for t2 in range(t1, self.n_grids):
                x, weights = self._get_design_matrix(self.time_grid[t1], self.time_grid[t2], bw)
                grid_cov_function[t1, t2] = self._wls(x, self.mean_pheno, weights)
        
        iu_rows, iu_cols = np.triu_indices(self.n_grids, k=1)
        grid_cov_function[(iu_cols, iu_rows)] = grid_cov_function[(iu_rows, iu_cols)]

        cut_time_grid = self.time_grid[
            (self.time_grid > np.quantile(self.time_grid, 0.25)) & 
            (self.time_grid < np.quantile(self.time_grid, 0.75))
        ]
        cut_time_grid = np.tile(cut_time_grid.reshape(-1, 1), 2)
        n_cut_time_grid = cut_time_grid.shape[0]
        rotation_matrix = np.array([[1, 1], [-1, 1]]).T * (np.sqrt(2) / 2)
        rotated_time_comb = np.dot(self.unique_time_comb, rotation_matrix)
        rotated_cut_time_grid = np.dot(cut_time_grid, rotation_matrix)
        cut_time_grid_diag = np.zeros(n_cut_time_grid, dtype=np.float32)
        for t in range(n_cut_time_grid):
            x, weights = self._get_design_matrix2(
                rotated_time_comb, 
                rotated_cut_time_grid[t],
                0.1
            )
            cut_time_grid_diag[t] = self._wls(x, self.mean_pheno, weights)

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
            self.resid_var = file["resid_var"][:]
            self.time_grid = file["time_grid"][:]

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
    log.info("Doing FPCA ...")
    pheno.generate_time_features()

    mean_estimator = Mean(pheno)
    mean, grid_mean = mean_estimator.estimate(0.05)
    
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

    log.info(f"\nSaved FPCA results to {args.out}_fpca.h5")