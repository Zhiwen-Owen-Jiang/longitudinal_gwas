import numpy as np
import pandas as pd
import script.dataset as ds
from script.utils import inv
from script.fpca import FPCAres


def projection_ldr(ldr, covar):
    """
    Computing S'(I - M)S/n = S'S - S'X(X'X)^{-1}X'S/n,
    where I is the identity matrix,
    M = X(X'X)^{-1}X' is the project matrix for X,
    S is the LDR matrix.

    Parameters:
    ------------
    ldr (n, r): low-dimension representaion of imaging data
    covar (n, p): covariates, including the intercept

    Returns:
    ---------
    ldr_cov: variance-covariance matrix of LDRs

    """
    n = ldr.shape[0]
    inner_ldr = np.dot(ldr.T, ldr)
    inner_covar = np.dot(covar.T, covar)
    inner_covar_inv = inv(inner_covar)
    ldr_covar = np.dot(ldr.T, covar)
    part2 = np.dot(np.dot(ldr_covar, inner_covar_inv), ldr_covar.T)
    ldr_cov = (inner_ldr - part2) / n
    ldr_cov = ldr_cov.astype(np.float32)

    return ldr_cov


def pace(pheno, fpca_res):
    """
    PACE estimator

    Parameters:
    ------------
    pheno: 
    fpca_res:

    Returns:
    ---------
    time_ldrs:
    
    """
    sub_time = pheno.sub_time
    unique_time = pheno.unique_time
    unique_time_idx = pheno.unique_time_idx
    n_sub = pheno.n_sub
    pheno = pheno.pheno

    resid_var = fpca_res.resid_var
    time_grid = fpca_res.time_grid
    eg_values = fpca_res.eg_values
    eg_vectors = fpca_res.eg_vectors
    grid_mean = fpca_res.grid_mean
    mean = np.interp(unique_time, time_grid, grid_mean)

    n_time = len(unique_time)
    n_ldrs = eg_vectors.shape[1]
    
    interp_eg_vectors = np.zeros((n_time, n_ldrs), dtype=np.float32)
    time_ldrs = np.zeros((n_sub, n_ldrs), dtype=np.float32)
    for j in range(n_ldrs):
        interp_eg_vectors[:, j] = np.interp(unique_time, time_grid, eg_vectors[:, j])

    fitted_cov = np.dot(interp_eg_vectors * eg_values, interp_eg_vectors.T)
    fitted_cov += np.diag([resid_var] * n_time)
    interp_eg_vectors = interp_eg_vectors * eg_values
    start, end = 0, 0
    for sub_idx, (_, time) in enumerate(sub_time.items()):
        end += len(time)
        time_idx = np.array([unique_time_idx[t] for t in time])
        y_i = pheno[start: end] # (n_time_i, )
        mean_i = mean[time_idx] # (n_time_i, )
        Sigma_i_inv = inv(fitted_cov[time_idx][:, time_idx]) # (n_time_i, n_time_i)
        eg_vector = interp_eg_vectors[time_idx] # (n_time_i, n_opt)
        time_ldrs[sub_idx] = np.dot(np.dot(eg_vector.T, Sigma_i_inv), y_i - mean_i)
        start = end

    return time_ldrs


def check_input(args):
    # required arguments
    if args.pheno is None:
        raise ValueError("--pheno is required")
    if args.covar is None:
        raise ValueError("--covar is required")
    if args.fpca_res is None:
        raise ValueError("--fpac-res is required")


def run(args, log):
    check_input(args)

    # read fpca results
    fpca_res = FPCAres(args.fpca_res)
    log.info(f"{fpca_res.n_bases} bases read from {args.fpca_res}")
    fpca_res.select_ldrs(args.n_ldrs)
    
    # read phenotype
    log.info(f"Read a longitudinal phenotype from {args.pheno}")
    pheno = ds.LongiPheno(args.pheno)
    log.info(f"{pheno.n_sub} unique subjects and {pheno.n_obs} observations.")

    # read covariates
    log.info(f"Read covariates from {args.covar}")
    covar = ds.Covar(args.covar, args.cat_covar_list)
    covar.keep_remove_covar(args.keep_covar_list, args.remove_covar_list)

    # keep common subjects
    common_idxs = ds.get_common_idxs(pheno.data.index, covar.data.index, args.keep)
    common_idxs = ds.remove_idxs(common_idxs, args.remove)
    log.info(f"{len(common_idxs)} subjects common in these files.")
    pheno.keep_and_remove(common_idxs)
    pheno.generate_time_features()
    covar.keep_and_remove(common_idxs)
    covar.cat_covar_intercept()
    log.info(
        f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept)."
    )

    # contruct ldrs
    ldrs = pace(pheno, fpca_res)
    log.info(f"Constructed {fpca_res.n_bases} LDRs.")

    # var-cov matrix of projected LDRs
    ldr_cov = projection_ldr(ldrs, np.array(covar.data))
    log.info(
        f"Removed covariate effects from LDRs and computed variance-covariance matrix.\n"
    )

    # save the output
    ldr_df = pd.DataFrame(ldrs, index=covar.get_ids())
    ldr_df.to_csv(f"{args.out}_ldr_top{fpca_res.n_bases}.txt", sep="\t")
    np.save(f"{args.out}_ldr_cov_top{fpca_res.n_bases}.npy", ldr_cov)

    log.info(f"Saved LDRs to {args.out}_ldr_top{fpca_res.n_bases}.txt")
    log.info(
        (
            f"Saved the variance-covariance matrix of covariate-effect-removed LDRs "
            f"to {args.out}_ldr_cov_top{fpca_res.n_bases}.npy"
        )
    )