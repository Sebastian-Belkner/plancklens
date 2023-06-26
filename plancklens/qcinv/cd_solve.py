"""Flexible conjugate directions solver module.

"""
import os
import numpy as np

from plancklens.qcinv.util_alm import eblm

def PTR(p, t, r):
    return lambda i: max(0, i - max(p, int(min(t, np.mod(i, r)))))


tr_cg = (lambda i: i - 1)
tr_cd = (lambda i: 0)


class cache_mem(dict):
    def __init__(self):
        pass

    def store(self, key, data):
        [dTAd_inv, searchdirs, searchfwds] = data
        self[key] = [dTAd_inv, searchdirs, searchfwds]

    def restore(self, key):
        return self[key]

    def remove(self, key):
        del self[key]

    def trim(self, keys):
        assert (set(keys).issubset(self.keys()))
        for key in (set(self.keys()) - set(keys)):
            del self[key]


def cd_solve(x, b, fwd_op, pre_ops, dot_op, criterion, tr, cache=cache_mem(), roundoff=25):
    """customizable conjugate directions loop for x=[fwd_op]^{-1}b.

    Args:
        x (array-like)              :Initial guess of linear problem  x =[fwd_op]^{-1}b.  Contains converged solution
                                at the end (if successful).
        b (array-like)              :Linear problem  x =[fwd_op]^{-1}b input data.
        fwd_op (callable)           :Forward operation in x =[fwd_op]^{-1}b.
        pre_ops (list of callables) :Pre-conditioners.
        dot_op (callable)           :Scalar product for two vectors.
        criterion (callable)        :Decides convergence.
        tr                          :Truncation / restart functions. (e.g. use tr_cg for conjugate gradient)
        cache (optional)            :Cacher for search objects. Defaults to cache in memory 'cache_mem' instance.
        roundoff (int, optional)    :Recomputes residual by brute-force every *roundoff* iterations. Defaults to 25.

    Note:
        fwd_op, pre_op(s) and dot_op must not modify their arguments!

    """


    def _preproc(mask, res):
        # res.elm = hp.map2alm(hp.alm2map(res.elm, nside=2048)*mask, lmax=3000)
        return res

    # import healpy as hp
    # from delensalot.config.config_helper import data_functions as df
    # mask_map = hp.read_map('/mnt/c/Users/sebas/OneDrive/SCRATCH/delensalot/generic/sims_cmb_len_lminB200_mfda_maskedsky_center/mask.fits')
    # mask_lm = hp.map2alm(mask_map, lmax=3000)
    # mask_lm_smoothed = hp.smoothalm(mask_lm, fwhm=df.a2r(40))
    # mask_map_smoothed = hp.alm2map(mask_lm_smoothed, nside=2048)
    # mask_map_smoothed_binary = np.where(mask_map_smoothed>0.9,1,0)

    n_pre_ops = len(pre_ops)

    residual = b - fwd_op(x)
    searchdirs = [op(residual) for op in pre_ops]
    # print("lmax of residual:".format(residual.lmax))

    iter = 0
    while not criterion(iter, x, _preproc(None, residual)): # _preproc(mask_map_smoothed_binary, residual)
        path = '/mnt/c/Users/sebas/OneDrive/SCRATCH/delensalot/generic/sims_cmb_len_lminB200_mfda_rhitssky_center/cgsolver/'
        fn = 'residual_{}'.format(iter)
        
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if type(residual) == type(eblm([[],[]])):
            np.save(path+fn+'_qe_elm', residual.elm)
            np.save(path+fn+'_qe_blm', residual.blm)
        else:
            np.save(path+fn+'_it1', residual)
        searchfwds = [fwd_op(searchdir) for searchdir in searchdirs]
        deltas = [dot_op(searchdir, residual) for searchdir in searchdirs]

        # calculate (D^T A D)^{-1}
        dTAd = np.zeros((n_pre_ops, n_pre_ops))
        for ip1 in range(0, n_pre_ops):
            for ip2 in range(0, ip1 + 1):
                dTAd[ip1, ip2] = dTAd[ip2, ip1] = dot_op(searchdirs[ip1], searchfwds[ip2])
        dTAd_inv = np.linalg.inv(dTAd)

        # search.
        alphas = np.dot(dTAd_inv, deltas)
        for (searchdir, alpha) in zip(searchdirs, alphas):
            x += searchdir * alpha

        # append to cache.
        cache.store(iter, [dTAd_inv, searchdirs, searchfwds])

        # update residual
        iter += 1
        if np.mod(iter, roundoff) == 0:
            residual = b - fwd_op(x)
        else:
            for (searchfwd, alpha) in zip(searchfwds, alphas):
                residual -= searchfwd * alpha

        # initial choices for new search directions.
        searchdirs = [pre_op(residual) for pre_op in pre_ops]

        # orthogonalize w.r.t. previous searches.
        prev_iters = range(tr(iter), iter)

        for titer in prev_iters:
            [prev_dTAd_inv, prev_searchdirs, prev_searchfwds] = cache.restore(titer)

            for searchdir in searchdirs:
                proj = [dot_op(searchdir, prev_searchfwd) for prev_searchfwd in prev_searchfwds]
                betas = np.dot(prev_dTAd_inv, proj)

                for (beta, prev_searchdir) in zip(betas, prev_searchdirs):
                    searchdir -= prev_searchdir * beta

        # clear old keys from cache
        cache.trim(range(tr(iter + 1), iter))

    return iter
