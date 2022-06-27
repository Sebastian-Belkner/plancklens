r"""Pico 2021 E and B, and E and B Noise simulation libaries.

    Note:
        These simulations are located on NERSC systems @
        /project/projectdirs/pico/reanalysis/

    Note:
        Units of the maps stored at NERSC are :math:`K` but this module returns maps in :math:`\mu K`

"""
import healpy as hp
import numpy as np

from plancklens import utils
from astropy.io import fits

class nilc_90b91_nside2048:
    r""" NILC 2021 simulation library at NERSC.

        Note:
            This now converts all maps to double precision
            (healpy 1.15 changed read_map default type behavior, breaking in a way that is not very clear as yet the behavior of the conjugate gradient inversion chain)
    """
    def __init__(self):
        self.emap = '/project/projectdirs/pico/reanalysis/nilc/ns2048/py91_ns2048_%04d/NILC_PICO91_E_reso8acm.fits'
        self.bmap = '/project/projectdirs/pico/reanalysis/nilc/ns2048/py91_ns2048_%04d/NILC_PICO91_B_reso8acm.fits'
        
        self.nemap = '/project/projectdirs/pico/reanalysis/nilc/ns2048/py91_ns2048_%04d/NILC_NOISE_PICO91_E_reso8acm.fits'
        self.nbmap = '/project/projectdirs/pico/reanalysis/nilc/ns2048/py91_ns2048_%04d/NILC_NOISE_PICO91_B_reso8acm.fits'
         
        self.data = None
        

    def hashdict(self):

        return {'cmbs':(self.emap, self.bmap), 'noise':(self.nemap, self.nbmap), 'data':self.data}


    def get_sim_tmap(self, idx):
        assert 0, 'To be implemented'
        r"""Returns pico 90b91 NILC temperature map for a simulation

            Args:
                idx: simulation index

            Returns:
                NILC simulation *idx*, including noise. Returns pico 90b91 NILC data map for *idx* =-1

        """

        if idx == -1:
            return self.get_dat_tmap()
        return None
        # return 1e6 * (hp.read_map(self.cmbs % idx, field=0, dtype=np.float64) + hp.read_map(self.noise % idx, field=0, dtype=np.float64))

    def get_dat_tmap(self):
        assert 0, 'To be implemented'

        return 1e6 * hp.read_map(self.data, field=0, dtype=np.float64)


    def get_tf(self, lmax, nside):

        ret = hp.gauss_beam(np.radians(8/60), lmax=lmax) * hp.pixwin(nside=nside, lmax=lmax) 
        return ret


    def get_sim_pmap(self, idx):
        r"""Returns dx12 SMICA polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                SMICA Q and U simulation *idx*, including noise. Returns dx12 SMICA data maps for *idx* =-1

        """
        lmax = 4096
        nside = 2048
        trsf = self.get_tf(lmax=lmax, nside=nside)
        if idx == -1:
            return self.get_dat_pmap()
        E = fits.open(self.emap%idx)[0].data
        B = fits.open(self.bmap%idx)[0].data
        elm = hp.almxfl(hp.map2alm(E, lmax=lmax), trsf)
        blm = hp.almxfl(hp.map2alm(B, lmax=lmax), trsf)

        _, Q, U = hp.alm2map([np.zeros_like(elm), elm, blm], nside=nside)

        return Q, U


    def get_noise_sim_pmap(self, idx):
        
        r"""Returns dx12 SMICA polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                SMICA Q and U simulation *idx*, including noise. Returns dx12 SMICA data maps for *idx* =-1

        """
        lmax = 4096
        nside = 2048
        trsf = hp.gauss_beam(np.radians(8/60), lmax=lmax) * hp.pixwin(nside=nside, lmax=lmax) 
        if idx == -1:
            return self.get_dat_pmap()
        NE = fits.open(self.nemap%idx)[0].data
        NB = fits.open(self.nbmap%idx)[0].data
        nelm = hp.almxfl(hp.map2alm(NE, lmax=lmax), trsf)
        nblm = hp.almxfl(hp.map2alm(NB, lmax=lmax), trsf)

        _, Q, U = hp.alm2map([np.zeros_like(nelm), nelm, nblm], nside=nside)

        return Q, U


    def get_dat_pmap(self):
        assert 0, 'To be implemented'
        
        return 1e6 * hp.read_map(self.data, field=1, dtype=np.float64), 1e6 * hp.read_map(self.data, field=2, dtype=np.float64)


class ffp10cmb_widnoise:
    r"""Simulation library with freq-0 FFP10 lensed CMB together with idealized, homogeneous noise.

        Args:
            transf: transfer function (beam and pixel window)
            nlevt: temperature noise level in :math:`\mu K`-arcmin.
            nlevp: polarization noise level in :math:`\mu K`-arcmin.
            pix_libphas: random phases simulation library (see plancklens.sims.phas.py) of the noise maps.

    """
    def __init__(self, transf, nlevt, nlevp, pix_libphas, nside=2048):
        assert pix_libphas.shape == (hp.nside2npix(nside),), pix_libphas.shape
        self.nlevt = nlevt
        self.nlevp = nlevp
        self.transf = transf
        self.pix_libphas = pix_libphas
        self.nside = nside

    def hashdict(self):
        return {'transf':utils.clhash(self.transf), 'nlevt':np.float32(self.nlevt), 'nlevp':np.float32(self.nlevp),
                'pix_phas':self.pix_libphas.hashdict()}

    def get_sim_tmap(self, idx):
        T = hp.alm2map(hp.almxfl(cmb_len_ffp10.get_sim_tlm(idx), self.transf), self.nside)
        nlevt_pix = self.nlevt / np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) / 60.
        T += self.pix_libphas.get_sim(idx, idf=0) * nlevt_pix
        return T

    def get_sim_pmap(self, idx):
        elm = hp.almxfl(cmb_len_ffp10.get_sim_elm(idx), self.transf)
        blm = hp.almxfl(cmb_len_ffp10.get_sim_blm(idx), self.transf)
        Q, U = hp.alm2map_spin((elm, blm), self.nside, 2, hp.Alm.getlmax(elm.size))
        del elm, blm
        nlevp_pix = self.nlevp / np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) / 60.
        Q += self.pix_libphas.get_sim(idx, idf=1) * nlevp_pix
        U += self.pix_libphas.get_sim(idx, idf=2) * nlevp_pix
        return Q, U

class cmb_len_ffp10:
    """ FFP10 input sim libraries, lensed alms.

        The lensing deflections contain the L=1 aberration term (constant across all maps)
        due to our motion w.r.t. the CMB frame.

    """
    def __init__(self):
        pass

    def hashdict(self):
        return {'sim_lib': 'ffp10 lensed scalar cmb inputs, freq 0'}

    @staticmethod
    def get_sim_tlm(idx):
        """
            Args:
                idx: simulation index

            Returns:
                lensed temperature simulation healpy alm array

        """
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=1)

    @staticmethod
    def get_sim_elm(idx):
        """
            Args:
                idx: simulation index

            Returns:
                lensed E-polarization simulation healpy alm array

        """
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=2)

    @staticmethod
    def get_sim_blm(idx):
        """
            Args:
                idx: simulation index

            Returns:
                lensed B-polarization simulation healpy alm array

        """
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=3)