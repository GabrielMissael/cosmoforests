"""Mask the flux data to a given wavelength range. Default to CIV forest."""

# Import modules
import numpy as np # Array manipulation

def mask_forest(z, wv, fluxes, lambda_min = 1420.0, lambda_max = 1520.0):
    """
    Mask the flux data to a given wavelength range. Default to CIV forest.
        
    Parameters
    ----------
    z : numpy.ndarrays
        The redshift of the quasars.
    wv : numpy.ndarray
        The wavelength data.
    fluxes : numpy.ndarray
        The flux data.
    lambda_min : float
        The minimum wavelength to mask to. Default is 1420.0.
    lambda_max : float
        The maximum wavelength to mask to. Default is 1520.0.
    
    Returns
    -------
    new_fluxes : numpy.ndarray
        The masked flux data.
    """

    # Number of bins
    n_bins = len(wv)

    new_fluxes = np.zeros((len(z), n_bins))
    mask = get_mask(wv, z, lambda_min = lambda_min, lambda_max = lambda_max)

    # Remove non-CIV-forest absorptions
    for i, flux in enumerate(fluxes):
        # Masks the CIV forest
        new_fluxes[i] = flux
        # Makes 1 whatever isn't in the CIV forest
        new_fluxes[i][~mask[i]] = -2.0
    
        #! TODO: Do we need to normalize the fluxes?

    return new_fluxes

def get_mask(wv, z, lambda_min = 1420.0, lambda_max = 1520.0):
    """Create a mask for a given wavelength range."""

    z_min = np.min(z)
    z_max = np.max(z)
    lambda_min_obs = lambda_min*(1.0 + z)
    lambda_max_obs = lambda_max*(1.0 + z)

    mask = []
    for i in range(len(z)):
        mask.append((wv > lambda_min_obs[i]) & (wv < lambda_max_obs[i]))

    return mask