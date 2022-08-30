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

    # Get wavelength interval of CIV forest
    z_min = np.min(z)
    z_max = np.max(z)
    lambda_min_obs = lambda_min*(1.0 + z)
    lambda_max_obs = lambda_max*(1.0 + z)

    new_fluxes = np.zeros((len(z), n_bins))

    # Remove non-CIV-forest absorptions
    for i, flux in enumerate(fluxes):
        # Masks the CIV forest
        mask = (wv > lambda_min_obs[i]) & (wv < lambda_max_obs[i])
        new_fluxes[i] = flux
        # Makes 1 whatever isn't in the CIV forest
        new_fluxes[i][~mask] = 1.0
    
        #! TODO: Do we need to normalize the fluxes?

    return new_fluxes
