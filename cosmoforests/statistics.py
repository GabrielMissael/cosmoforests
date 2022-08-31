"""Compute statistics on the data."""

import numpy as np # Array manipulation

def get_statistics(fluxes, get_deltas = False):
    """Get statistics for the transmission files.
    
    Parameters
    ----------
    fluxes : numpy.ndarray
        The fluxes to get statistics for.
    
    Returns
    -------
    mean : numpy.ndarray
        The mean of the fluxes.
    std : numpy.ndarray
        The standard deviation of the fluxes.
    deltas: numpy.ndarray
        The deltas of the fluxes.
    """

    # Get mean for each wavelength bin
    mean = np.mean(fluxes, axis = 0)

    # Get standard deviation for each wavelength bin
    std = np.std(fluxes, axis = 0)

    if get_deltas:
        # Compute the deltas
        deltas = fluxes/mean - 1.0

        return mean, std, deltas
    
    return mean, std
