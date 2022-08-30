"""Compute statistics on the data."""

import numpy as np # Array manipulation

def get_statistics(fluxes):
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

    # Compute the deltas
    deltas = fluxes/mean - 1.0

    return mean, std, deltas
