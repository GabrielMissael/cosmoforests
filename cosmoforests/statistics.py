"""Compute statistics on the data."""

import numpy as np # Array manipulation
from cosmoforests.mask_forest import mask_forest
from cosmoforests.io import get_qso_data

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
    # Mask -2 values
    masked = np.ma.masked_equal(fluxes, -2.0)

    # Get mean for each wavelength bin
    mean = np.mean(masked, axis = 0)

    # Get standard deviation for each wavelength bin
    std = np.std(masked, axis = 0)

    # Number of elements in each wavelength bin
    n = np.sum(masked.mask == False, axis = 0)

    if get_deltas:
        # Replace -2 with 1
        fluxes = np.where(fluxes == -2.0, 1.0, fluxes)

        # Compute the deltas
        deltas = fluxes/mean - 1.0

        return mean, std, n, deltas
    
    return mean, std, n

def general_stats(fits_path, bin_size = 0.8, save = None, load = False, lambda_min = 1420.0,
                 lambda_max = 1520.0, verbose = False, forest_type = 'F_METALS'):

    """Compute the general statistics for all the data files.

    This function computes the mean and standard deviation for all the data files.

    Parameters
    ----------
    fits_path : list
        List of paths to the data files.
    bin_size : float
        The size of the wavelength bins.
    save : str
        The path to save the statistics to. If None, the statistics are not saved.
        Default is None.
    load : bool
        Whether to load the statistics from a file. If True, the statistics are loaded
        from the file specified by the save parameter. Default is False.
    lambda_min : float
        The minimum wavelength to mask to. Default is 1420.0 (CIV forest).
    lambda_max : float
        The maximum wavelength to mask to. Default is 1520.0 (CIV forest).
    verbose : bool
        Whether to print the progress. Default is False.
    forest_type : str
        The type of forest to mask to. Default is 'F_METALS' (CIV forest).
        Options: 'F_LYA', 'F_LYB', 'F_METALS'

    Returns
    -------
    wv : numpy.ndarray
        The wavelength bins.
    mean : numpy.ndarray
        The mean of the fluxes.
    std : numpy.ndarray
        The standard deviation of the fluxes.
    total_qsos : int
        The total number of quasars.
    n_files : int
        The number of files processed.
    """


    if not load:
        # Number of .fits files
        n_files = len(fits_path)

        # Get wavelenght
        _, wv, _ = get_qso_data(fits_path[0], bin_size = bin_size, verbose = False)

        # Number of bins
        n_bins = len(wv)

        # Initialize arrays to store the statistics
        mean = np.zeros((n_files, n_bins))
        std = np.zeros((n_files, n_bins))
        n_qso = np.zeros((n_files, n_bins))

        # Loop over the files
        i = 0
        for path in fits_path:
            # Get the data
            z, wv, fluxes = get_qso_data(path, bin_size = bin_size, verbose = False,  forest_type = forest_type)

            # Mask the flux data to a given wavelength range. Default to CIV forest.
            fluxes = mask_forest(z, wv, fluxes, lambda_min = lambda_min, lambda_max = lambda_max)

            # Get statistics for the transmission files.
            mean[i], std[i], n_qso[i] = get_statistics(fluxes, get_deltas = False)

            # Increment the counter
            i += 1

            if verbose:
                if i%50 == 0:
                    print(f'Finished {i} files out of {n_files}.')
        
        # Compute the overall mean

        total_mean = np.zeros(n_bins)
        total_qsos = np.sum(n_qso, axis = 0)

        for i in range(n_files):
            total_mean += mean[i] * n_qso[i]

        total_mean /= total_qsos

        # Compute the overall standard deviation

        total_std = np.zeros(n_bins)

        for i in range(n_files):
            total_std += n_qso[i] * (std[i]**2 + (mean[i] - total_mean)**2)

        total_std = np.sqrt(total_std / total_qsos)

        if save != None:
            # Stack horizontally
            data = np.hstack((wv.reshape(-1, 1), total_mean.reshape(-1, 1), total_std.reshape(-1, 1)))

            # Save the data to a .csv file
            np.savetxt(save, data, delimiter = ',')

            # Remove extension from file name
            save = save.split('.')[0]

            # Save total number of QSOs in txt file
            with open(save + '_info.txt', 'w') as f:
                f.write(f'Total number of QSOs: {total_qsos}\n')
                f.write(f'Total number of files: {n_files}\n')
                f.write(f'Wavelength range: {lambda_min} - {lambda_max}\n')
                f.write(f'Bin size: {bin_size}\n')

    else:

        if save == None:
            raise ValueError('No file specified to load from.')
        
        else:
            # Load the data from the .csv file
            try:
                data = np.loadtxt(save, delimiter = ',')
            except:
                raise ValueError('CSV file not found.')
            wv = data[:, 0]
            total_mean = data[:, 1]
            total_std = data[:, 2]

        n_files = len(fits_path)

    return wv, total_mean, total_std, n_files