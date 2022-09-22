"""Functions for reading data"""

# Library imports
import os # File management
from astropy.io import fits # Astropy for FITS files
import numpy as np # Array manipulation

def get_docs_path(data_dir, verbose = True, search = '.fits'):
    """
    Get the path to files in the data directory with a given extension or string.

    Parameters
    ----------
    data_dir : str
        The path to the data directory.
    verbose : bool
        Whether to print the number of FITS files found, and the path format. Default is True.
    search : str
        The string to search for in the file names. Default is '.fits'.
    
    Returns
    -------
    fits_path : list of str
        The path to the FITS files in the data directory.
    """
    # List of files in the data directory
    fits_path = []

    # Loop through the files in the data directory
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            # If the file is a FITS file, add it to the list of FITS files
            if search in name:
                fits_path.append(os.path.join(root, name))

    if verbose:
        print(f'Number of .fits files = {len(fits_path)}')
        print(f'Path example: {fits_path[0]}')

    return fits_path

def get_qso_data(file_path, bin_size = None, verbose = False, forest_type = 'F_METALS'):
    """
    Get the wavelength, flux and redshift data from a FITS file.
    
    Parameters
    ----------
    file_path : str
        The path to the FITS file.
    bin_size : float
        The size of the wavelength bins.
        Default is None (not changing the bin size).
    verbose : bool
        Whether to print shapes of the data. Default is False.
    forest_type : str
        The type of forest to mask to. Default is 'F_METALS' (CIV forest).
        Options: 'F_LYA', 'F_LYB', 'F_METALS'
    
    Returns
    -------
    z: numpy.ndarray
        The redshift of the quasars.
    wv : numpy.ndarray
        The wavelength data.
    fluxes : numpy.ndarray
        The flux data.
    """

    # Read the FITS file
    hdul = fits.open(file_path)

    # Get wavelength and flux data
    wv = np.array(hdul['WAVELENGTH'].data)
    fluxes = np.array(hdul[forest_type].data)
    # Redshifts
    z = np.array(hdul['METADATA'].data['Z'])

    # Close the FITS file
    hdul.close()

    # If the bin size is not None, bin the data
    if bin_size is not None:

        # Change bin size
        new_bin_size = bin_size

        # Original bin size with one decimal place
        original_bin_size = wv[1] - wv[0]
        original_bin_size = round(original_bin_size, 1)

        # Number of bins to average over
        stack_number = new_bin_size / original_bin_size

        # Check if the number of bins to average over is an integer
        if abs(stack_number - round(stack_number)) > 0.00001:
            raise ValueError(f'New bin size {new_bin_size} must be a'\
                +f' multiple of the original bin size {original_bin_size:.3f}')

        # Ceil to first integer
        stack_number = int(round(stack_number))

        # New wavelength array
        wv = np.arange(wv[0], wv[-1] + bin_size, bin_size)

        # Remove extra bins from the fluxes
        if verbose:
            print(f'Original number of bins in flux: {len(fluxes[0])}')
        remove = len(fluxes[0]) % stack_number
        if remove != 0:
            fluxes = fluxes[:, :-remove]

        if verbose:
            print(f'New number of bins in flux: {len(fluxes[0])}')

        # Reshape the flux array
        if verbose:
            print(f'Original shape: {fluxes.shape}')
        fluxes = fluxes.reshape(len(fluxes), -1, stack_number)
        if verbose:
            print(f'New shape: {fluxes.shape}')

        # Average over the last axis
        fluxes = np.mean(fluxes, axis=-1)
        if verbose:
            print(f'Final shape: {fluxes.shape}')
        
        # Make wv and fluxes the same shape
        n_bins = len(fluxes[0])
        wv = wv[:n_bins]

    return z, wv, fluxes

def print_rebin_example():
    example = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9]])

    print('Original array:')
    print(example)
    print(example.shape)

    print('Reshaped array:')
    stack_number = 3
    example = example.reshape(2, -1, stack_number)
    print(example)
    print(example.shape)

    print('Averaged array:')
    print(np.mean(example, axis=-1))
    print(np.mean(example, axis=-1).shape)
