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

def get_qso_data(file_path):
    """
    Get the wavelength, flux and redshift data from a FITS file.
    
    Parameters
    ----------
    file_path : str
        The path to the FITS file.
    
    Returns
    -------
    z: numpy.ndarray
        The redshift of the quasars.
    wv : numpy.ndarray
        The wavelength data.
    flux : numpy.ndarray
        The flux data.
    """

    # Read the FITS file
    hdul = fits.open(file_path)

    # Get wavelength and flux data
    wv = np.array(hdul['WAVELENGTH'].data)
    fluxes = np.array(hdul['F_METALS'].data)
    # Redshifts
    z = np.array(hdul['METADATA'].data['Z'])

    # Close the FITS file
    hdul.close()

    return z, wv, fluxes
