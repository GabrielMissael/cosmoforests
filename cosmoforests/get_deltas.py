import fitsio
from cosmoforests.mask_forest import get_mask
from cosmoforests.io import get_qso_data
from cosmoforests.mask_forest import get_mask
import numpy as np
from astropy.io import fits
import os

def get_delta_path(fits_path, deltas_dir, data_dir):
    delta_path = fits_path.replace(data_dir, '')
    delta_path = delta_path.split('/')[-1]
    delta_path = delta_path.replace('transmission-16', 'delta').split('.')[0]
    delta_path += '.fits'
    delta_path = os.path.join(deltas_dir, delta_path)

    return delta_path

def delta_files_from_transmission(fits_path, deltas_dir, data_dir, total_mean, bin_size):

    # Get QSO data
    z, wv, fluxes = get_qso_data(fits_path, bin_size = bin_size, forest_type = 'F_METALS')

    loglam = np.log10(wv)
    mask = get_mask(wv, z)
    hdul = fits.open(fits_path)

    # Create fits file for deltas
    delta_path = get_delta_path(fits_path, deltas_dir, data_dir)

    # If the file already exists, delete it
    if os.path.exists(delta_path):
        os.remove(delta_path)

    # Check if directory exists
    if not os.path.exists(os.path.dirname(delta_path)):
        os.makedirs(os.path.dirname(delta_path))

    results = fitsio.FITS(delta_path, 'rw', clobber=True)

    for i in range(len(fluxes)):

        deltas = fluxes[i]/total_mean -1

        header = [{'name': 'RA',  'value': np.radians(hdul['METADATA'].data['RA'][i]), 'comment': 'Right Ascension [rad]'},
            {'name': 'DEC', 'value': np.radians(hdul['METADATA'].data['DEC'][i]), 'comment': 'Declination [rad]'},
            {'name': 'Z',   'value': hdul['METADATA'].data['Z'][i], 'comment': 'Redshift'},
            {'name': 'TARGETID', 'value':hdul['METADATA'].data['MOCKID'][i], 'comment': 'Target ID'},
            {'name': 'THING_ID', 'value':i, 'comment': 'Thing ID (fake)'},
            {'name': 'PLATE', 'value':i, 'comment': 'Plate (fake)'},
            {'name': 'MJD', 'value':i, 'comment': 'MJD (fake)'},
            {'name': 'FIBERID', 'value':i, 'comment': 'fiberid (fake)'}]

        weigths = np.ones(np.shape(loglam[mask[i]]))
        cols = [loglam[mask[i]], deltas[mask[i]], weigths, weights]
        names = ['LOGLAM', 'DELTA', 'WEIGHT', 'CONT']
        units    = ['LogAngstrom', '', '', '']

        results.write(cols,
            names   = names,
            header  = header,
            units   = units,
            extname  =  f'File {fits_path}, QSO {i}'
            )

    results.close()
