import numpy as np
import struct
import nitime.timeseries as nts
import nitime.analysis as nta
import MRS.utils as ut
import MRS.analysis as ana


def float_from_str(my_str):
    """
    Cast a string into a list of floats robustly, stripping white-spaces
    """
    out = []
    for x in my_str.split(' '):
        try:
            out.append(float(x.translate(None, ' ')))
            go_on = True
            while go_on:
                if not out.count(''):
                    go_on = False
                else:
                    out.remove('')
        except ValueError:
            out.append(x.translate(None, ''))

    # Try to get just a bare float if you can:
    if len(out) == 1:
        return out[0]
    else:
        return out


def read_rda(fname):
    """
    Read Siemens RDA data from file

    Parameters
    ----------
    fname :

    Returns
    -------
    hdr_dict : a dict with the fields of the RDA file-header
    data : ndarray of complex dtype


    """
    f = file(fname, 'r')
    hdr_end = '>>> End of header <<<'

    hdr_dict = {}
    # Skip the first line of the header (which is just a start line):
    l = f.readline()
    # Go ahead and read the first line of the actual header:
    l = f.readline()
    while not l.startswith(hdr_end):
        split = l.split(':')
        hdr_dict[split[0]] = float_from_str(split[1])
        l = f.readline()

    # The rest of the file contains the complex time-series
    data = np.fromfile(f)
    f.close()
    # Sort into real and imaginary:
    data = np.array(data[::2] + data[1::2] *1j)

    return hdr_dict, data


def analyze_rda(fname1, fname2):
    """

    """
    hdr1, data1 = read_rda(fname1)
    hdr2, data2 = read_rda(fname2)
    dur1 = hdr1['DwellTime']
    dur2 = hdr2['DwellTime']
    ts1 = nts.TimeSeries(data1, duration=dur1, time_unit='ms')
    ts2 = nts.TimeSeries(data2, duration=dur2, time_unit='ms')

    f, c1 = ana.get_spectra(ts1,  dict(lb=10, filt_order=256))
    f, c2 = ana.get_spectra(ts2,  dict(lb=10, filt_order=256))
    f_ppm = ut.freq_to_ppm(f, hz_per_ppm=hdr1['MRFrequency'])

    return f_ppm, c1, c2


def affine_from_hdr(hdr, R=0, A=0, S=1):
    """
    
    """

    mm_per_vox = [hdr['PixelSpacingRow'],
                  hdr['PixelSpacingCol'],
                  hdr['PixelSpacingCol']]
    
    roi_loc = [hdr['VOIPositionCor'],
               hdr['VOIPositionSag'],
               hdr['VOIPositionTra']]
    
    image_tlhc = np.array((-roi_loc[0] - mm_per_vox[0]/2.,
                           roi_loc[1] + mm_per_vox[1]/2.,
                           roi_loc[2] - mm_per_vox[1]/2.))
    
    image_trhc = image_tlhc - np.array([mm_per_vox[0], 0., 0.])
    image_brhc = image_trhc + np.array([0., mm_per_vox[1], 0.])

    lr_diff = image_trhc - image_tlhc
    si_diff = image_trhc - image_brhc

    if not np.all(lr_diff==0) and not np.all(si_diff==0):
        row_cos =  lr_diff / np.sqrt(lr_diff.dot(lr_diff))
        col_cos = -si_diff / np.sqrt(si_diff.dot(si_diff))
    else:
        row_cos = np.array([1.,0,0])
        col_cos = np.array([0,-1.,0])

    slice_norm = np.array([R, A, S])
    
    rot = np.array(((-row_cos[0], -col_cos[0], -slice_norm[0]),
                     (-row_cos[1], -col_cos[1], -slice_norm[1]),
                     (row_cos[2], col_cos[2], slice_norm[2])), dtype=float)

    aff = np.zeros((4,4))
    aff[0:3,0:3] = rot
    aff[:,3] = np.append(image_tlhc, 1).T
    aff[0:3,0:3] = np.dot(aff[0:3,0:3], np.diag(mm_per_vox))
    return aff

