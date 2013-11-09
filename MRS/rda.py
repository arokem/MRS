import numpy as np
import struct
import nitime.timeseries as nts

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
