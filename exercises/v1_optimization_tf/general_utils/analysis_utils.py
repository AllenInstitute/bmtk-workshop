
import numpy as np
from scipy import stats

def freedman_diaconis(data, returnas="width"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width. 
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively. 


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if returnas=="width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    
    return result

def freedman_diaconis_bin_sizes(data, returnas="bins"):
    """
    Use the Freedman-Diaconis rule to compute optimal histogram bin sizes along each dimension.

    Parameters
    ----------
    data: np.ndarray
        Two-dimensional array with shape (N, 2).
    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin along each axis.
        If "bins", return the number of bins suggested by the rule along each axis.

    Returns
    -------
    result: tuple
        If returnas=="width", returns (width_x, width_y).
        If returnas=="bins", returns (bins_x, bins_y).
    """
    data = np.asarray(data, dtype=np.float_)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Data should be a 2D array with shape (N, 2)")

    N = data.shape[0]
    widths = []
    bins = []
    for dim in range(2):
        data_dim = data[:, dim]
        IQR = stats.iqr(data_dim, rng=(25, 75), scale=1.0, nan_policy="omit")
        bw = (2 * IQR) / np.power(N, 1/3)  # Bin width along this dimension

        datmin, datmax = data_dim.min(), data_dim.max()
        datrng = datmax - datmin

        # Ensure bin width is positive to avoid division by zero
        if bw > 0:
            bin_num = int(np.ceil(datrng / bw))
        else:
            bin_num = 1  # If data is constant, use one bin

        widths.append(bw)
        bins.append(bin_num)

    return tuple(bins) if returnas == "bins" else tuple(widths)
