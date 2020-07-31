from skimage._shared.utils import check_shape_equality
from _contingency_table import contingency_table
import numpy as np
import scipy.sparse as sparse
from skimage.util.dtype import dtype_range
from skimage._shared.utils import warn, check_shape_equality

__all__ = ['adapted_rand_error']


def adapted_rand_error(image_true=None, image_test=None, *, table=None,
                       ignore_labels=(0,)):
    r"""Compute Adapted Rand error as defined by the SNEMI3D contest. [1]_
    Parameters
    ----------
    image_true : ndarray of int
        Ground-truth label image, same shape as im_test.
    image_test : ndarray of int
        Test image.
    table : scipy.sparse array in crs format, optional
        A contingency table built with skimage.evaluate.contingency_table.
        If None, it will be computed on the fly.
    ignore_labels : sequence of int, optional
        Labels to ignore. Any part of the true image labeled with any of these
        values will not be counted in the score.
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float
        The adapted Rand precision: this is the number of pairs of pixels that
        have the same label in the test label image *and* in the true image,
        divided by the number in the test image.
    rec : float
        The adapted Rand recall: this is the number of pairs of pixels that
        have the same label in the test label image *and* in the true image,
        divided by the number in the true image.
    Notes
    -----
    Pixels with label 0 in the true segmentation are ignored in the score.
    References
    ----------
    .. [1] Arganda-Carreras I, Turaga SC, Berger DR, et al. (2015)
           Crowdsourcing the creation of image segmentation algorithms
           for connectomics. Front. Neuroanat. 9:142.
           :DOI:`10.3389/fnana.2015.00142`
    """
    if image_test is not None and image_true is not None:
        check_shape_equality(image_true, image_test)

    if table is None:
        p_ij = contingency_table(image_true, image_test,
                                  ignore_labels=ignore_labels, normalize=False)
    else:
        p_ij = table

    # Sum of the joint distribution squared
    sum_p_ij2 = p_ij.data @ p_ij.data - p_ij.sum()

    a_i = p_ij.sum(axis=1).A.ravel()
    b_i = p_ij.sum(axis=0).A.ravel()

    # Sum of squares of the test segment sizes (this is 2x the number of pairs
    # of pixels with the same label in im_test)
    sum_a2 = a_i @ a_i - a_i.sum()
    # Same for im_true
    sum_b2 = b_i @ b_i - b_i.sum()

    precision = sum_p_ij2 / sum_a2
    recall = sum_p_ij2 / sum_b2

    fscore = 2. * precision * recall / (precision + recall)
    are = 1. - fscore

    return are, precision, recall

__all__ = ['variation_of_information']


def variation_of_information(image0=None, image1=None, *, table=None,
                             ignore_labels=()):
    """Return symmetric conditional entropies associated with the VI. [1]_
    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If X is the ground-truth segmentation, then H(X|Y) can be interpreted
    as the amount of under-segmentation and H(X|Y) as the amount
    of over-segmentation. In other words, a perfect over-segmentation
    will have H(X|Y)=0 and a perfect under-segmentation will have H(Y|X)=0.
    Parameters
    ----------
    image0, image1 : ndarray of int
        Label images / segmentations, must have same shape.
    table : scipy.sparse array in csr format, optional
        A contingency table built with skimage.evaluate.contingency_table.
        If None, it will be computed with skimage.evaluate.contingency_table.
        If given, the entropies will be computed from this table and any images
        will be ignored.
    ignore_labels : sequence of int, optional
        Labels to ignore. Any part of the true image labeled with any of these
        values will not be counted in the score.
    Returns
    -------
    vi : ndarray of float, shape (2,)
        The conditional entropies of image1|image0 and image0|image1.
    References
    ----------
    .. [1] Marina Meilă (2007), Comparing clusterings—an information based
        distance, Journal of Multivariate Analysis, Volume 98, Issue 5,
        Pages 873-895, ISSN 0047-259X, :DOI:`10.1016/j.jmva.2006.11.013`.
    """
    h0g1, h1g0 = _vi_tables(image0, image1, table=table,
                            ignore_labels=ignore_labels)
    # false splits, false merges
    return np.array([h1g0.sum(), h0g1.sum()])


def _xlogx(x):
    """Compute x * log_2(x).
    We define 0 * log_2(0) = 0
    Parameters
    ----------
    x : ndarray or scipy.sparse.csc_matrix or csr_matrix
        The input array.
    Returns
    -------
    y : same type as x
        Result of x * log_2(x).
    """
    y = x.copy()
    if isinstance(y, sparse.csc_matrix) or isinstance(y, sparse.csr_matrix):
        z = y.data
    else:
        z = np.asarray(y)  # ensure np.matrix converted to np.array
    nz = z.nonzero()
    z[nz] *= np.log2(z[nz])
    return y


def _vi_tables(im_true, im_test, table=None, ignore_labels=()):
    """Compute probability tables used for calculating VI.
    Parameters
    ----------
    im_true, im_test : ndarray of int
        Input label images, any dimensionality.
    table : csr matrix, optional
        Pre-computed contingency table.
    ignore_labels : sequence of int, optional
        Labels to ignore when computing scores.
    Returns
    -------
    hxgy, hygx : ndarray of float
        Per-segment conditional entropies of ``im_true`` given ``im_test`` and
        vice-versa.
    """
    check_shape_equality(im_true, im_test)

    if table is None:
        # normalize, since it is an identity op if already done
        pxy = contingency_table(
            im_true, im_test,
            ignore_labels=ignore_labels, normalize=True
        )

    else:
        pxy = table

    # compute marginal probabilities, converting to 1D array
    px = np.ravel(pxy.sum(axis=1))
    py = np.ravel(pxy.sum(axis=0))

    # use sparse matrix linear algebra to compute VI
    # first, compute the inverse diagonal matrices
    px_inv = sparse.diags(_invert_nonzero(px))
    py_inv = sparse.diags(_invert_nonzero(py))

    # then, compute the entropies
    hygx = -px @ _xlogx(px_inv @ pxy).sum(axis=1)
    hxgy = -_xlogx(pxy @ py_inv).sum(axis=0) @ py

    return list(map(np.asarray, [hxgy, hygx]))


def _invert_nonzero(arr):
    """Compute the inverse of the non-zero elements of arr, not changing 0.
    Parameters
    ----------
    arr : ndarray
    Returns
    -------
    arr_inv : ndarray
        Array containing the inverse of the non-zero elements of arr, and
        zero elsewhere.
    """
    arr_inv = arr.copy()
    nz = np.nonzero(arr)
    arr_inv[nz] = 1 / arr[nz]
    return arr_inv

__all__ = ['mean_squared_error',
           'normalized_root_mse',
           'peak_signal_noise_ratio',
           ]


def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = np.result_type(image0.dtype, image1.dtype, np.float32)
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1

def mean_squared_error(image0, image1):
    """
    Compute the mean-squared error between two images.
    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.
    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.
    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.
    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def mean_absoluted_error(image0, image1):

    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean(np.abs(image0 - image1), dtype=np.float64)

def normalized_root_mse(image_true, image_test, *, normalization='euclidean'):
    """
    Compute the normalized root mean-squared error (NRMSE) between two
    images.
    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    normalization : {'euclidean', 'min-max', 'mean'}, optional
        Controls the normalization method to use in the denominator of the
        NRMSE.  There is no standard method of normalization across the
        literature [1]_.  The methods available here are as follows:
        - 'euclidean' : normalize by the averaged Euclidean norm of
          ``im_true``::
              NRMSE = RMSE * sqrt(N) / || im_true ||
          where || . || denotes the Frobenius norm and ``N = im_true.size``.
          This result is equivalent to::
              NRMSE = || im_true - im_test || / || im_true ||.
        - 'min-max'   : normalize by the intensity range of ``im_true``.
        - 'mean'      : normalize by the mean of ``im_true``
    Returns
    -------
    nrmse : float
        The NRMSE metric.
    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_nrmse`` to
        ``skimage.metrics.normalized_root_mse``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    check_shape_equality(image_true, image_test)
    image_true, image_test = _as_floats(image_true, image_test)

    # Ensure that both 'Euclidean' and 'euclidean' match
    normalization = normalization.lower()
    if normalization == 'euclidean':
        denom = np.sqrt(np.mean((image_true * image_true), dtype=np.float64))
    elif normalization == 'min-max':
        denom = image_true.max() - image_true.min()
    elif normalization == 'mean':
        denom = image_true.mean()
    else:
        raise ValueError("Unsupported norm_type")
    return np.sqrt(mean_squared_error(image_true, image_test)) / denom

def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.
    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.
    Returns
    -------
    psnr : float
        The PSNR metric.
    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_psnr`` to
        ``skimage.metrics.peak_signal_noise_ratio``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    check_shape_equality(image_true, image_test)

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im_true.", stacklevel=2)
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = np.min(image_true), np.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "im_true has intensity values outside the range expected for "
                "its data type.  Please manually specify the data_range")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    image_true, image_test = _as_floats(image_true, image_test)

    err = mean_squared_error(image_true, image_test)
    return 10 * np.log10((data_range ** 2) / err)