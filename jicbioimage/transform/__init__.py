"""Module containing image transformation functions.

The :mod:`jicbioimage.transform` module contains a number of built-in general
purpose transformations that have had the
:func:`jicbioimage.core.transformation` function decorator applied to them.
"""

from functools import wraps

import numpy as np

import scipy.ndimage.filters

import skimage.io
import skimage.morphology
import skimage.exposure

from jicimagelib.io import AutoName, AutoWrite
from jicimagelib.image import Image
from jicimagelib.util.array import (
    normalise,
    reduce_stack,
    dtype_contract,
)

from jicimagelib.transform import transformation


__version__ = "0.4.2"


@transformation
def max_intensity_projection(stack):
    """Return maximum intensity projection of a stack.
    
    :param stack: 3D array from which to project third dimension 
    :returns: :class:`jicimagelib.image.Image`
    """
    return reduce_stack(stack, max)

@transformation
def min_intensity_projection(stack):
    """Return minimum intensity projection of a stack.
    
    :param stack: 3D array from which to project third dimension 
    :returns: :class:`jicimagelib.image.Image`
    """
    return reduce_stack(stack, min)

@transformation
@dtype_contract(input_dtype=np.float, output_dtype=np.float)
def smooth_gaussian(image, sigma=1):
    """Returns Gaussian smoothed image.

    :param image: numpy array or :class:`jicimagelib.image.Image`
    :param sigma: standard deviation
    :returns: :class:`jicimagelib.image.Image`
    """
    return scipy.ndimage.filters.gaussian_filter(image, sigma=sigma, mode="nearest")

@transformation
@dtype_contract(output_dtype=np.float)
def equalize_adaptive_clahe(image, ntiles=8, clip_limit=0.01):
    """Return contrast limited adaptive histogram equalized image.
    
    The return value is normalised to the range 0 to 1.

    :param image: numpy array or :class:`jicimagelib.image.Image` of dtype float
    :param ntiles: number of tile regions
    :param clip_limit: clipping limit in range 0 to 1,
                       higher values give more contrast
    """
    # Convert input for skimage.
    skimage_float_im = normalise(image)
    
    if np.all(skimage_float_im):
        raise(RuntimeError("Cannot equalise when there is no variation."))
    
    normalised = skimage.exposure.equalize_adapthist(skimage_float_im,
        ntiles_x=ntiles, ntiles_y=ntiles, clip_limit=clip_limit)

    assert np.max(normalised) == 1.0
    assert np.min(normalised) == 0.0

    return normalised

@transformation
@dtype_contract(output_dtype=np.bool)
def threshold_otsu(image, multiplier=1.0):
    """Return image thresholded using Otsu's method.
    """
    otsu_value = skimage.filters.threshold_otsu(image)
    return image > otsu_value * multiplier

@transformation
@dtype_contract(input_dtype=np.bool, output_dtype=np.bool)
def remove_small_objects(image, min_size=50):
    """Remove small objects from an boolean image.

    :param image: boolean numpy array or :class:`jicimagelib.image.Image`
    :returns: boolean :class:`jicimagelib.image.Image`
    """ 
    return skimage.morphology.remove_small_objects(image, min_size=min_size)
