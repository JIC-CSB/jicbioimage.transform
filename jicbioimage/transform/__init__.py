"""Module containing image transformation functions.

The :mod:`jicbioimage.transform` module contains a number of built-in general
purpose transformations that have had the
:func:`jicbioimage.core.transformation` function decorator applied to them.
"""

import numpy as np

import scipy.ndimage.filters

import skimage.io
import skimage.morphology
import skimage.exposure
import skimage.filters

from jicbioimage.core.util.array import (
    normalise,
    reduce_stack,
    dtype_contract,
)

from jicbioimage.core.transform import transformation


__version__ = "0.5.1"


@transformation
def max_intensity_projection(stack):
    """Return maximum intensity projection of a stack.

    :param stack: 3D array from which to project third dimension
    :returns: :class:`jicbioimage.core.image.Image`
    """
    return reduce_stack(stack, max)


@transformation
def min_intensity_projection(stack):
    """Return minimum intensity projection of a stack.

    :param stack: 3D array from which to project third dimension
    :returns: :class:`jicbioimage.core.image.Image`
    """
    return reduce_stack(stack, min)


@transformation
def mean_intensity_projection(stack):
    """Return mean intensity projection of a stack.

    :param stack: 3D array from which to project third dimension
    :returns: :class:`jicbioimage.core.image.Image`
    """
    return reduce_stack(stack, np.mean)


@transformation
def median_intensity_projection(stack):
    """Return mean intensity projection of a stack.

    :param stack: 3D array from which to project third dimension
    :returns: :class:`jicbioimage.core.image.Image`
    """
    return reduce_stack(stack, np.median)


@transformation
@dtype_contract(input_dtype=np.float, output_dtype=np.float)
def smooth_gaussian(image, sigma=1):
    """Returns Gaussian smoothed image.

    :param image: numpy array or :class:`jicbioimage.core.image.Image`
    :param sigma: standard deviation
    :returns: :class:`jicbioimage.core.image.Image`
    """
    return scipy.ndimage.filters.gaussian_filter(image,
                                                 sigma=sigma,
                                                 mode="nearest")


@transformation
@dtype_contract(output_dtype=np.bool)
def threshold_otsu(image, multiplier=1.0):
    """Return image thresholded using Otsu's method.
    """
    otsu_value = skimage.filters.threshold_otsu(image)
    return image > otsu_value * multiplier


@transformation
@dtype_contract(input_dtype=np.bool, output_dtype=np.bool)
def remove_small_objects(image, min_size=50, connectivity=1):
    """Remove small objects from an boolean image.

    :param image: boolean numpy array or :class:`jicbioimage.core.image.Image`
    :returns: boolean :class:`jicbioimage.core.image.Image`
    """
    return skimage.morphology.remove_small_objects(image,
                                                   min_size=min_size,
                                                   connectivity=connectivity)


@transformation
def invert(image):
    """Return an inverted image of the same dtype.

    Assumes the full range of the input dtype is in use and
    that no negative values are present in the input image.

    :param image: :class:`jicbioimage.core.image.Image`
    :returns: inverted image of the same dtype as the input
    """
    if image.dtype == bool:
        return np.logical_not(image)
    maximum = np.iinfo(image.dtype).max
    maximum_array = np.ones(image.shape, dtype=image.dtype) * maximum
    return maximum_array - image


@transformation
@dtype_contract(input_dtype=bool, output_dtype=bool)
def dilate_binary(image, selem=None):
    """Return dilated image.

    :param image: :class:`jicbioimage.core.image.Image`
    :param selem: neighborhood expressed as 1's and 0's, default is a cross
    :returns: dilated image
    """
    return skimage.morphology.binary_dilation(image, selem)


@transformation
@dtype_contract(input_dtype=bool, output_dtype=bool)
def erode_binary(image, selem=None):
    """Return eroded image.

    :param image: :class:`jicbioimage.core.image.Image`
    :param selem: neighborhood expressed as 1's and 0's, default is a cross
    :returns: eroded image
    """
    return skimage.morphology.binary_erosion(image, selem)

@transformation
@dtype_contract(output_dtype=np.float)
def find_edges_sobel(image, mask=None):
    """Return edges detected using the Sobel method.

    :param image: :class:`jicbioimage.core.image.Image`
    :param mask: Optional mask indicating regions to ignore
    :returns: eroded image
    """
    return skimage.filters.sobel(image, mask=mask)
