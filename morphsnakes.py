from itertools import cycle

import numpy as np
from scipy import ndimage as ndi


_P2 = [np.eye(3),
       np.array([[0, 1, 0]] * 3),
       np.flipud(np.eye(3)),
       np.rot90([[0, 1, 0]] * 3)]


class _iterable(object):
    """
    call fonctions
    """
    def __init__(self, iterable):
        self.function = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.function)
        return f(*args, **kwargs)


def sup_inf(u):
    """compute erosion image"""
    if not np.ndim(u) == 2:
        raise ValueError("image has 2 dimension")

    P = _P2
    erosion_liste = []
    for i in P:
        erosion_liste.append(ndi.binary_erosion(u, i))
    return np.array(erosion_liste, dtype=np.int8).max(0)


def inf_sup(u):
    """compute dilatation image"""

    if not np.ndim(u) == 2:
        raise ValueError("image as 2 dimension")

    P = _P2
    dilatation_liste = []
    for i in P:
        dilatation_liste.append(ndi.binary_dilation(u, i))
    return np.array(dilatation_liste, dtype=np.int8).min(0)


_for = _iterable([lambda u: sup_inf(inf_sup(u)), lambda u: inf_sup(sup_inf(u))])


def _check_image(image, init_level_set):
    """Check image."""
    if not image.ndim == 2:
        raise ValueError("image must be 2 dimension.")

    if len(image.shape) != len(init_level_set.shape):
        raise ValueError("level set dim not equal image dim.")


def create_circle(img, center=None, radius=None):
    """Create a circle with binary values for level set.

    img : tuple of positive integers
        Shape of the image
    center :  Coordinates of the circle, if None center of image
    radius : Radius of the circle
    """

    if center is None:
        center = tuple(i // 2 for i in img)

    if radius is None:
        radius = min(img) * 3.0 / 8.0

    grid = np.mgrid[[slice(i) for i in img]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid)**2, 0))
    res = np.int8(phi > 0)
    return res


def _init_level_set(init_level_set, img):
    """initialize level sets with a string."""
    if isinstance(init_level_set, str):
        if init_level_set == 'circle':
            res = create_circle(img)
        else:
            raise ValueError("init_level_set must be circle")
    else:
        res = init_level_set
    return res


def inverse_gaussian_gradient(image, alpha=100.0, sigma=5):
    """Inverse of gradient magnitude.

    Compute magnitude gradient and invert it

    image : image input
    alpha : coef for invertion
    sigma : standard deviation  Gaussian filter
    """
    norm_gradient = ndi.gaussian_gradient_magnitude(image, sigma, mode='nearest')
    return 1.0 / np.sqrt(1.0 + alpha * norm_gradient)


def morphological_geodesic_active_contour(image_preproces, iterations, init_level_set='circle', smoothing=1,
                                          threshold='0', balloon=0, iter_callback=lambda x: None):

    """ Geodesic active contour with morphological operators

    image_preproces : Preprocessed image
    init_level_set : Initial level set
    threshold : threshold for borders
    iterations : Number iteration
    balloon : Balloon force
    smoothing : Number of time smoothing operator applied
    iter_callback : function
    """

    image = image_preproces
    init_level_set = _init_level_set(init_level_set, image.shape)

    _check_image(image, init_level_set)

    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    gradient_image = np.gradient(image)
    if balloon != 0:
        balloon_force = image > threshold / np.abs(balloon)

    level_set = np.int8(init_level_set > 0)
    iter_callback(level_set)

    for _ in range(iterations):
        if balloon > 0:
            tmp = ndi.binary_dilation(level_set, structure)
        elif balloon < 0:
            tmp = ndi.binary_erosion(level_set, structure)
        if balloon != 0:
            level_set[balloon_force] = tmp[balloon_force]

        tmp = np.zeros_like(image)
        gradient_level_set = np.gradient(level_set)
        for grad_img, grad_lev_set in zip(gradient_image, gradient_level_set):
            tmp += grad_img * grad_lev_set
        level_set[tmp > 0] = 1
        level_set[tmp < 0] = 0

        for _ in range(smoothing):
            level_set = _for(level_set)
        iter_callback(level_set)

    return level_set
