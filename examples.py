import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
import morphsnakes as ms


PATH_IMG_TREFLE = 'images/trefle.jpg'
PATH_IMG_PIECE = 'images/piece.jpeg'
PATH_IMG_TIGER = 'images/tigerfiltre.jpg'


def visual_callback_2d(background, figure=None):
    """
    compute callback for morphological_geodesic_active_contour

    background : Image background for visualization
    figure : Figure matplotlib
    """
    
    if figure is None:
        figure = plt.figure()
    figure.clf()
    ax1 = figure.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)
    ax2 = figure.add_subplot(1, 2, 2)
    ax3 = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(level_set):
        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(level_set, [0.3], colors='r')
        ax3.set_data(level_set)
        figure.canvas.draw()
        plt.pause(0.001)
    return callback


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def example_trefle():

    img = imread(PATH_IMG_TREFLE)[..., 0] / 255.0
    gradient_inverse = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=5.48)
    init_ls = ms.create_circle(img.shape, None, 20)
    callback = visual_callback_2d(img)
    ms.morphological_geodesic_active_contour(gradient_inverse, iterations=100, init_level_set=init_ls,
                                             smoothing=1, threshold=0.32, balloon=1, iter_callback=callback)


def example_piece():

    imgage_color = imread(PATH_IMG_PIECE) / 255.0
    img = rgb2gray(imgage_color)
    gradient_inverse = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=2)
    init_ls = ms.create_circle(img.shape, None, 150)
    callback = visual_callback_2d(imgage_color)
    ms.morphological_geodesic_active_contour(gradient_inverse, iterations=200, init_level_set=init_ls,
                                             smoothing=2, threshold=0.3, balloon=-1, iter_callback=callback)


def example_tiger():

    imgage_color = imread(PATH_IMG_TIGER) / 255.0
    img = rgb2gray(imgage_color)
    gradient_inverse = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=2)
    init_ls = ms.create_circle(img.shape, None, 150)
    callback = visual_callback_2d(imgage_color)
    ms.morphological_geodesic_active_contour(gradient_inverse, iterations=150, init_level_set=init_ls,
                                             smoothing=2, threshold=0.3, balloon=-1, iter_callback=callback)


if __name__ == '__main__':
    example_trefle()
    example_piece()
    example_tiger()

    plt.show()
