import numpy as np
import matplotlib.pyplot as plt
import math

from matplotlib.colors import Normalize as Norm 
from matplotlib.colors import PowerNorm, LogNorm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# # load sif with open_sif
# from open_sif import elSif

# def load_sif(file_path):
#     data = elSif(file_path)
#     # need to transpose for my purposes
#     for i, image in enumerate(data.photons):  
#         data.photons[i] = np.flip(image, axis=0)
#     return data


def get_reg_matrix(before, after):
    A = np.vstack((np.hstack((before[0], [1])),
                np.hstack((before[1], [1])),
                np.hstack((before[2], [1]))))

    B = np.vstack((after[0], after[1], after[2]))

    X = np.linalg.solve(A, B)

    return X.T

def check_reg_matrix(matrix, before, after):
    for i in range(3):
        print(f"{i}. Actual: ({after[i][0]:.2f}, {after[i][1]:.2f}); \
              Predicted: ({np.matmul(matrix, before[i])[0]:.2f}, \
              {np.matmul(matrix, before[i])[1]:.2f})")
        
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
#express in terms of two vectors
def express(x, origin, point1, point2):
    x -= origin
    vec1 = point1 - origin
    vec2 = point2 - origin
    coeffs = np.linalg.solve(np.vstack((vec1, vec2)).T, x)
    return coeffs

def check_angle(origin, point1, point2):
    vec1 = point1 - origin
    vec2 = point2 - origin
    return np.dot(vec1, vec2)/(np.linalg(vec1)*np.linalg(vec2))

def norm(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2)

#get particle image from closest gridpoints images
def get_image(point, grid, grid_images):
    distances = [distance(point, val) for val in grid]
    sorted_ind = np.argsort(distances)
    gridpoints =  sorted_ind[:3]

    before =  [np.array(grid[gridpoint]) for gridpoint in gridpoints]
    cos_angle = (before[1]-before[0])@(before[2]-before[0])/norm(before[1]-before[0])/norm(before[2]-before[0])

    # avoid collinear points
    k = 3
    while np.abs(cos_angle) > 0.1:
        gridpoints[-1] = sorted_ind[k]
        before =  [np.array(grid[gridpoint]) for gridpoint in gridpoints]
        cos_angle = (before[1]-before[0])@(before[2]-before[0])/norm(before[1]-before[0])/norm(before[2]-before[0]) 
        k += 1

    before =  [grid[gridpoint] for gridpoint in gridpoints]
    after = [grid_images[gridpoint] for gridpoint in gridpoints]

    coeffs = express(np.array(point), np.array(before[0]), np.array(before[1]), np.array(before[2]))

    object = np.array(before[0]) + coeffs[0]*(np.array(before[1])-np.array(before[0])) + coeffs[1]*(np.array(before[2])-np.array(before[0]))
    image = np.array(after[0]) + coeffs[0]*(np.array(after[1])-np.array(after[0])) + coeffs[1]*(np.array(after[2])-np.array(after[0]))

    return object, image

def normalize(array):
    return (array - np.min(array))/(np.max(array) - np.min(array))

def get_closest(point, grid_dict):
    distances = [distance(point, val[1]) for val in grid_dict.values()]
    sorted_ind = np.argsort(distances)
    ids_sorted_by_distance = np.array(list(grid_dict.keys()))[sorted_ind]
    return ids_sorted_by_distance

def exp(x, a, b, c):
    return a*np.exp(b*x)+c



# brightness stuff

def gaussian2D_asym_rotate(xy, A, mux, muy, stdx, stdy, theta, offset):
    x,y = xy
    sint = np.sin(theta) ** 2
    cost = np.cos(theta) ** 2

    a = cost/2/stdx**2 + sint/2/stdy**2
    b = -np.sin(2 * theta) / (4 * stdx**2) + np.sin(2 * theta) / (4 * stdy**2)
    c = sint/2/stdx**2 + cost/2/stdy**2

    power = a*(x-mux)**2 + 2*b*(x-mux)*(y-muy) + c*(y-muy)**2

    return A*np.exp(-power) + offset

def get_estim_rel_bright(dict, gaussian, coeff):
    estimated_illum_brightness = [gaussian((val[0][0], val[0][1]), *coeff) for val in dict.values()]
    estimated_illum_brightness_relative = (np.array(estimated_illum_brightness)-coeff[-1])/coeff[0]
    return estimated_illum_brightness_relative

def estimate_relative_brightness(coord, coeff, gaussian=gaussian2D_asym_rotate):
    return (gaussian((coord[0], coord[1]), *coeff)-coeff[-1])/coeff[0]

def plot_beam_and_gaussian_fit(calibration, popt, date):
    xdata = [val[1][0] for val in calibration.values()]
    ydata =[val[1][1] for val in calibration.values()]
    brightnesses = [val[2][0] for val in calibration.values()]
    norm = Norm(vmin = np.min(brightnesses), vmax = np.max(brightnesses))
    norm_colors = cm.jet(norm(brightnesses))

    norm_colors_fit = cm.jet(norm(gaussian2D_asym_rotate((xdata, ydata), *popt)))

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5))

    for i, val in enumerate(list(calibration.values())):
        ax1.scatter(*val[1], color = norm_colors[i])
        ax2.scatter(*val[1], color = norm_colors_fit[i])
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax2.set_aspect('equal')
    ax2.invert_yaxis()

    ax1.set_title(f"Raw brightnesses_{date}")
    ax2.set_title("Gaussian fit")

    return fig, (ax1, ax2)

# fit PSFs

from scipy.optimize import curve_fit
from scipy.special import j0, j1
import math


def gaussian2d(xy, pps, mux, muy, stdx, stdy, offset):
    """
    generates a gaussian distribution over a meshgrid with independent x and y (ie. not rotated)

    parameters:
        xy ():
        amplitude (float): the amplitude in an unnormalized gaussian distribution
        mux (float): the x location of the mean
        muy (float): the y location of the mean
        stdx (float): the standard deviation over the x axis
        stdy (float): the standard deviation over the y axis
        offset (float): the minimum value of every position
    returns:
        distribution (): ----
    """
    x, y = xy

    amplitude = pps / (2 * np.pi * stdx * stdy)
    power = -0.5 * (((x - mux) / stdx) ** 2 + ((y - muy) / stdy) ** 2)
    distribution = offset + amplitude * np.exp(power)
    return distribution.ravel()


def gaussian2d_rotate(xy, pps, mux, muy, stdx, stdy, theta, offset):
    """
    generates a gaussain distribution over a meshgrid with dependent x and y

    parameters:
        xy ():
        amplitude (float): the amplitude in an unnormalized gaussian distribution
        mux (float): the x location of the mean
        muy (float): the y location of the mean
        stdx (float): the standard deviation over the x axis
        stdy (float): the standard deviation over the y axis
        theta (float): the covariance of the x and y axes as described by an angle (in radians). if cos(theta) = 0, then x and y are independent
        offset (float): the minimum value of every position
    returns:
        distribution (): ----
    """
    # https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
    # TODO: reformulate as rotation matrix applied to independent covariance matrix (stdx 0 ; 0 stdy)
    x, y = xy
    sint = np.sin(theta) ** 2
    cost = np.cos(theta) ** 2
    two_sq = lambda x: 2 * x**2

    amplitude = pps / (2 * np.pi * stdx * stdy)
    a = cost / two_sq(stdx) + sint / two_sq(stdy)
    b = -np.sin(2 * theta) / (2 * two_sq(stdx)) + np.sin(2 * theta) / (2 * two_sq(stdy))
    c = sint / two_sq(stdx) + cost / two_sq(stdy)

    power = -1 * (
        a * (x - mux) ** 2 + 2 * b * (x - mux) * (y - muy) + c * (y - muy) ** 2
    )
    distribution = offset + amplitude * np.exp(power)
    return distribution.ravel()


def airy1_2d_rotate_squared(xy, pps, mux, muy, a, b, theta, offset):
    x, y = xy
    xp = (x - mux) * np.cos(theta) - (y - muy) * np.sin(theta)
    yp = (x - mux) * np.sin(theta) + (y - muy) * np.cos(theta)

    norm = 4 / (3 * np.pi) * (2 * np.pi * a * b)
    amplitude = pps / norm  # TODO: beware of overflow
    u = np.sqrt(xp**2 / a**2 + yp**2 / b**2 )
    distribution = amplitude * (j1(u) / u) ** 2
    # spherical abberation
    # size = len(x)
    # x = np.linspace(-size // 2, size // 2, size)
    # y = np.linspace(-size // 2, size // 2, size)
    # X, Y = np.meshgrid(x, y)
    # r = np.sqrt(X**2 + Y**2)
    # aberration = aberration_factor * r**2
    # distribution *= np.exp(-aberration)
    distribution += offset
    return distribution.ravel()

def fit_distribution(obj, distribution=gaussian2d, p0=None, bounds=None):
    """
    fits a matrix to the provided distribution

    parameters:
        obj (numpy array): the matrix to be fit
        distribution (): the distribution to be estimated
        p0 (): the initial guess of the distribution parameters
        bounds (2-tuple of array like): the bounds of each parameter

    returns:
        (popt, pcov) (): popt are the estimated parameters and pcov is the covariance of the estimates.
    """
    obj = obj.astype(np.int64)
    if bounds == None:
        # restrict params to by strictly positive
        bounds_low = [0] * len(p0)
        bounds_high = [np.inf] * len(p0)
        bounds = [bounds_low, bounds_high]

    h, w = obj.shape
    x = np.linspace(0, w - 1, w, dtype=np.int64)
    y = np.linspace(0, h - 1, h, dtype=np.int64)
    xy = np.meshgrid(x, y)

    popt, pcov = curve_fit(distribution, xy, obj.ravel(), p0=p0, bounds=bounds, maxfev=1e4)
    return popt, pcov
