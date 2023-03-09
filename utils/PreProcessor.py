import cv2
import numpy as np
from numpy import uint16
from skimage.measure import label
from skimage.morphology import closing, opening
from matplotlib import pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from metrics.scores import dice_coeff

"""
#########################################################
################### PROJECT CONSTANTS ###################
#########################################################
"""

MAX_VOXEL_VALUE = 2 ** 16 - 1
MIN_VOXEL_VALUE = 0
CONNECTED_COMPONENTS = 1
INTENSITY_HISTOGRAM = 2
RAW_DENSITY_HISTOGRAM = 3
WITH_BUFFER_MARGIN = 1.5
PREDICTED_DENSITY_HISTOGRAM = 4
RADIUS_PADDING = 2
WHITE_COLOR = (255, 255, 255)
FILL_SHAPE = -1
HIGHEST_DICE = -1
INITIAL_RADIUS = 17
ROWS_PADDING = 20
COLUMNS_PADDING = 20
GET_X_CENTER, GET_Y_CENTER, GET_CIRCLE_RADIUS = 0, 1, 2


def process_img(img_data: np.ndarray):
    """
    Simply processing the image, by applying the histogram equalization and bilateral filtering to reduce
    the noise in each slice, making it easier to generalize the thresholding application stage, which is done right
    after this stage.
    :param img_data:
    :return:
    """
    img_data[img_data > 250] = 0
    img_data[img_data < 0] = 0
    img_data = img_data.astype(np.uint8)
    for i in range(img_data.shape[2]):
        ct_slice = img_data[:, :, i]
        ct_slice = cv2.equalizeHist(ct_slice)
        bilateral = cv2.bilateralFilter(ct_slice, d=9, sigmaColor=95, sigmaSpace=50)
        img_data[:, :, i] = bilateral.astype(np.uint8)
    return img_data


def _tasks_scheduler(num_cores: int, resolution: int, start=150, stop=514):
    """
    Given a number of processes, we use this dispatcher in order to divide the tasks uniformly across the
    workers.
    :param num_cores: number of cpus in the working computer
    :return: Ranges of H.U values to be  calculated by each process
    """
    jobs = (stop - start) // resolution
    d = jobs // num_cores
    ranges = [
        np.arange(start=start + r * d * resolution, stop=start + r * d * resolution + d * resolution,
                  step=resolution) for r in range(num_cores - 1)]

    d += jobs % num_cores
    start = ranges[-1][-1] + resolution
    last_job = np.arange(start=start, stop=start + d * resolution,
                         step=resolution)
    ranges.append(last_job)
    return ranges


def apply_threshold(minimal_intensity: int, maximal_intensity: int, img_data: np.ndarray):
    """
    This function is given as inputs a grayscale NIFTI file (.nii.gz) and two integers – the minimal and maximal
     thresholds. The function generates a segmentation NIFTI file of the same dimensions, with a binary segmentation
    – 1 for voxels between Imin and Imax, 0 otherwise. This segmentation NIFTI file should be saved under the name
     <nifty_file>_seg_<Imin>_<Imax>.nii.gz.
    The function returns 1 if successful, 0 otherwise. Preferably, raise descriptive errors when returning 0.
    :param img_data:
    :param minimal_intensity:
    :param maximal_intensity:
    :return:
    """

    img = np.copy(img_data.astype(dtype=np.uint16))
    img[(img <= maximal_intensity) & (img > minimal_intensity)] = MAX_VOXEL_VALUE
    img[img < MAX_VOXEL_VALUE] = MIN_VOXEL_VALUE
    closed_img = closing(img)
    return closed_img


def find_circles(patched_slice, prev_circle=(0, 0, 0)):
    """
    This function finds the best circles in a given patched CT-scan slice. it does so by finding circles on
    different levels of the 2-level image-pyramid. by using the pyramid, we reduce the 'noisy' circles when we
    eliminate those who has no significant pairing between the different layers.
    :param patched_slice: an approximated ROI of the aorta in each CT-scan slice.
    :param prev_circle: a previously detected circle in a previous slice (if we're not in the first slice)
    :return: best_circle: the best circle in terms of pyramid matched pair, and also similarity circles to the
    prev_circle.
    """
    pyramid = [np.copy(patched_slice), cv2.pyrDown(np.copy(patched_slice))]
    # Define the parameters for the HoughCircles function
    circles_all_floors = []
    best_circle = None
    for factor, level in enumerate(pyramid):
        # Will enter this if-statement only at the first slice, when there are no previous circles available
        if not prev_circle[2]:
            hough_radii = np.arange(10, 18)
        else:
            hough_radii = np.arange(max(7, prev_circle[GET_CIRCLE_RADIUS] - 2),
                                    min(prev_circle[GET_CIRCLE_RADIUS] + 2, 18))
        # Apply HoughCircles to detect circles
        hough_res = hough_circle(level, hough_radii)
        # Select the most prominent 3 circles
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=3)
        cur = np.array([cx, cy, radii]).T
        circles_all_floors.append(cur)
        if factor > 0:
            lower_lev = circles_all_floors[factor - 1]
            best_circles = find_top_matching_circles(cur, lower_lev,
                                                     mask_shape=pyramid[factor - 1].shape)
            if not prev_circle[GET_CIRCLE_RADIUS]:
                return best_circles[0]

            best_circle = best_circ(best_circles, slice_ROI=patched_slice, prev_circle=prev_circle)
    return best_circle


def best_circ(cur_circles, slice_ROI, prev_circle):
    """
    given all current circles returned by hough transform, this function evaluates the dice coefficient for each
    circle with the previous circle selected as the best circle w.r.t the previous slice
    :param cur_circles: an array of circles which are most likely to be the best circle in terms of pyramid
    matched pair, and also similarity circles to the prev_circle.
    :param slice_ROI: a patch of the Region Of Interest (ROI), containing the aorta at the current slice
    :param prev_circle: a previously detected circle in a previous slice (if we're not in the first slice)
    :return: (X: float, Y: float, Rad: float) the best circle in terms of pyramid matched pair, and also similarity
    circles to the prev_circle.
    """
    dices, dist_scores, var_values = [], [], []
    prev_x, prev_y, prev_rad = prev_circle
    prev_mask = np.zeros(slice_ROI.shape).astype(uint16)
    cv2.circle(prev_mask, (int(prev_x), int(prev_y)), int(prev_rad), WHITE_COLOR, FILL_SHAPE)
    for cir in cur_circles:
        x, y, rad = cir[GET_X_CENTER], cir[GET_Y_CENTER], cir[GET_CIRCLE_RADIUS]
        cur_mask = np.zeros(slice_ROI.shape).astype(uint16)
        cv2.circle(cur_mask, (int(x), int(y)), int(rad), WHITE_COLOR, FILL_SHAPE)
        dices.append(dice_coeff(prev_mask, cur_mask))
    dist_score_ord = np.argsort(dices)
    return cur_circles[dist_score_ord[HIGHEST_DICE]]


def find_top_matching_circles(cur, lower_lev, mask_shape: tuple[int, int]):
    """
    Given 2 lists of circles from 2-consecutive slices, this function will find the 3-best circles from the
    previous slice that has the best fitting circle in the current slice, w.r.t dice coefficient.
    :param cur: list of circles from current slice
    :param lower_lev: list of circles from previous slice
    :param mask_shape: the shape of the mask, usually it is the shape of the slice
    :return: a list of length 3 (at most), where each item is a lst (prev_idx, cur_idx, dice) where
    prev_idx, cur_idx, are the indices of the best matching circles between the two consecutive slice
    lists
    """
    all_dice_scores = []
    for idx, circ in enumerate(lower_lev):
        x_prev, y_prev, r_prev = circ
        score = [idx, 0, 0]
        prev_circle_mask = np.zeros(mask_shape)
        cv2.circle(prev_circle_mask, (int(x_prev), int(y_prev)), int(r_prev), WHITE_COLOR, FILL_SHAPE)
        for jdx, cur_circ in enumerate(cur):
            x_cur, y_cur, r_cur = cur_circ
            cur_circle_mask = np.zeros(mask_shape)
            cv2.circle(cur_circle_mask, (int(x_cur * 2), int(y_cur * 2)), int(r_cur * 2), WHITE_COLOR, FILL_SHAPE)
            cur_score = dice_coeff(prev_circle_mask, cur_circle_mask)
            if cur_score > score[2]:
                score[1] = jdx
                score[2] = cur_score

        all_dice_scores.append(score)
    all_dice_scores.sort(key=lambda tup: tup[2])
    matched_circles = [lower_lev[i] for i, _, _ in all_dice_scores[-1:-min(3, len(all_dice_scores)):-1]]
    return matched_circles


def generate_plot_fig(img_of, title):
    plt.imshow(img_of, cmap="gray")
    plt.title(title)
    plt.savefig(title)


def get_n_largest_component(threshold_img: np.ndarray, n=1):
    """
    This function should be called after we performed a thresholding for the skeleton.
    It will utilize the result kept in self.bones, and will return the largest connected component, i.e., the
    patience skeleton.
    @:param: threshold_img: a binary nd-image
    :return: (N) Largest connected component(s) found in threshold_img
    """
    labels = label(threshold_img)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + n
    largestCC_img = threshold_img * largestCC
    largestCC_img = opening(largestCC_img)
    largestCC_img = closing(largestCC_img)
    return largestCC_img


def find_all_minima(connectivity_cmps: list):
    """
    Given an array of integers, this function will find all the minima points, and save the indices of all of them
    in the _dips array.
    @:param: connectivity_cmps: a list containing the number of connected components remained after different
    thresholding applied on a CT scan
    :return: The index of all minima points in the given input
    """
    minimas = np.array(connectivity_cmps)
    # Finds all local minima
    return np.where((minimas[1:-1] < minimas[0:-2]) * (
            minimas[1:-1] < minimas[2:]))[0]
