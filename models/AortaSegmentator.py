from unittest import result
import cv2
import numpy as np
from skimage.morphology import dilation
import nibabel as nib
from matplotlib import pyplot as plt
from skimage.segmentation import morphological_chan_vese
from metrics.scores import dice_coeff, vod_score
from utils.PreProcessor import get_n_largest_component, find_circles, generate_plot_fig, process_img

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


class AortaSegmentator:

    def __init__(self, file_path_scan: str, file_path_l1: str, output_directory: str, resolution=14, Imax=1300):
        self.resolution = resolution
        self.filepath = file_path_scan
        self.raw_ct = nib.as_closest_canonical(nib.load(file_path_scan))
        self.output_directory = output_directory
        self.ct_data = self.raw_ct.get_fdata()
        self.raw_l1 = nib.as_closest_canonical(nib.load(file_path_l1))
        self.l1_data = self.raw_l1.get_fdata()
        self.Imax = Imax

    def pipelined_aorta_segmentation(self, gt):
        """
        When Called, this function will will find and segment the Aorta's ROI by using the provided L1 CT-scan to get
        a minimal ROI, by making the assumption that the Aorta should be near the L1 vertebrate.
        This function looks for significant circles in a "close neighborhood" around the aorta, and after such circle
        was found, we apply fine tuning using Chan-Vese to get more accurate segmentation results
        :return: Creating a file named <case_i>_Aorta_segmentation.nii.gz in the root directory of the project.
        This file contains the Aorta's segmentation in the CT scan provided by the user.
        """
        rows_center_start, rows_center_stop, column_center_start, axial_upper_bound, axial_lower_bound, col_border = \
            self.find_L1_borders(self.l1_data)

        circ = (0, 0, 0)
        rad = INITIAL_RADIUS
        # Perform image processing on the entire CT-scan before moving on to the segmentation task
        self.ct_data = process_img(self.ct_data)
        processed_patch = self.ct_data[rows_center_start:rows_center_stop, int(col_border[axial_upper_bound - 1]):
                                                                           int(col_border[axial_upper_bound - 1]
                                                                               + 2 * rad), axial_upper_bound - 1]
        dims = self.ct_data.shape[:2]
        aorta_segmentation = np.ones(self.ct_data.shape)
        aorta_segmentation[:, :, :axial_lower_bound] = 0
        aorta_segmentation[:, :, axial_upper_bound:] = 0
        seg = np.array(processed_patch).astype(np.uint8)

        for axial_idx in range(axial_upper_bound - 1, axial_lower_bound - 1, -1):
            # Recalculate the border of L1 vertebrate:
            border_col = int(max(column_center_start, col_border[axial_idx]))
            x = self.ct_data[rows_center_start:rows_center_stop,
                int(col_border[axial_idx]): int(col_border[axial_idx]) + rad * 2, axial_idx]
            title = "ROI Patch"
            generate_plot_fig(x, title)
            # Approximate the correct threshold values according to the recent segmentation ROI:
            threshold_low, threshold_high = self.threshold_finder(seg)
            roi_patch = np.copy(self.ct_data[rows_center_start:rows_center_stop,
                                int(col_border[axial_idx]): int(col_border[axial_idx]) + rad * 2, axial_idx])
            roi_patch[(roi_patch < threshold_low) | (roi_patch > threshold_high)] = 0
            roi_patch[roi_patch > 0] = 1
            roi_patch = dilation(roi_patch)

            title = "Binary mask"
            generate_plot_fig(img_of=roi_patch, title=title)

            roi_patch = get_n_largest_component(roi_patch)

            # Find best circle among all circles using HoughTransform
            res_circ = find_circles(roi_patch, circ)
            x, y, r = res_circ[GET_X_CENTER], res_circ[GET_Y_CENTER], res_circ[GET_CIRCLE_RADIUS]
            circ = res_circ
            aorta_prediction = np.zeros(roi_patch.shape).astype(np.uint8)

            cv2.circle(aorta_prediction, (x, y), r + RADIUS_PADDING, WHITE_COLOR, FILL_SHAPE)

            title = "Circular segmentation"
            generate_plot_fig(aorta_prediction, title)
            # Perform final segmentation using chan-vese algorithm, localized to the approximate circular ROI
            aorta_prediction = morphological_chan_vese(aorta_prediction * roi_patch, num_iter=50, init_level_set='disk',
                                                       smoothing=4, lambda2=4)

            # Prepare mask for performing the slicing on the
            aorta_mask = np.zeros(dims)
            aorta_mask[rows_center_start:rows_center_stop, border_col: border_col + rad * 2] = np.copy(aorta_prediction)
            aorta_prediction[aorta_prediction > 0] = 1
            patch = self.ct_data[rows_center_start:rows_center_stop, border_col: border_col + rad * 2, axial_idx - 1]
            seg = np.array(patch * aorta_prediction)
            plt.imshow(seg, cmap="gray")
            plt.title("Final segmentation")
            plt.show()

            # Performing the aorta segmentation
            aorta_segmentation[:, :, axial_idx] *= aorta_mask

            # Update the new bounding box for the next slice to segment
            rad = int(r * WITH_BUFFER_MARGIN)
            column_center_start = x + col_border[axial_idx] - r
            rows_center_start += y
            rows_center_stop = rows_center_start + rad
            column_center_start -= rad
            rows_center_start -= rad
            col_border[axial_idx - 1] = x + col_border[axial_idx] - r
            aorta_segmentation[:, :, axial_idx] *= aorta_mask
        dice_score = dice_coeff(gt[:, :, axial_lower_bound: axial_upper_bound],
                                aorta_segmentation[:, :, axial_lower_bound:
                                                         axial_upper_bound])
        vod = vod_score(gt[:, :, axial_lower_bound: axial_upper_bound],
                        aorta_segmentation[:, :, axial_lower_bound:
                                                 axial_upper_bound])
        case_i = self.filepath.split('.')[0].split('/')[1]
        print(f"case: {case_i}, Dice score is: {dice_score}")
        print(f"case: {case_i}, VOD score is: {vod}")

        final = nib.as_closest_canonical(nib.Nifti1Image(aorta_segmentation, self.raw_ct.affine))
        nib.save(final, f"{case_i}_Aorta_segmentation.nii.gz")
        return result

    @staticmethod
    def threshold_finder(roi_patch: np.ndarray):
        """
        Given the appropriate path of where we predicted (or guessed at the first iteration) the aorta location,
        we than calculate the appropriate values of the next threshold (High and low)to apply, in order to extract the
        correct values belong to the Aorta's
        :param roi_patch: Patch from the full slice, where we'll later try to find the aorta using HoughTransform.
        :return:
        """
        roi_indices = np.where((roi_patch > 50) & (roi_patch < 250))
        avg = np.mean(roi_patch[roi_indices])
        std = np.std(roi_patch[roi_indices])
        std = max(int(std), 25)
        return int(avg - std), int(avg + std)

    @staticmethod
    def find_L1_borders(l1_img: np.ndarray):
        """
        Given a 3d segmentation of L1 vertebrate, this function will calculate the boundaries of the given L1, to be
        used later for the final stage, the Aorta segmentation.
        :param l1_img:
        :return:
        """
        # Find the non-zero values of the input image along x, y, z axes
        x_nonzero_in, y_nonzero_in, z_nonzero_in = np.nonzero(l1_img)
        axial_upper_bound, axial_lower_bound = np.max(z_nonzero_in), np.min(z_nonzero_in)

        # Determine the start and stop center points of the column and row
        column_center_start = np.max(y_nonzero_in)
        rows_center_start = np.min(x_nonzero_in)
        rows_center_stop = (np.max(x_nonzero_in) + np.min(x_nonzero_in)) // 2

        # Create a copy of the input image and remove the pixels on the left side of the mid-column
        l1_cpy = np.copy(l1_img)
        mid_col = int(np.min(y_nonzero_in) + (column_center_start - np.min(y_nonzero_in)) // 1.2)
        l1_cpy[:, :mid_col, :] = 0

        # Calculate the column start line from the bottom of the axial view
        col_start_line = np.zeros((l1_img.shape[2],))
        for ax_i in range(axial_upper_bound - 1, axial_lower_bound - 1, -1):
            _, sl = np.where(l1_cpy[:, :, ax_i] > 0)
            try:
                # try to find  the highest index that bounds L1 (if there is any)
                sl = np.max(sl)
                col_start_line[ax_i] = sl
            except:
                col_start_line[ax_i] = int(col_start_line[ax_i + 1])

        holes = np.nonzero(col_start_line)
        upper_holes = np.max(holes)
        lower_holes = np.min(holes)
        col_start_line[upper_holes:axial_upper_bound] = col_start_line[upper_holes]
        col_start_line[axial_lower_bound:lower_holes] = col_start_line[lower_holes]
        return rows_center_start + ROWS_PADDING, rows_center_stop, column_center_start - COLUMNS_PADDING, \
               axial_upper_bound, axial_lower_bound, col_start_line.astype(np.uint)
