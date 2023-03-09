import cv2
import numpy as np
from skimage.measure import label
import nibabel as nib
import multiprocessing
from utils.PreProcessor import process_img, _tasks_scheduler, apply_threshold, find_all_minima, \
    get_n_largest_component

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


class SkeletonSegmentator:
    def __init__(self, file_path_scan: str, file_path_l1: str, output_directory: str, resolution=14, Imax=1300):
        self.resolution = resolution
        self.filepath = file_path_scan
        self.raw_ct = nib.as_closest_canonical(nib.load(file_path_scan))
        self.output_directory = output_directory
        self.ct_data = self.raw_ct.get_fdata()
        self.raw_l1 = nib.as_closest_canonical(nib.load(file_path_l1))
        self.l1_data = self.raw_l1.get_fdata()
        self.Imax = Imax

    def skeleton_threshold_finder(self):
        """
        This function will find the best threshold for the skeleton segmentation, by iterating over Haunsfield units
        in the range of lowerbound (start parameter to stop parameter) to Imax given in the class constructor.
        We use start=150, stop=514 and Imax=1300 as a preset for the sake of the requirements for this assignment
        :return: File containing the skeleton layer, as a result of the threshold application.
        """
        with multiprocessing.Manager() as manager:
            processed_scan = self.ct_data
            # processed_scan = process_img(self.ct_data)
            processed_scan = np.array(processed_scan, dtype=np.int16)
            final_img = nib.Nifti1Image(processed_scan, self.raw_ct.affine)
            nib.save(final_img, filename="conversion.nii.gz")
            self.ct_data = processed_scan
            # Prepare processes for task:
            num_cores = multiprocessing.cpu_count()
            # Prepare tasks distribution between all processes:
            ranges = _tasks_scheduler(num_cores, start=150, stop=514, resolution=self.resolution)
            # Save process's results in a dictionary, where keys are PIDs and values are [connected components,
            # [threshold images]] v
            results = manager.dict()
            # Create all the processes, according to the number of cores available:
            processes = [multiprocessing.Process(target=self.do_segmentation,
                                                 args=(
                                                     ranges[pid], self.Imax, results, pid, self.ct_data)) for
                         pid
                         in range(num_cores)]
            # Execution
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            cmps = []
            img_threshold_result = []
            for p in range(num_cores):
                pid_ccmps, pid_imgs = results[p]
                cmps.extend(pid_ccmps)
                img_threshold_result.extend(pid_imgs)
            # Find all local minima
            dips = find_all_minima(cmps)
            self.bones = img_threshold_result[dips[0]]
            manager.shutdown()
            final_skeleton = nib.Nifti1Image(get_n_largest_component(self.bones), self.raw_ct.affine)
            case_i = self.filepath.split('.')[0].split('/')[1]
            nib.save(final_skeleton, filename=f"{case_i}_Skeleton_result.nii.gz")
            return self.bones

    @staticmethod
    def do_segmentation(Imin_range, Imax, result_keep, pid, img_data):
        """
        This function is being called by each running process, on a different range of threshold values
        of Haunsfield units (HU). Each process will perform a segmentation by threshold between Imax, which is constant
        for the entire run, to the values in Imin_range, where each iteration we keep the thresholded image, and also
        keep the number of all connected components we got after we applied the threshold on the CT scan.
        we keep the results -> (connecteed components count, thresholded image) to a dictionary, where the key represents the
        the process id.
        :param Imin_range: range of lower bounds for the thresholding of the skeleton. (HU)
        :param Imax: Maximum value of the thresholding (HU)
        :param result_keep: keeping results for each process running this function
        :param pid: process ID
        :param img_data: the image data we want to segment the skeleton from
        :return:
        """
        img_res = []
        ccmps = []
        for i_min in Imin_range:
            img = apply_threshold(minimal_intensity=i_min, maximal_intensity=Imax, img_data=img_data)
            _, cmp = label(img, return_num=True)
            ccmps.append(cmp)
            img_res.append(img)
        process_results = ccmps, img_res
        result_keep[pid] = process_results