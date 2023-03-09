import nibabel as nib
from models import AortaSegmentator
from models.SkeletonSegmentator import SkeletonSegmentator

if __name__ == "__main__":
    # Running over all cases with 'Ground Truth' to compare with
    for case_i in range(3, 5):
        s = f"resources/Case{case_i}_CT.nii.gz"
        case = s.split('.')[0].split('/')[1]
        aorta = f"resources/Case{case_i}_Aorta.nii.gz"
        aorta_raw = nib.as_closest_canonical(nib.load(aorta))
        aorta_gt = aorta_raw.get_fdata()
        pp = SkeletonSegmentator(file_path_scan=f"resources/Case{case_i}_CT.nii.gz",
                          file_path_l1=f"resources/Case{case_i}_L1.nii.gz", output_directory="")
        pp.skeleton_threshold_finder()
        aorta_segmentator = AortaSegmentator.AortaSegmentator(file_path_scan=f"resources/Case{case_i}_CT.nii.gz",
                          file_path_l1=f"resources/Case{case_i}_L1.nii.gz", output_directory="")
        aorta_segmentator.pipelined_aorta_segmentation(aorta_gt)
