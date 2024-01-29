import slicerio                   
import os
import numpy as np
from glob import glob
import pydicom
import nrrd
from pydicom.pixel_data_handlers.util import apply_voi_lut
import SimpleITK as sitk
from PIL import Image
import argparse
from tqdm import tqdm


def set_args():
    parser = argparse.ArgumentParser(description="Preprocessing CT Images", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--anno_dir", default="/home/mdhi/dataset/SNUDH_1차/Annotation (100) 수정", type=str, dest='anno_dir') # annotation 정보들이 저장되어 있는 root 경로
    parser.add_argument("--dst_dir", default="./", type=str, dest='dst_dir')    # 파일이 저장되는 폴더 경로
    parser.add_argument("--dicom_dir", default="/home/mdhi/dataset/SNUDH_1차/비식별화 (200)", type=str, dest='dicom_dir')    # dicom file 이 있는 폴더 경로
    parser.add_argument("--data_format", default="nii", type=str, dest='data_format')    # 저장하고자 하는 파일 형식

    return parser.parse_args()


def main(anno_dir, dicom_dir, dst_dir, data_format):
    """
        Convert Raw CT Images to preprocessed Images
    
        Parameters: 
            anno_dir(str) : raw data directory path
            dicom_dir (str) : DICOM files directory path
            dst_dir(str) :  destination directory path
            data_format(str) : desired format to be converted (e.g: "nii" or "tiff")
        Returns:
            Preprocessed Images
    """

    # Total Paths in annotation directory
    paths = os.listdir(anno_dir)
    if data_format == "nii'":
        nii_path = os.path.join(dst_dir, "nii")
        os.makedirs(nii_path, exist_ok=True)    # If dst_dir does not exist, Create new path

    elif data_format == "tiff":
        tiff_path = os.path.join(dst_dir, "tiff")
        os.makedirs(tiff_path, exist_ok=True)

    for path in tqdm(paths):

        folder_dir = os.path.join(anno_dir, path)
        # Import nrrd images(input) and masked images(label)
        nrrd_image = list(set(glob(f'{folder_dir}/*.nrrd')) - set(glob(f'{folder_dir}/Segmentation*.nrrd')))[0]
        nrrd_mask_image = glob(f'{anno_dir}/{path}/*.seg.nrrd')[0]
        
        nrrd_images, _ = nrrd.read(nrrd_image)
        masked, header = nrrd.read(nrrd_mask_image)

        dicom_file = glob(f'{dicom_dir}/*/*/{path}/*')[0]
        dcm = pydicom.read_file(dicom_file) # DICOM Headers

        # Dicom 전처리
        nrrd_images = nrrd_images.transpose()
        nrrd_images = apply_voi_lut(nrrd_images, dcm)

        # 특정 영역 제외
        segmentation_info = slicerio.read_segmentation_info(nrrd_mask_image)
        segment_names = slicerio.segment_names(segmentation_info)
        segment_list = ','.join(segment_names).split(',')

        segment_names_to_label_values = [(segment_list[i], i) for i in range(0, len(segment_list)) if segment_list[i] != "Segment_34"]
        segmentation_info = {"segments": [info for info in segmentation_info['segments'] if info['id'] != "Segment_34"]}
        extracted_voxels, extracted_header = slicerio.extract_segments(masked, header, segmentation_info, segment_names_to_label_values)
        extracted_voxels = extracted_voxels.astype('float32')

        if data_format=="tiff":
            tiff_path2 = os.path.join(tiff_path, path)
            os.makedirs(tiff_path2, exist_ok=True)

            # Save images (TIFF format)
            for i in range(0, len(nrrd_images)):
                image = nrrd_images[i]
                mask = extracted_voxels[i]

                image = Image.fromarray(image)
                mask_image = Image.fromarray(mask)
                image.save(f'{tiff_path2}/{path}_{i+1}.tiff')
                mask_image.save(f'{tiff_path2}/{path}_{i+1}_mask.tiff')

        elif data_format=="nii":
            # Save images (TIFF format)
            sitk_image_from_numpy = sitk.GetImageFromArray(nrrd_images)
            sitk_mask_from_numpy = sitk.GetImageFromArray(extracted_voxels)

            sitk.WriteImage(sitk_image_from_numpy, f'{ nii_path }/{path}.nii.gz')
            sitk.WriteImage(sitk_mask_from_numpy, f'{ nii_path }/{path}_mask.nii.gz')


if __name__ == '__main__':

    # Import args
    args = set_args()
    # annotation 폴더 경로
    anno_dir = args.anno_dir
    # 전처리된 결과 저장할 폴더 경로
    dst_dir = args.dst_dir
    # Dicom Header 파일 폴더 경로
    dicom_dir = args.dicom_dir
    data_format = args.data_format

    print(f"Anno Path: { anno_dir }, \n Destination Path: { dst_dir },\n Dicom files Path: { dicom_dir },\n Data format: { data_format }")
    main(anno_dir, dicom_dir, dst_dir, data_format)

