import slicerio                   
import os
import numpy as np
from glob import glob
import pydicom
import nrrd
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
import SimpleITK as sitk
from PIL import Image
import argparse
from tqdm import tqdm


def set_args():
    parser = argparse.ArgumentParser(description="Preprocessing CT Images", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument("--anno_dir", default="/home/mdhi/dataset/SNUDH_1차/Annotation (100) 수정", type=str, dest='anno_dir') # annotation 정보들이 저장되어 있는 root 경로
    parser.add_argument("--dst_dir", default="./", type=str, dest='dst_dir')    # 파일이 저장되는 폴더 경로
    parser.add_argument("--dicom_path", default="/home/mdhi/dataset/SNUDH_1차/비식별화 (200)", type=str, dest='dicom_dir')    # dicom file 이 있는 폴더 경로
    parser.add_argument("--data_format", default="nii", type=str, dest='data_format')    # 저장하고자 하는 파일 형식

    return parser.parse_args()


def main(dicom_dir, dst_dir, data_format):
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
    # paths = os.listdir(anno_dir)
    if data_format == "nii'":
        path = os.path.join(dst_dir, "nii")
        os.makedirs(path, exist_ok=True)    # If dst_dir does not exist, Create new path

    elif data_format == "tiff":
        path = os.path.join(dst_dir, "tiff")
        os.makedirs(path, exist_ok=True)

    dcm = pydicom.read_file(dicom_dir) # DICOM Headers
    images = dcm.pixel_array

    # Get Dicom meta Data
    # (0002, 0003) Media Storage SOP Instance UID
    file_name = dcm.file_meta[0x0002, 0x0003].value
    
    # Dicom 전처리
    images = apply_modality_lut(images, dcm)  # RescaleSlope, RescaleIntercept
    images = apply_voi_lut(images, dcm)      

    if data_format=="tiff":
        tiff_path = os.path.join(path, file_name)
        os.makedirs(tiff_path, exist_ok=True)

        # Save images (TIFF format)
        for i in range(0, len(images)):
            image = images[i]
            image = Image.fromarray(image)
            image.save(f'{tiff_path}/{file_name}_{i+1}.tiff')

    elif data_format=="nii":
        # Save images (NII format)
        sitk_image_from_numpy = sitk.GetImageFromArray(images)
        sitk.WriteImage(sitk_image_from_numpy, f'{ path }/{file_name}.nii.gz')


if __name__ == '__main__':

    # Import args
    args = set_args()
    # annotation 폴더 경로
    # anno_dir = args.anno_dir
    # 전처리된 결과 저장할 폴더 경로
    dst_dir = args.dst_dir
    # Dicom Header 파일 폴더 경로
    dicom_dir = args.dicom_dir
    data_format = args.data_format

    print(f"Destination Path: { dst_dir },\n Dicom files Path: { dicom_dir },\n Data format: { data_format }")
    main(dicom_dir, dst_dir, data_format)
