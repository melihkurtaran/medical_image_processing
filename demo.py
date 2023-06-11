#Import Libraries
import numpy as np
import scipy
from matplotlib import pyplot as plt, animation
import matplotlib
import pydicom
import os
import matplotlib
from PIL import Image
from scipy.optimize import least_squares
import math
from typing import List

def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """
    return img_dcm[:, :, img_dcm.shape[1]//2]   


def median_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """
    return img_dcm[:, img_dcm.shape[2]//2, :]


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis=2)


def AIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the average intensity projection on the sagittal orientation. """
    return np.mean(img_dcm, axis=2)


def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """
    return np.max(img_dcm, axis=1)


def AIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the average intensity projection on the coronal orientation. """
    return np.mean(img_dcm, axis=1)


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)

def find_centroid(mask: np.ndarray) -> np.ndarray:
    # Your code here:
    #   Consider using `np.where` to find the indices of the voxels in the mask
    #   ...
    idcs = np.where(mask == 1)
    centroid = np.stack([
        np.mean(idcs[0]),
        np.mean(idcs[1]),
        np.mean(idcs[2]),
    ])
    return centroid

def visualize_axial_slice(img: np.ndarray, masks: List[np.ndarray], mask_centroids: List[np.ndarray]) -> np.ndarray:
    """ Visualize the axial slice (first dimension) of multiple regions with different colors for each mask. """
    fused_slices = []
    
    for i in range(img.shape[0]):
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=np.amin(img[i]), vmax=np.amax(img[i]))
        
        fused_slice = 0.75 * cmap(norm(img[i]))[..., :3]
        
        for j, mask in enumerate(masks):
            color = plt.cm.get_cmap('tab10')(j)  # Get a different color for each mask
            mask_region = mask[i].astype(bool)
            fused_slice[mask_region] = color[:-1]  # Assign the color to the mask region
        
        fused_slices.append(fused_slice[..., 0])
    
    return np.array(fused_slices)

# Pixel dimensions in millimeters
pixel_dimensions_mm = [400, 122.5, 122.5]

# Empty lists to store pixel and segmentation data
pixel_data = []
segmentation_data = []

folder_name = "/Users/melih/Desktop/medical/manifest-1643035385102/HCC-TACE-Seg/HCC_007/12-27-1997-NA-AP LIVER PRT WWO-67834"

# Path to the segmentation DICOM file
segmentation_path = folder_name + "/300.000000-Segmentation-39839/1-1.dcm"

# Read the segmentation DICOM dataset
segmentation_dataset = pydicom.dcmread(segmentation_path)

# Extract the pixel array from the segmentation dataset
segmentation_array = segmentation_dataset.pixel_array

# Directory containing the image DICOM files
directory = folder_name + "/4.000000-Recon 2 LIVER 3 PHASE AP-28011/"

# Get a sorted list of file names in the directory
directories = sorted(os.listdir(directory))

# Iterate over the files in the directory
for filename in directories:
    if filename.endswith(".dcm"):
        # Construct the path to the DICOM file
        path = os.path.join(directory, filename)
        
        # Read the DICOM dataset
        dataset = pydicom.dcmread(path)
        # Append the pixel array to the pixel_data list
        pixel_data.append(dataset.pixel_array)

# Convert the pixel_data list to a NumPy array
img_dcm = np.array(pixel_data)

# Flip the segmentation array along the vertical axis (axis=1)
segmentation_array = np.flip(segmentation_array, axis=1)

liver = find_centroid(segmentation_array[0:75])
tumor = find_centroid(segmentation_array[75:149])
portal_vein = find_centroid(segmentation_array[150:224])
abdominal_aorta = find_centroid(segmentation_array[225:299])
mask_centroid = [liver,tumor,portal_vein,abdominal_aorta]

array_seg = [segmentation_array[0:74],segmentation_array[75:149],segmentation_array[150:224],segmentation_array[225:299]] 
segmented_img_dcm = visualize_axial_slice(img_dcm[0:74], array_seg, mask_centroid)

# Get the minimum and maximum pixel values from the segmented DICOM image
segmented_img_min = np.amin(segmented_img_dcm)
segmented_img_max = np.amax(segmented_img_dcm)

# Use the 'viridis' colormap for visualizing the image
colormap = matplotlib.colormaps["viridis"]

# Create a figure and axes for plotting
fig, ax = plt.subplots()

# Create directories if they don't exist
os.makedirs("result/", exist_ok=True)

# Define the number of rotations
num_rotations = 18

# Initialize empty lists for storing projections
segmented_img_projections = []

pixel_len_mm  = pixel_dimensions_mm

# Rotate the image and generate projections
for idx, angle in enumerate(np.linspace(0, 360 * (num_rotations - 1) / num_rotations, num=num_rotations)):

    # Rotate the segmented image on the axial plane
    rotated_img = rotate_on_axial_plane(segmented_img_dcm, angle)

    # Get the maximum intensity projection on the sagittal plane
    projection = MIP_sagittal_plane(rotated_img)

    # Visualize the projection
    ax.imshow(
        projection,
        cmap=colormap,
        vmin=segmented_img_min,
        vmax=segmented_img_max,
        aspect=pixel_len_mm[0] / pixel_len_mm[1],
    )

    # Save the projection as an image
    plt.savefig(f"result/Projection_{idx}.png")

    # Add the projection to the list
    segmented_img_projections.append(projection)

# Generate animation data
animation_frames = [
    [
        ax.imshow(
            projection,
            animated=True,
            cmap=colormap,
            vmin=segmented_img_min,
            vmax=segmented_img_max,
            aspect=pixel_len_mm[0] / pixel_len_mm[1],
        )
    ]
    for projection in segmented_img_projections
]

# Create an animation from the frames
animation_output = animation.ArtistAnimation(fig, animation_frames, interval=30, blit=True)

# Show the figure
plt.show()
