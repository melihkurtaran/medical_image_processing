# Medical Image Processing

This repository focuses on DICOM image visualization and coregistration for medical image processing.

## Files

- `demo.py`: This file contains a demo code for presentation purposes. It demonstrates the creation of an animation using DICOM files.

- `Objective_2` folder: This folder contains the data related to the second objective of the project.

- `result` folder: Inside this folder, you can find 64 projections and one animation GIF file that resulted from the project.

- `manifest` folder: This folder contains the data related to the first objective of the project.

- `medical_imaging_project.ipynb`: This Jupyter Notebook file contains the main code for the entire project.

## Objectives

### 1) DICOM Loading and Visualization

- Load the HCC_007 segmentation and CT images using PyDicom. Analyzing the headers of both images to extract relevant information. Reslicing the segmentation image based on the DICOM headers. Creating a segmentation image with the the tumor region.
- Generating Maximum Intensity Projections, Applying alpha fusion to enhance image and region visibility, Creating an animation showcasing at 64 projections.

### 2) 3D Rigid Coregistration

- ImplementING a rigid motion model for image coregistration using landmarks, evaluating the quality of the coregistration based on MAE and MSE and mutual information.
- Visualizing the thalamus mask in the input space on various indexes.

## How to Use

1. Clone the repository to your local machine.
2. Ensure you have the necessary dependencies installed (provide a list if applicable).
3. Run `demo.py` to see the demo code in action.
4. Explore the `Objective_2` and `manifest` folders for additional data related to the project.
5. Review the `medical_imaging_project.ipynb` notebook for the main code implementation.
