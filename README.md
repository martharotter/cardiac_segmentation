# Cardiac Segmentation

## Setup: 
1. Read reference paper / links
2. Clone respository locally
3. Install UV: https://github.com/astral-sh/uv
4. Run `uv sync` in project directory
5. Run `source .venv/bin/activate`
6. [Optional] Install 3D Slicer(https://https://www.slicer.org/)

## TO DO: 
- [x] We need to select ≥2 pathologies out of the list: Selecting Normal & Hypertrophic Cardiomyopathy gives us 135 patients
- [x] Preprocess data (stored in `data/out/` folder)
- [ ] Set up U-Net baseline
- [ ] Modify or write dataloader class
- [ ] Baseline run & store results

## Data structure
Image files are split up by patient ID and include the following (we are interested in SA, or Short-Axis views):
- 001_SA_CINE.nii.gz: cine
- 001_SA_ED.nii.gz: end-diastole image
- 001_SA_ED_gt.nii.gz: segmentation at ED
- 001_SA_ES.nii.gz: end-systole image
- 001_SA_ES_gt.nii.gz: segmentation at ES

"gt" files are the masks - they are "ground truth" segmentations

## References
* M&Ms-2 Challenge (https://www.ub.edu/mnms-2/)
* Deep Learning Segmentation of the Right Ventricle in Cardiac MRI: The M&Ms Challenge(https://ieeexplore-ieee-org.ucd.idm.oclc.org/document/10103611)
* U-Net: Convolutional Networks for Biomedical Image Segmentation(https://doi.org/10.48550/arXiv.1505.04597)
