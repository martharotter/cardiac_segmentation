import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

class CinePreprocessor:
    def __init__(self, data_path='./cardiac_segmentation/dataset/', 
                 output_path='./cardiac_segmentation/out/',
                 cine_output_path='./cardiac_segmentation/cine_classification/'):
        """
        Initialize the CINE preprocessor with data paths.
        
        Args:
            data_path (str): Path to the original dataset
            output_path (str): Path for general preprocessing output
            cine_output_path (str): Path for CINE-specific preprocessing output
        """
        self.data_path = data_path
        self.output_path = output_path
        self.cine_output_path = cine_output_path
        
    def display_images(self, limit=30):
        """
        Display a few slices from MRI files for visualization.
        
        Args:
            limit (int): Maximum number of files to display
        """
        print("Displaying MRI images...")
        mri_files = []
        
        for (dirpath, dirnames, filenames) in os.walk(self.output_path):
            for file in filenames:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    mri_files.append(os.path.join(dirpath, file))
                if len(mri_files) >= limit:
                    break
            if len(mri_files) >= limit:
                break

        # Display a few slices from each file
        for mri_path in mri_files[200:]:
            img = nib.load(mri_path)
            data = img.get_fdata()
            print(f"Displaying: {mri_path}, shape: {data.shape}")
            
            if data.ndim == 4:
                # Pick the middle time frame and middle slice
                mid_time = data.shape[3] // 2
                mid_slice = data.shape[2] // 2
                plt.imshow(data[:, :, mid_slice, mid_time], cmap='gray')
                plt.title(f"{os.path.basename(mri_path)} - slice {mid_slice}, time {mid_time}")
                plt.axis('off')
                plt.show()
            elif data.ndim == 3:
                mid_slice = data.shape[2] // 2
                plt.imshow(data[:, :, mid_slice], cmap='gray')
                plt.title(f"{os.path.basename(mri_path)} - slice {mid_slice}")
                plt.axis('off')
                plt.show()
            elif data.ndim == 2:
                plt.imshow(data, cmap='gray')
                plt.title(f"{os.path.basename(mri_path)} - 2D image")
                plt.axis('off')
                plt.show()
            else:
                print(f"Cannot display image with shape {data.shape}")

    def preprocessing(self):
        """
        General preprocessing for all SA (short axis) MRI data.
        Processes non-CINE SA images and their ground truth labels.
        """
        os.makedirs(os.path.join(self.output_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'labels'), exist_ok=True)
        
        for (dirpath, dirnames, filenames) in os.walk(self.data_path):
            for file in filenames:
                if file.split('_')[1] == 'SA':
                    dir_patient = file.split('_')[0]
                    
                    # Process non-CINE files
                    if file.split('_')[2].split('.')[0] != 'CINE':
                        if 'gt' not in file:
                            # Process image files
                            original = nib.load(os.path.join(self.data_path+dir_patient+'/', file)).get_fdata()
                            affine = nib.load(os.path.join(self.data_path+dir_patient+'/', file)).affine
                            hdr = nib.load(os.path.join(self.data_path+dir_patient+'/', file)).header

                            for i in range(original.shape[2]):
                                out = nib.Nifti1Image(original[:, :, i], affine, header=hdr)
                                nib.save(out, os.path.join(self.output_path+'images/', 
                                         file.split('.')[0]+'_'+format(i,"04")+'.nii.gz'))
                        else:
                            # Process ground truth files
                            original_gt = nib.load(os.path.join(self.data_path+dir_patient+'/', 
                                              file.split('.')[0]+'.nii.gz')).get_fdata()
                            affine_gt = nib.load(os.path.join(self.data_path+dir_patient+'/', 
                                            file.split('.')[0]+'.nii.gz')).affine
                            hdr_gt = nib.load(os.path.join(self.data_path+dir_patient+'/', 
                                         file.split('.')[0]+'.nii.gz')).header

                            for i in range(original_gt.shape[2]):
                                out_gt = nib.Nifti1Image(original_gt[:, :, i], affine_gt)
                                nib.save(out_gt, os.path.join(self.output_path+'labels/', 
                                         file.split('.')[0]+'_'+format(i,"04")+'.nii.gz'))

    def preprocessing_cine(self):
        """
        Process only CINE MRIs for classification purposes.
        Creates separate output folders for CINE images and labels.
        Uses ED ground truth as labels for CINE data.
        """
        os.makedirs(os.path.join(self.cine_output_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.cine_output_path, 'labels'), exist_ok=True)
        
        cine_count = 0
        for (dirpath, dirnames, filenames) in os.walk(self.data_path):
            for file in filenames:
                if file.split('_')[1] == 'SA':
                    dir_patient = file.split('_')[0]
                    
                    # Only process CINE files
                    if file.split('_')[2].split('.')[0] == 'CINE':
                        print(f"Processing CINE file: {file}")
                        cine_count += 1
                        
                        # Process CINE image
                        original = nib.load(os.path.join(self.data_path+dir_patient+'/', file)).get_fdata()
                        affine = nib.load(os.path.join(self.data_path+dir_patient+'/', file)).affine
                        hdr = nib.load(os.path.join(self.data_path+dir_patient+'/', file)).header

                        # Check if corresponding ED ground truth exists
                        ed_gt_file = os.path.join(self.data_path+dir_patient+'/', f"{dir_patient}_SA_ED_gt.nii.gz")
                        if os.path.exists(ed_gt_file):
                            ed_gt = nib.load(ed_gt_file).get_fdata()
                            print(f"  Using ED ground truth for labels: {ed_gt_file}")
                        else:
                            print(f"  Warning: No ED ground truth found for {dir_patient}")
                            continue

                        # For CINE, we might want to process all time frames
                        if original.ndim == 4:  # 4D data (x, y, slice, time)
                            for slice_idx in range(original.shape[2]):
                                for time_idx in range(original.shape[3]):
                                    # Save CINE image
                                    out = nib.Nifti1Image(original[:, :, slice_idx, time_idx], affine, header=hdr)
                                    img_filename = file.split('.')[0]+'_slice'+format(slice_idx,"03")+'_time'+format(time_idx,"03")+'.nii.gz'
                                    nib.save(out, os.path.join(self.cine_output_path+'images/', img_filename))
                                    
                                    # Save corresponding ED ground truth as label (same for all time frames)
                                    if slice_idx < ed_gt.shape[2]:  # Make sure slice exists in ground truth
                                        out_gt = nib.Nifti1Image(ed_gt[:, :, slice_idx], affine)
                                        label_filename = file.split('.')[0]+'_slice'+format(slice_idx,"03")+'_time'+format(time_idx,"03")+'.nii.gz'
                                        nib.save(out_gt, os.path.join(self.cine_output_path+'labels/', label_filename))
                                    
                        elif original.ndim == 3:  # 3D data (x, y, slice)
                            for slice_idx in range(original.shape[2]):
                                # Save CINE image
                                out = nib.Nifti1Image(original[:, :, slice_idx], affine, header=hdr)
                                img_filename = file.split('.')[0]+'_slice'+format(slice_idx,"03")+'.nii.gz'
                                nib.save(out, os.path.join(self.cine_output_path+'images/', img_filename))
                                
                                # Save corresponding ED ground truth as label
                                if slice_idx < ed_gt.shape[2]:  # Make sure slice exists in ground truth
                                    out_gt = nib.Nifti1Image(ed_gt[:, :, slice_idx], affine)
                                    label_filename = file.split('.')[0]+'_slice'+format(slice_idx,"03")+'.nii.gz'
                                    nib.save(out_gt, os.path.join(self.cine_output_path+'labels/', label_filename))
        
        print(f"Processed {cine_count} CINE files for classification")

    def check_output_directories(self):
        """
        Check if the output directories exist and display their contents.
        """
        print("Current working directory:", os.getcwd())
        print("CINE Images directory exists:", os.path.exists(os.path.join(self.cine_output_path, 'images/')))
        print("CINE Labels directory exists:", os.path.exists(os.path.join(self.cine_output_path, 'labels/')))
        
        if os.path.exists(os.path.join(self.cine_output_path, 'images/')):
            files = os.listdir(os.path.join(self.cine_output_path, 'images/'))
            print(f"Files in CINE images: {files[:5]} (showing first 5 of {len(files)})")
        
        if os.path.exists(os.path.join(self.cine_output_path, 'labels/')):
            files = os.listdir(os.path.join(self.cine_output_path, 'labels/'))
            print(f"Files in CINE labels: {files[:5]} (showing first 5 of {len(files)})") 