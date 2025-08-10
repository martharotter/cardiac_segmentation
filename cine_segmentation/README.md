# CINE Cardiac Segmentation Project

This project implements 2D+Time UNet-based segmentation for cardiac CINE MRI sequences, using temporal context to improve segmentation accuracy.

## Project Overview

The goal is to segment cardiac structures (left ventricle, right ventricle) from CINE MRI sequences by leveraging temporal information from previous frames. This approach uses a 5-frame temporal context window to provide the model with information about cardiac motion and structure across time.

## Files Overview

### **Core Training & Model Files**

#### `cine_2d_time_segmentation.py` (16KB, 424 lines)
**Main training script** that creates the `best_2d_time_cine_segmentation_model.pth`
- **`Cine2DTimeDataset`** class: Handles temporal sequence data loading with 5-frame context
- **`Cine2DTimeUNet`** class: Neural network architecture (6 input channels: 5 context + 1 target)
- **`train_2d_time_cine_model()`** function: Training loop with validation and model saving
- **`main()`** function: Orchestrates the entire training process
- **Data augmentation**: Random rotations, flips, and intensity variations
- **Loss function**: Combined Dice + Cross-entropy loss for segmentation

#### `best_2d_time_cine_segmentation_model.pth` (119MB)
**Trained model weights** - The final trained UNet model that achieved the best performance on validation data during training.

### **Evaluation & Analysis Files**

#### `evaluate_2d_time_cine_holdout.py` (8.9KB, 233 lines)
**Holdout set evaluation script** for testing the trained model on unseen data
- Loads the trained model and evaluates on a separate test set
- Computes Dice scores, IoU, and other segmentation metrics
- Generates detailed performance reports and visualizations

#### `visualize_2d_time_cine_results.py` (9.2KB, 241 lines)
**Results visualization script** for analyzing model performance
- Creates plots showing training/validation loss progression
- Visualizes Dice score improvements over epochs
- Generates comparison plots between different models
- Helps identify overfitting and training convergence

### **Data Preprocessing Files**

#### `preprocessing.py` (10KB, 186 lines)
**Data preprocessing pipeline** for preparing CINE MRI data
- **`CinePreprocessor`** class: Handles both CINE and non-CINE cardiac MRI data
- **`preprocessing()`** method: Processes general SA (Short Axis) images and ground truth
- **`preprocessing_cine()`** method: Specialized processing for CINE temporal sequences
- **Data format conversion**: Converts NIfTI files to individual slices
- **Label generation**: Uses ED (End Diastole) ground truth as labels for CINE frames

#### `run_preprocessing.py` (644B, 26 lines)
**Simple execution script** to run the preprocessing pipeline
- Imports and instantiates the `CinePreprocessor`
- Calls preprocessing methods to prepare the dataset
- Quick way to execute the full preprocessing workflow

#### `images/` (directory)
**Preprocessed image data** - Contains the processed CINE MRI slices ready for training
- Individual NIfTI files for each slice and time frame
- Organized by patient ID and temporal sequence
- Preprocessed and normalized for optimal training

## Data Flow

1. **Raw Data**: M&Ms-2 cardiac MRI dataset (360 subjects)
2. **Preprocessing**: `preprocessing.py` converts NIfTI files to training-ready format
3. **Training**: `cine_2d_time_segmentation.py` trains the UNet model
4. **Evaluation**: `evaluate_2d_time_cine_holdout.py` tests on holdout data
5. **Analysis**: `visualize_2d_time_cine_results.py` analyzes performance

## Key Features

- **Temporal Context**: Uses 5 previous frames to inform current frame segmentation
- **Multi-class Segmentation**: Segments LV, RV, and myocardium
- **Data Augmentation**: Robust training with rotations, flips, and intensity changes
- **Performance Monitoring**: Comprehensive logging and visualization of training progress
- **Model Persistence**: Saves best performing model based on validation metrics

## Usage

1. **Preprocess data**: `python run_preprocessing.py`
2. **Train model**: `python cine_2d_time_segmentation.py`
3. **Evaluate model**: `python evaluate_2d_time_cine_holdout.py`
4. **Visualize results**: `python visualize_2d_time_cine_results.py`

## Dependencies

- PyTorch (for deep learning)
- Nibabel (for NIfTI file handling)
- NumPy & Matplotlib (for data processing and visualization)
- Scikit-learn (for metrics and data splitting)
- tqdm (for progress bars)

This project represents a complete pipeline from raw cardiac MRI data to trained segmentation model, with comprehensive evaluation and visualization tools.