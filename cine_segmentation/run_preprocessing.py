#!/usr/bin/env python3
"""
Script to run CINE preprocessing using the CinePreprocessor class.
"""

from preprocessing import CinePreprocessor

def main():
    """
    Main function to run CINE preprocessing.
    """
    # Initialize the preprocessor
    preprocessor = CinePreprocessor()
    
    # Run CINE preprocessing
    print("Starting CINE preprocessing...")
    preprocessor.preprocessing_cine()
    
    # Check output directories
    print("\nChecking output directories...")
    preprocessor.check_output_directories()
    
    print("\nCINE preprocessing completed!")

if __name__ == "__main__":
    main() 