"""
fMRI Object Recognition Analysis Script

This script performs machine learning classification on fMRI data from the Haxby dataset
to decode visual object categories from brain activation patterns. The analysis focuses
on the ventral temporal cortex, a brain region crucial for visual object recognition.

The workflow includes:
1. Data loading and visualization
2. fMRI data preprocessing and masking
3. Machine learning classification with SVM
4. Performance evaluation and visualization

Author: [Your Name]
Date: September 2025
"""

# Standard library imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Third-party imports
import nibabel as nib
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMasker
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Dataset configuration
DATASET_NAME = "haxby"
SUBJECT_INDEX = 0  # First subject in the dataset

# Visualization parameters
MAIN_FIGURE_SIZE = (14, 5)
ANATOMICAL_FIGURE_SIZE = (4, 5.4)
ANATOMICAL_CUT_COORDS = [-14]
ANATOMICAL_DISPLAY_MODE = "z"

# Mask visualization configuration
MASK_COLORS = {
    'ventral_temporal': 'orange',
    'house_selective': 'blue', 
    'face_selective': 'limegreen'
}

MASK_LABELS = {
    'ventral_temporal': 'Ventral Temporal',
    'house_selective': 'House ROI',
    'face_selective': 'Face ROI'
}

# Contour plotting parameters
CONTOUR_LINE_WIDTH = 4.0
CONTOUR_LEVELS = [0]
ANTIALIASED = False

# Machine learning parameters
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42
SVM_KERNEL = 'linear'

# Data preprocessing parameters
STANDARDIZE_VOXELS = True  # Z-score normalization for better ML performance

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def load_and_prepare_dataset():
    """
    Load the Haxby fMRI dataset and extract relevant files.
    
    The Haxby dataset contains fMRI data from subjects viewing different
    categories of visual objects (faces, houses, cats, bottles, etc.).
    
    Returns:
        tuple: (functional_data_path, mask_paths, anatomical_path, labels_path)
    """
    print("Loading Haxby fMRI dataset...")
    haxby_data = datasets.fetch_haxby()
    
    # Extract file paths for the first subject
    functional_path = haxby_data.func[SUBJECT_INDEX]
    anatomical_path = haxby_data.anat[SUBJECT_INDEX]
    labels_path = haxby_data.session_target[SUBJECT_INDEX]
    
    # Extract mask paths
    mask_paths = {
        'ventral_temporal': haxby_data.mask_vt[SUBJECT_INDEX],
        'house_selective': haxby_data.mask_house[SUBJECT_INDEX],
        'face_selective': haxby_data.mask_face[SUBJECT_INDEX]
    }
    
    return functional_path, mask_paths, anatomical_path, labels_path


def visualize_brain_masks(mask_paths, anatomical_path):
    """
    Create visualization of brain masks overlaid on anatomical image.
    
    Args:
        mask_paths (dict): Dictionary containing paths to different brain masks
        anatomical_path (str): Path to anatomical brain image
    """
    print("Creating brain mask visualization...")
    
    # Create main mask visualization figure
    main_fig = plt.figure(figsize=MAIN_FIGURE_SIZE)
    plotting.plot_roi(
        mask_paths['ventral_temporal'], 
        bg_img=anatomical_path, 
        cmap="spring", 
        black_bg=False,  # type: ignore
        figure=main_fig
    )
    main_fig.suptitle("Ventral Temporal Cortex Mask", fontsize=14)
    main_fig.show()
    
    # Create detailed anatomical visualization with multiple masks
    anatomical_fig = plt.figure(figsize=ANATOMICAL_FIGURE_SIZE, facecolor="k")
    
    # Plot anatomical background
    display = plotting.plot_anat(
        anatomical_path, 
        display_mode=ANATOMICAL_DISPLAY_MODE, 
        cut_coords=ANATOMICAL_CUT_COORDS, 
        figure=anatomical_fig
    )
    
    # Add contours for each mask with unique colors
    legend_patches = []
    legend_labels = []
    
    for mask_type, mask_path in mask_paths.items():
        color = MASK_COLORS[mask_type]
        label = MASK_LABELS[mask_type]
        
        # Add mask contours to the display
        display.add_contours( # type: ignore
            mask_path, 
            contours=1, 
            antialiased=ANTIALIASED, 
            linewidth=CONTOUR_LINE_WIDTH, 
            levels=CONTOUR_LEVELS, 
            colors=[color]
        )
        
        # Create legend elements
        legend_patches.append(Rectangle((0, 0), 1, 1, fc=color))
        legend_labels.append(label)
    
    # Add legend to the plot
    plt.legend(legend_patches, legend_labels, loc="lower right")
    anatomical_fig.suptitle("Brain ROI Masks", color='white', fontsize=14)
    anatomical_fig.show()


def preprocess_fmri_data(functional_path, ventral_temporal_mask_path):
    """
    Load and preprocess fMRI data using brain masking and normalization.
    
    Args:
        functional_path (str): Path to 4D fMRI data file
        ventral_temporal_mask_path (str): Path to ventral temporal cortex mask
        
    Returns:
        tuple: (preprocessed_data, original_shape)
    """
    print("Preprocessing fMRI data...")
    
    # Load 4D fMRI data (x, y, z, timepoints)
    fmri_image = nib.load(functional_path) # type: ignore
    original_shape = fmri_image.shape # type: ignore
    print(f"Original fMRI data shape: {original_shape}")
    
    # Create masker for data extraction and preprocessing
    # - mask_img: Focus analysis on ventral temporal cortex
    # - standardize: Apply z-score normalization for better ML performance
    brain_masker = NiftiMasker(
        mask_img=ventral_temporal_mask_path, 
        standardize=STANDARDIZE_VOXELS
    )
    
    # Transform 4D data (x,y,z,time) to 2D matrix (timepoints × voxels)
    # Each row = one brain scan, each column = one voxel's activation
    preprocessed_data = brain_masker.fit_transform(fmri_image)
    print(f"Preprocessed data shape: {preprocessed_data.shape}")
    print(f"Number of timepoints: {preprocessed_data.shape[0]}")
    print(f"Number of voxels in mask: {preprocessed_data.shape[1]}")
    
    return preprocessed_data, original_shape


def load_experimental_labels(labels_path):
    """
    Load and preprocess experimental stimulus labels.
    
    Args:
        labels_path (str): Path to CSV file containing stimulus labels
        
    Returns:
        tuple: (encoded_labels, label_encoder, unique_stimuli)
    """
    print("Loading experimental labels...")
    
    # Load stimulus labels from CSV file
    label_data = pd.read_csv(labels_path, sep=" ")
    stimulus_labels = label_data['labels']
    
    # Get unique stimulus categories
    unique_stimuli = stimulus_labels.unique()
    print(f"Stimulus categories: {unique_stimuli}")
    print(f"Total number of trials: {len(stimulus_labels)}")
    
    # Use LabelEncoder for improved performance and consistency
    # This converts string labels to numerical format required by ML algorithms
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(stimulus_labels)
    
    print(f"Label encoding mapping:")
    for i, category in enumerate(label_encoder.classes_):
        print(f"  {category} -> {i}")
    
    return encoded_labels, label_encoder, unique_stimuli


def train_classification_model(fmri_data, encoded_labels):
    """
    Train machine learning model to classify object categories from brain data.
    
    Args:
        fmri_data (array): Preprocessed fMRI data (timepoints × voxels)
        encoded_labels (array): Encoded stimulus labels
        
    Returns:
        tuple: (trained_model, test_predictions, actual_labels, label_encoder)
    """
    print("Training classification model...")
    
    # Split data into training and testing sets
    # This prevents overfitting by evaluating on unseen data
    X_train, X_test, y_train, y_test = train_test_split(
        fmri_data, 
        encoded_labels, 
        test_size=TEST_SIZE_RATIO, 
        random_state=RANDOM_STATE,
        stratify=encoded_labels  # Ensure balanced class distribution
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Initialize Support Vector Machine classifier
    # Linear kernel works well for high-dimensional neuroimaging data
    classifier = SVC(kernel=SVM_KERNEL, random_state=RANDOM_STATE)
    
    # Train the classifier on brain activation patterns
    print("Fitting SVM classifier...")
    classifier.fit(X_train, y_train)
    
    # Generate predictions on test set
    test_predictions = classifier.predict(X_test)
    
    return classifier, test_predictions, y_test, X_test


def evaluate_model_performance(y_true, y_pred, label_encoder):
    """
    Evaluate and display model classification performance.
    
    Args:
        y_true (array): True labels for test set
        y_pred (array): Predicted labels from model
        label_encoder (LabelEncoder): Fitted label encoder for category names
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*80)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Classification Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Display detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 50)
    
    # Convert encoded labels back to original category names for readability
    category_names = label_encoder.classes_
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=category_names,
        digits=3
    )
    print(report)


def main():
    """
    Main execution function that orchestrates the entire fMRI analysis pipeline.
    """
    print("="*80)
    print("fMRI OBJECT RECOGNITION ANALYSIS")
    print("="*80)
    
    try:
        # Step 1: Load dataset
        functional_path, mask_paths, anatomical_path, labels_path = load_and_prepare_dataset()
        
        # Step 2: Visualize brain masks
        visualize_brain_masks(mask_paths, anatomical_path)
        input("Press Enter to continue to data preprocessing...")
        print("\n" + "-"*80 + "\n")
        
        # Step 3: Preprocess fMRI data
        fmri_data, original_shape = preprocess_fmri_data(
            functional_path, 
            mask_paths['ventral_temporal']
        )
        
        # Step 4: Load and encode experimental labels
        encoded_labels, label_encoder, unique_stimuli = load_experimental_labels(labels_path)
        
        # Step 5: Train classification model
        classifier, predictions, true_labels, test_data = train_classification_model(
            fmri_data, 
            encoded_labels
        )
        
        # Step 6: Evaluate model performance
        evaluate_model_performance(true_labels, predictions, label_encoder)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise
    
    finally:
        input("Press Enter to close...")


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()