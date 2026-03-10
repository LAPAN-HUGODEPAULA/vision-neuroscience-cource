"""The "ERG - Analyzing ERG Data with Python" 

### **1. Setup and Data Loading**

This script analyzes Electroretinogram (ERG) data using machine learning techniques.
ERG measures electrical activity in the retina in response to light stimuli.
The script processes PERG (Pattern ERG) data to classify healthy vs. abnormal retinal function.
"""

# Import necessary libraries for data processing, signal analysis, and machine learning
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing and array operations
from scipy.signal import butter, filtfilt  # Digital signal processing filters
import matplotlib.pyplot as plt  # Data visualization and plotting
import seaborn as sns  # Statistical data visualization built on matplotlib
from sklearn.model_selection import GridSearchCV, train_test_split  # ML model selection and data splitting
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron neural network classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Model evaluation metrics
from sklearn.preprocessing import LabelEncoder  # Encode categorical labels as integers
from imblearn.over_sampling import SMOTE  # Synthetic Minority Oversampling Technique for handling class imbalance
import requests  # HTTP library for downloading data from web
import zipfile  # Library for handling ZIP archives
import io  # Core tools for working with streams
import os  # Operating system interface for file/directory operations

# ==================== CONFIGURATION PARAMETERS ====================
# Dataset configuration
DATASET_URL = "https://physionet.org/static/published-projects/perg-ioba-dataset/perg-ioba-dataset-1.0.0.zip"
DATASET_DIR = 'perg-ioba-dataset-1.0.0'
METADATA_FILE = 'participants_info.csv'

# Class labels configuration
HEALTHY_CLASS_NAME = 'Healphy'  # Renamed from 'Normal'
DISEASED_CLASS_NAME = 'Deseased'  # Renamed from 'Not normal'
ORIGINAL_HEALTHY_LABEL = 'Normal'  # Original label in dataset
TARGET_COLUMN = 'diagnosis1'  # Column containing diagnostic information

# Signal processing parameters
SAMPLING_FREQUENCY = 1700  # Hz - ERG signal sampling rate
FILTER_LOW_CUT = 1  # Hz - Low frequency cutoff for bandpass filter
FILTER_HIGH_CUT = 200  # Hz - High frequency cutoff for bandpass filter
FILTER_ORDER = 5  # Filter order for Butterworth filter
VISUALIZATION_FILTER_HIGH_CUT = 200  # Hz - Different filter for visualization

# ERG signal parameters
ERG_CHANNEL = 'RE_1'  # Right eye channel 1 for analysis
TIME_WINDOW_VISUALIZATION = 200  # ms - Time window for signal visualization

# Machine learning parameters
TEST_SIZE = 0.3  # Proportion of data for testing (30%)
RANDOM_STATE = 0  # Random seed for reproducibility
MAX_ITERATIONS = 2000  # Maximum iterations for MLP training
CV_FOLDS = 5  # Number of cross-validation folds (default for GridSearchCV)

# Model hyperparameters for grid search
SOLVER_OPTIONS = ('adam', 'lbfgs')  # Optimization algorithms
HIDDEN_LAYER_SIZES = [40, 50, 60, 70]  # Number of neurons in hidden layer
LEARNING_RATES = [0.07, 0.05, 0.03]  # Initial learning rates

# Visualization parameters
FIGURE_SIZE_PIE = (6, 4)  # Size for pie chart
FIGURE_SIZE_SIGNAL = (12, 6)  # Size for signal plots
FONT_SIZE_PIE = 8  # Font size for pie chart labels

# Subject IDs for example plotting
EXAMPLE_HEALTHY_SUBJECT = '0001'  # Healthy control subject for demonstration
EXAMPLE_DISEASED_SUBJECT = '0059'  # Diseased subject for demonstration

# ==================== END CONFIGURATION ====================

# Initialize label encoder
label_encoder = LabelEncoder()

# --- Download and Unzip the Dataset ---
def download_and_unzip_physionet_data(url, extract_to='.'):
    """
    Downloads and unzips ERG dataset from PhysioNet repository.
    
    PhysioNet is a repository of freely-available medical research data.
    This function handles automatic dataset retrieval and extraction.
    
    Args:
        url (str): URL to the dataset ZIP file on PhysioNet
        extract_to (str): Local directory path where data should be extracted
    
    Returns:
        str: Path to the extracted dataset directory
    """
    print("Downloading dataset...")
    # Send HTTP GET request to download the ZIP file
    response = requests.get(url)
    # Create a ZipFile object from the downloaded content in memory
    z = zipfile.ZipFile(io.BytesIO(response.content))

    # Create target directory structure if it doesn't exist
    dir_name = os.path.join(extract_to, DATASET_DIR)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f"Extracting files to {dir_name}...")
    # Extract all files from the ZIP archive to the target directory
    z.extractall(dir_name)
    print("Download and extraction complete.")
    return dir_name

# data_dir = download_and_unzip_physionet_data(DATASET_URL)  # Uncomment to download
data_dir = DATASET_DIR  # Assuming the dataset is already downloaded for this demo

# --- Load the Metadata ---
# Load participant information containing subject IDs, demographics, and diagnoses
metadata_path = os.path.join(data_dir, 'csv', METADATA_FILE)
metadata = pd.read_csv(metadata_path)

# Rename diagnostic classes
def rename_diagnostic_classes(df, target_col=TARGET_COLUMN):
    """
    Rename diagnostic classes from original dataset labels to our standardized labels.
    
    Args:
        df (DataFrame): Metadata dataframe containing diagnostic information
        target_col (str): Column name containing diagnostic labels
    
    Returns:
        DataFrame: DataFrame with renamed diagnostic classes
    """
    df_renamed = df.copy()
     
    # Apply mapping, with all non-normal classes becoming DISEASED_CLASS_NAME
    df_renamed[target_col] = df_renamed[target_col].apply(
        lambda x: HEALTHY_CLASS_NAME if x == ORIGINAL_HEALTHY_LABEL else DISEASED_CLASS_NAME
    )
    
    return df_renamed

# Apply class renaming
metadata = rename_diagnostic_classes(metadata)

# Display basic information about the dataset
print("Metadata Information:")
print(metadata.head())
print(f"\nValue Counts for {TARGET_COLUMN}:")
# Count the number of subjects in each diagnostic category
diagnosis_counts = metadata[TARGET_COLUMN].value_counts()
print(diagnosis_counts)

# Create a pie chart to visualize the distribution of diagnoses
plt.figure(figsize=FIGURE_SIZE_PIE)
plt.pie(diagnosis_counts.values, labels=diagnosis_counts.index, autopct='%1.0f%%',  # type: ignore
        textprops={'fontsize': FONT_SIZE_PIE})  # type: ignore
plt.title('Diagnosis Distribution')
plt.show()

"""### **2. Data Preprocessing and Feature Extraction**

Extracting meaningful features from the raw ERG signals.
Two key components of the PERG waveform: the **P50** (a positive peak around 50ms) and the **N95** (a negative trough around 95ms). 
From these, we'll calculate amplitudes and latencies.

A **bandpass filter** will remove noise and baseline drift from the signals.
"""

# --- Signal Processing and Feature Extraction Functions ---

def bandpass_filter(data, lowcut=FILTER_LOW_CUT, highcut=FILTER_HIGH_CUT, 
                   fs=SAMPLING_FREQUENCY, order=FILTER_ORDER):
    """
    Applies a Butterworth bandpass filter to remove noise from ERG signals.
    
    Bandpass filtering removes frequencies outside the range of interest,
    eliminating both low-frequency baseline drift and high-frequency noise.
    
    Args:
        data (array): Raw signal data to be filtered
        lowcut (float): Lower cutoff frequency in Hz
        highcut (float): Upper cutoff frequency in Hz  
        fs (float): Sampling frequency in Hz
        order (int): Filter order (higher = steeper rolloff)
    
    Returns:
        array: Filtered signal data
    """
    # Calculate Nyquist frequency
    nyq = 0.5 * fs
    # Normalize cutoff frequencies by Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    # Design Butterworth bandpass filter coefficients
    b, a = butter(order, [low, high], btype='band') # type: ignore
    # Apply zero-phase filtering (forward and backward pass to avoid phase distortion)
    y = filtfilt(b, a, data)
    return y

def extract_features_from_signal(signal, fs=SAMPLING_FREQUENCY):
    """
    Extracts frequency-domain features from PERG signals using FFT analysis.
    
    Instead of traditional time-domain features (P50, N95 peaks), this function
    uses frequency-domain analysis via Fast Fourier Transform (FFT) to capture
    the spectral characteristics of the ERG signal.
    
    Args:
        signal (array): Raw ERG signal data
        fs (float): Sampling frequency in Hz
    
    Returns:
        tuple: (fft_freq, fft_power) - frequency bins and corresponding power values
    """
 
    # Apply bandpass filter to remove noise and baseline drift
    # Frequency range captures most relevant ERG signal components
    filtered_signal = bandpass_filter(signal, FILTER_LOW_CUT, FILTER_HIGH_CUT, fs)

    # Compute Fast Fourier Transform to convert time-domain signal to frequency-domain
    fft_vals = np.fft.rfft(filtered_signal)  # Real FFT (signal is real-valued)
    fft_freq = np.fft.rfftfreq(len(filtered_signal), 1/fs)  # Frequency bins in Hz
    # Calculate power spectrum in decibels (dB)
    fft_power = 20*np.log10(np.abs(fft_vals))  # Power spectrum in dB

    return fft_freq, fft_power

# --- Process all files and build the feature dataset ---
# Initialize list to store extracted features for all subjects
features_list = []

print("\nExtracting features from each ERG signal...")
# Iterate through each subject in the metadata
for index, row in metadata.iterrows():
    # Construct file path for subject's ERG data (zero-padded 4-digit ID)
    file_path = os.path.join('./', data_dir, 'csv', f"{row['id_record']:0>4}.csv")

    # Check if the data file exists for this subject
    if os.path.exists(file_path):
        # Load the ERG signal data from CSV file
        signal_df = pd.read_csv(file_path)
        # Extract specified ERG channel data for analysis
        signal = signal_df[ERG_CHANNEL].values

        # Extract frequency-domain features from the signal
        fft_freq, fft_power = extract_features_from_signal(signal)
        
        # Create dictionary to store subject metadata and features
        data_dict = {
            'ID': row['id_record'],  # Subject identifier
            'GROUP': row[TARGET_COLUMN],  # Diagnostic group (target variable)
        }
        
        # Add FFT power values as features, with frequency as column names
        # This creates a feature vector where each frequency bin becomes a feature
        for freq, power in zip(fft_freq, fft_power):
            data_dict.update({f'FFT_{freq:.1f}': power})
        
        # Add this subject's feature vector to the collection
        features_list.append(data_dict)

# Convert list of feature dictionaries to a structured DataFrame
features_df = pd.DataFrame(features_list)

print("Feature extraction complete.")
print("Feature DataFrame:")
print(features_df.head())

"""### **3. Model Training and Evaluation**

Now that we have our features, we can train a classifier. We'll use **Multi-layer Perceptron (MLP)**, 
a neural network model for binary classification. The process involves:

1. Splitting the data into training and testing sets.
2. Handling class imbalance using SMOTE oversampling.
3. Training the model with hyperparameter optimization.
4. Evaluating its performance on the unseen test data.
"""

# --- Prepare data for Machine Learning ---
# Separate features (X) from target labels (y)
# Drop ID and GROUP columns to get only the numerical features
X = features_df.drop(columns=['ID', 'GROUP']).values
print("\nFeature matrix shape:")
print(X.shape)

# Encode diagnostic labels using LabelEncoder for better performance
# LabelEncoder converts categorical labels to integers efficiently
y_categorical = features_df['GROUP']
y = label_encoder.fit_transform(y_categorical)

# Display class mapping for clarity
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))  # type: ignore
print(f"\nClass encoding mapping: {class_mapping}")
print(f"Class distribution:")
unique, counts = np.unique(y, return_counts=True)
for class_idx, count in zip(unique, counts):
    class_name = label_encoder.inverse_transform([class_idx])[0]
    print(f"  {class_name}: {count}")

# Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
# SMOTE generates synthetic examples of the minority class to balance the dataset
oversample = SMOTE(random_state=RANDOM_STATE)
X, y = oversample.fit_resample(X, y) # type: ignore
print(f"\nAfter oversampling, feature matrix shape: {X.shape}")
print("Class distribution after oversampling:")
unique, counts = np.unique(y, return_counts=True)
for class_idx, count in zip(unique, counts):
    class_name = label_encoder.inverse_transform([class_idx])[0]
    print(f"  {class_name}: {count}")

# Split data into training and testing sets
# stratify=y ensures both sets maintain the same class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# --- Train the MLP Classifier Model ---
print("\nTraining the MLP Classifier model...")
# Initialize Multi-layer Perceptron classifier with specified parameters
mlp_classifier = MLPClassifier(max_iter=MAX_ITERATIONS, random_state=RANDOM_STATE)

# Define hyperparameter grid for optimization
# GridSearchCV will test all combinations to find the best model
hyperparameters = {
    'solver': SOLVER_OPTIONS,  # Optimization algorithms
    'hidden_layer_sizes': HIDDEN_LAYER_SIZES,  # Number of neurons in hidden layer
    'learning_rate_init': LEARNING_RATES  # Initial learning rates
}

# Create GridSearchCV object to find optimal hyperparameters
# Uses cross-validation to evaluate each parameter combination
model = GridSearchCV(mlp_classifier, param_grid=hyperparameters, cv=CV_FOLDS, verbose=2)

# Train the model on the training data
# GridSearchCV automatically finds the best hyperparameters
model.fit(X_train, y_train)
print("Model training complete.")
print(f"Best parameters found: {model.best_params_}")

# --- Evaluate the Model ---
print("\nEvaluating model performance...")
# Make predictions on the test set (unseen data)
y_pred = model.predict(X_test)

# Calculate overall accuracy (percentage of correct predictions)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Convert encoded labels back to original class names for reporting
y_test_names = label_encoder.inverse_transform(y_test)
y_pred_names = label_encoder.inverse_transform(y_pred)

# Generate detailed classification report
# Shows precision, recall, and F1-score for each class
print("\nClassification Report:")
target_names = [HEALTHY_CLASS_NAME, DISEASED_CLASS_NAME]
print(classification_report(y_test_names, y_pred_names, target_names=target_names))

# Create and visualize confusion matrix
# Shows true vs predicted classifications in a 2x2 grid
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_names, y_pred_names, labels=target_names)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=target_names, 
           yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- Visualize a Sample Signal ---
def plot_sample_signal(subject_id):
    """
    Plots both time-domain and frequency-domain representations of an ERG signal.
    
    This visualization helps understand:
    1. Raw vs filtered signal quality
    2. Frequency content of the ERG signal
    3. Differences between healthy and pathological signals
    
    Args:
        subject_id (str): Subject ID (e.g., '0001') to plot
    """
    # Load the ERG signal data for the specified subject
    file_path = os.path.join('./', data_dir, 'csv', f"{subject_id}.csv")
    signal = pd.read_csv(file_path)[ERG_CHANNEL].values

    # Apply bandpass filter with different cutoff for visualization
    filtered_signal = bandpass_filter(signal, FILTER_LOW_CUT, 
                                    VISUALIZATION_FILTER_HIGH_CUT, SAMPLING_FREQUENCY)
    # Create time vector in milliseconds
    time = np.arange(len(signal)) / SAMPLING_FREQUENCY * 1000

    # Extract frequency-domain features for spectral analysis
    fft_freq, fft_power = extract_features_from_signal(signal)

    # Create subplot layout: signal on top, spectrum on bottom
    plt.figure(figsize=FIGURE_SIZE_SIGNAL)
    
    # Plot 1: Time-domain signal (raw and filtered)
    plt.subplot(2, 1, 1)
    plt.plot(time, signal, label='Raw Signal', color='gray', alpha=0.6) # type: ignore
    plt.plot(time, filtered_signal, label='Filtered Signal', color='blue')
    # Get diagnosis for this subject from metadata
    diagnosis = metadata[metadata["id_record"] == int(subject_id)][TARGET_COLUMN].iloc[0]
    plt.title(f'PERG Signal for Subject {subject_id} ({diagnosis})')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, TIME_WINDOW_VISUALIZATION)  # Focus on relevant time window
    
    # Plot 2: Frequency-domain spectrum
    plt.subplot(2, 1, 2)
    plt.plot(fft_freq, fft_power, label='FFT Power', color='orange')
    plt.title(f'FFT Power Spectrum for Subject {subject_id}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (µV²/Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot example signals from different diagnostic groups
print(f"\nPlotting example signals...")
plot_sample_signal(EXAMPLE_HEALTHY_SUBJECT)  # Healthy control subject
plot_sample_signal(EXAMPLE_DISEASED_SUBJECT)  # Diseased subject

# Pause execution to allow user to examine results
input("Press Enter to finish the demo...")


