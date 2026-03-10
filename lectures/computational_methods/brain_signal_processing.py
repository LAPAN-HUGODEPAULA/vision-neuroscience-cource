import mne

# Load sample dataset from MNE-Python
# This downloads/accesses the sample MEG/EEG dataset included with MNE
# fif file extension stands for "Functional Imaging Format", a common file format for MEG/EEG data  
data_path = mne.datasets.sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

# Load the raw data file (.fif format) and preload all data into memory for faster processing
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Select only EEG and EOG channels for analysis
# meg=False excludes magnetometer/gradiometer channels
# eeg=True includes electroencephalography channels
# eog=True includes electrooculography channels (eye movement artifacts)
raw.pick_types(meg=False, eeg=True, eog=True)

# Initialize Independent Component Analysis (ICA) for artifact removal
# ICA decomposes the signal into independent components to isolate artifacts
# n_components=15 limits the decomposition to 15 independent components
# random_state=97 ensures reproducible results across runs

ica = mne.preprocessing.ICA(n_components=15, random_state=97)

# Train the ICA model on the raw EEG data to learn the component structure
ica.fit(raw)

# Automatically detect components that correlate with eye movement artifacts
# EOG artifacts are common in EEG and need to be removed for clean neural signals
# This function identifies which ICA components represent eye blinks/movements
eog_indices, eog_scores = ica.find_bads_eog(raw)

# Mark the identified EOG components for exclusion from the cleaned data
ica.exclude = eog_indices

# Apply ICA artifact removal to create cleaned data
# Create a copy of the original data to preserve the raw recording
raw_clean = raw.copy()

# Remove the marked artifact components from the data while preserving neural signals
ica.apply(raw_clean)

# Visualize the cleaned EEG data in an interactive plot
# This allows inspection of the data quality after artifact removal
plt = raw_clean.plot()
plt.show()

input("Press Enter to close...")
