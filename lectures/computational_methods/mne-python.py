import mne

# Load the data
data_path = mne.datasets.sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Pick EEG channels
raw.pick_types(meg=False, eeg=True, eog=True)

# Set up and run the ICA
ica = mne.preprocessing.ICA(n_components=15, random_state=97)
ica.fit(raw)

# Find and remove the EOG artifacts
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices

# Apply the ICA to the raw data
raw_clean = raw.copy()
ica.apply(raw_clean)

# Plot the cleaned data
raw_clean.plot()
