# -*- coding: utf-8 -*-

from PlantHyperspectralSVD import specim_loading, plant_selection, data_extraction, spectral_comparison 
import pandas as pd
import numpy as np

# Absolute path to SPECIM IQ image folder
path = r'Abosolute\\path\\to\\SPECIM\\IMGE\\FOLDER'
labels = ['leaf']  # Labels for selected plants
threshold_val = 2.2  # Threshold value for masking. 
# Note: This value can be adjusted based on species and experimental conditions.
# For example, a threshold value of 1 works well for M. polymorpha under control conditions,
# but a higher value of 5 is required for conditions such as nitrate deficiency.

#%%
# Load hyperspectral image into numpy cube
cube = specim_loading(path)

# GUI for plant selection
# Double-click to select plants, press "q" to finish
location, _ = plant_selection(cube) 

# Add label column to location DataFrame
location['label'] = labels  

# Masking and Extraction of spectral data
# Adjust threshold values as needed 
spectra, img, masked = data_extraction(cube, location, threshold=threshold_val, path=path) 

# Concatenate spectral data for different areas
spectra_per_area = pd.DataFrame(np.concatenate([
    spectra.iloc[:, 2:206].to_numpy(), 
    spectra.iloc[:, 206:410].to_numpy(), 
    spectra.iloc[:, 410:614].to_numpy(), 
    spectra.iloc[:, 614:820].to_numpy()
], axis=0))

# Create new index labels
name = [f"{label}_{area}" for area in ['whole', 'peripheral', 'paracentral', 'central'] for label in location['label']]

# Set the new index
spectra_per_area.index = name

# Call the spectral_comparison function
spectral_comparison(spectra_per_area, path)
