# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 20:59:03 2022
@author: Daisuke
"""
# Load packages required
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from glob import glob
from PlantHyperspectralSVD import specim_loading, plant_masking

#%%
"""
i. Download SPECIM IQ sample spectra (or generate spectra using the GUI in the step 0). 
Note: The files contain ImageID, PlantID, growing_condition (label), area, and radius in the first 5 columns.
ii. Download the PlantHyperspectralSVD.py file and import it into your Python environment.
iii. Download the PlantHyperspectralSVD_Analysis.py (or copy the following code), then run it:
"""
# Read sample spectra files and concatenate them into one DataFrame
all_spectra = pd.DataFrame()
for files in glob(r'C:/Users/daisuke/Downloads/SPECIM_sample_spectra/*spectrum.csv'):
    all_spectra = pd.concat([all_spectra, pd.read_csv(files)])

# Create and print labels
label = all_spectra['label'].values
print(label)

# Keep only spectral information
all_spectra = all_spectra.iloc[:,5:]

#%%
"""
Step 1: Normalization
Normalize pixel intensity across all wavelength channels using the mean reflectance near 900 nm. 
The channels from 167 to 172 correspond to the wavelengths from 892.95 to 908.24 nm.
"""
# Normalize the spectral data by dividing by the mean of specific columns (167 to 173)
all_spectra = all_spectra.divide(all_spectra.iloc[:, 167:173].mean(axis=1).values, axis=0)

# Set a variable to remove lower-wavelength channels
n = 4

# Extract specific spectral regions
whole = all_spectra.iloc[:, n:204]
peripheral = all_spectra.iloc[:, 204+n:408]
paracentral = all_spectra.iloc[:, 408+n:612]
central = all_spectra.iloc[:, 612+n:816]

#%%
"""
Step 2: Singular value decomposition (SVD)
Perform Singular Value Decomposition (SVD) transformation and save the first four SVD components. 
"""
# Concatenate the peripheral and central spectral regions
concat = np.concatenate((peripheral, central), axis=0)
print(concat.shape)  # Print the shape of the concatenated array

# Perform Truncated SVD
svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
svd.fit(concat)
svd_values = svd.transform(concat)

# Uncomment the next line to save the SVD components to a CSV file
#np.savetxt(r'PATH\\TO\\FOLDER\\TO\\SAVE\\SVD_selected_components.csv', svd.components_, delimiter=",")
np.savetxt(r'C:/Users/daisuke/Downloads/SPECIM_sample_spectra/SVD_selected_components.csv', svd.components_, delimiter=",")

scatter_label = np.concatenate((label + "_peripheral", label + "_central"), axis=0)
# Plot the SVD results
for i in [0, 2]:
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=np.array(svd_values[:, i]), y=np.array(svd_values[:, i + 1]), hue=scatter_label)
    plt.xlabel('Dimension %i (%.1f %%)' % (i + 1, (100 * svd.explained_variance_ratio_[i])))
    plt.ylabel('Dimension %i (%.1f %%)' % (i + 2, (100 * svd.explained_variance_ratio_[i + 1])))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

#%%
"""
Step 3: SVD weight matrix
plot the SVD weight matrix that would highlights leaf color patterns.
The right singular vectors V* represent a weight matrix, revealing how leaf reflectance at individual wavelengths contributes to the identified patterns.
"""

# Define SPECIM IQ wavelength bands
specim_wavelength = [397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56,
446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80,
501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62,
548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70,
595.65, 598.60, 601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04,
643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65,
690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53,
738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67,
786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07,
835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74,
883.79, 886.84, 889.90, 892.95, 896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88,
981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58
]

# Plot the SVD components against the SPECIM IQ wavelength bands
for i in [0,1,2,3]:
    plt.plot(specim_wavelength[4:], svd.components_[i], label=f'Dimension {i}')

plt.xlabel('Wavelength [nm]')
plt.ylabel('Values')
plt.legend()
plt.show()

#%%
"""
Step 4: Application
Apply the selected SVD component(s) to hyperspectral images, and visualize them in SVD pseudo-colors
"""
# Initialize an empty array to store the hyperspectral data cubes
cubes = np.empty(shape=[512, 0, 204])

# Load and normalize hyperspectral images from the specified folder
for folder in glob(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\Representative_Hypserspectral_images\*')[:4]:
    # Load hyperspectral data cube
    cube = specim_loading(folder)
    # If necessary, conduct pixel-wise normalization using nIR channels. 
    normalized_cube = cube / np.mean(cube[:, :, 167:173], axis=2, keepdims=True)    
    # Concatenate the normalized cube to the existing cubes array
    cubes = np.concatenate((cubes, normalized_cube), axis=1)

'''
Masking background white pixels
'''
# Apply a mask to the hyperspectral data cube to remove background pixels
img, mask, _ = plant_masking(cubes, threshold=2.2, kn=1)
# Apply the mask to the data cube
masked_cube = cubes * mask[:, :, np.newaxis]

# Display the mask
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()

# Perform SVD transformation on the masked hyperspectral data
for i in [1,2,3]:
    # Calculate the SVD-transformed image for the current component
    svd_pic = np.dot(masked_cube[:, :, n:204], svd.components_[i])

    # Adjust the pixel values for better visualization
    svd_pic[np.nonzero(svd_pic)] += 10

    # Plot the SVD-transformed image
    plt.figure(figsize=(cubes.shape[1]//10, cubes.shape[0]//10))
    
    # Adjust vmax and vmin as needed
    plt.imshow(svd_pic, 
               vmax=np.percentile(svd_pic[np.nonzero(svd_pic)], 99.9), 
               vmin=np.percentile(svd_pic[np.nonzero(svd_pic)], 0.1) * 0.95, 
               cmap='nipy_spectral')  
    plt.axis('off')
    plt.colorbar()
    plt.title(f'SVD {i}', size=cubes.shape[1]//10)
    plt.show()

#%%
"""
Generate boxplots illustrating the values along the top SVD dimensions extracted from peripheral, paracentral, and central areas.. 
"""
def SVD_box_plot(df, group_by='ID', data_column=None):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set colors for the box plot
    colors = np.repeat(sns.color_palette()[1:4], repeats=4, axis=0)

    # Calculate mean values grouped by the specified column
    mean_df = df.groupby(group_by, as_index=True).mean()
    df = df.set_index(group_by)
    print(mean_df[data_column])

    # Create the box plot
    plt.figure()
    ax = sns.boxplot(x=df.index, y=data_column, data=df, color="0.8")
    ax = sns.stripplot(x=df.index, y=data_column, data=df, jitter=True, size=4, palette=colors)
    ax.set_position([0.25, 0.3, 0.65, 0.6])
    plt.xticks(rotation=90)
    fig = ax.get_figure()
    fig.set_size_inches(mean_df.count()[0] / 2 + 1, 4)
    plt.xlabel(None)

    return fig, ax

# Create box plots for each SVD component
for i in [0, 1, 2, 3]:
    df_boxplot = pd.DataFrame(np.dot(np.concatenate((peripheral, paracentral, central), axis=0), svd.components_[i]), columns=[f'SVD {i}'])
    df_boxplot['ID'] = np.concatenate((label + '_peripheral', label + '_paracentral', label + '_central'))
    fig, _ = SVD_box_plot(df_boxplot, group_by='ID', data_column=f'SVD {i}')
