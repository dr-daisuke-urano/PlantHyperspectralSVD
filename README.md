# Spatial leaf color patterns highlighted with Singular Value Decomposition (SVD)
This is the official implementation for Shalini Krishnamoorthi et al. (2024) [https://www.cell.com/cell-reports/home].

## Background
Leaf reflectance spectra are widely utilized for diagnosing plant stresses, which often manifest as distinct leaf color patterns. Figure 1 shows Marchantia polymorpha (liverwort) plants grown under various nutrient deficiencies. Nitrate deficiency (0xN) and phosphate deficiency (0xP) result in early senescence and purple pigmentation in the central area, respectively. Iron deficiency (0xFe) induces leaf chlorosis starting from the peripheral growing area, while calcium deficiency (0.05xCa) leads to irreversible necrosis of the growing edges. These color changes are location-dependent, as shown in leaf reflectance spectra separately obtained from three distinct regions: a central circle, paracentral annulus, and peripheral annulus (Figure 1A).

<p></P>

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figures/Figure1.png" alt="Alt text" width="70%">
Figure 1: (A) Illustration depicting the central, paracentral, and peripheral regions of liverworts. (B-F) Images and corresponding reflectance spectra of liverworts cultivated under various nutrient deficiencies. Dashed lines in green, red, and blue denote absorption wavelengths for chlorophyll, anthocyanin, and water, respectively.

## Project Overview
The SPECIM IQ hyperspectral camera used in this study generates 512x512 pixel images with 204 wavelength channels. We identified SVD components that best highlight leaf color changes using the following steps and applied the SVD components to visualize specific leaf color changes associated with nutrient deficiency responses.

Step 0: Reflectance Spectra: Obtain Leaf Reflectance Spectra from a central circle, paracentral annulus, and peripheral annulus.
Step 1: Normalization: Normalize reflectance spectra using the reflectance near 900 nm.
Step 2: SVD Transformation: Perform SVD transformation, then select and save the SVD component(s) that best highlight leaf color patterns.
Step 3: Pseudo-Colored Image Generation: Generate pseudo-colored images using the top SVD components.
Step 4: Application: Apply the selected SVD component(s) to hyperspectral images of other leaves. In our publication, these pseudo-colored images were used to diagnose nutrient stresses in liverwort and lettuce.

## Dependencies
To create a Conda environment with the dependencies used in Krishmoorthi S (2024), download environment.yml file and use the following command:

```bash
conda env create --name PlantHyperspectralSVD --file environment.yml
```

- python 3.12.3
- matplotlib 3.8.4
- numpy 1.26.4
- opencv-python 4.9.0.80
- pandas 2.2.1
- seaborn 0.11.2
- scikit-learn 1.5.0
- scipy 1.13.1

## Usage
### Step 0 (Prerequisites): 
Before proceeding with Singular Value Decomposition, background masking, and extraction of leaf pixels from the three distinct areas are required. In this GitHub repository, we provide a simple GUI that assists users in selecting plant leaves and extracting spectral information from hyperspectral images obtained with the SPECIM IQ camera. The following code obtains the mean reflectance spectra from pixels within the central, paracentral, and peripheral areas, as well as from the entire leaf, and saves them in CSV format. Sample hyperspectral images for the control (ID: 421), phosphate deficiency (ID: 397), nitrate deficiency (ID: 323), and iron deficiency (ID: 347) conditions are provided at [https://github.com/dr-daisuke-urano/PlantHyperspectralSVD/tree/main/SPECIM_sample_images]

```python
"""
i. Download SPECIM IQ sample images
ii. Download the PlantHyperspectralSVD.py file and import it into your Python environment.
iii. Download the PlantHyperspectralSVD_DataExtraction.py (or copy the following code), then run it:
"""
from PlantHyperspectralSVD import specim_loading, plant_selection, data_extraction, specim_plot 
import pandas as pd
import numpy as np

# Absolute path to SPECIM IQ image folder
path = r'Abosolute\\path\\to\\SPECIM\\IMAGE\\FOLDER'
labels = ['leaf']  # Labels for selected plants
threshold_val = 2.2  # Threshold value for masking. This value can be adjusted based on species and experimental conditions.
```

```python
cube = specim_loading(path) # Load hyperspectral image into numpy cube
location, _ = plant_selection(cube) # GUI for plant selection. Double-click to select plants, press "q" to finish
location['label'] = labels  # Add label column to location DataFrame
spectra, img, masked = data_extraction(cube, location, threshold=threshold_val, path=path) # Masking and Extraction of spectral data. Adjust threshold values as needed 

# Concatenate spectral data for different areas
spectra_per_area = pd.DataFrame(np.concatenate([
    spectra.iloc[:, 2:206].to_numpy(), 
    spectra.iloc[:, 206:410].to_numpy(), 
    spectra.iloc[:, 410:614].to_numpy(), 
    spectra.iloc[:, 614:818].to_numpy()
], axis=0))

spectra_per_area.index = [f"{label}_{area}" for area in ['whole', 'peripheral', 'paracentral', 'central'] for label in location['label']] # Create new index labels
specim_plot(spectra_per_area, path) # Call the specim plot function
```

### Step 1: Pixel-by-pixel normalization of leaf reflectance spectra with nIR bands.  
Plant leaves exhibit high reflectance of near-infrared (nIR) light. While reflectance at far-red wavelengths (700–780 nm) is significantly reduced under stress, reflectance at longer wavelengths remains relatively stable, regardless of plant growing conditions. In this project, we observed that bands near 900 nm exhibit the smallest Coefficients of Variation (CoV) among the wavelength range from 400 to 1000 nm. Therefore, we utilized the 890–910 nm nm bands as the reference to normalize the leaf reflectance spectra. The small CoV at ~900 nm was consistently observed in lettuce.

### Step 1 procedure: 
```python
# Read each CSV file and concatenate them into one DataFrame
all_spectra = pd.DataFrame()
for files in glob(r'Absolute\\path\\to\\SPECIM\\SPECTRA\\FOLDER\\*spectrum.csv'):
    all_spectra = pd.concat([all_spectra, pd.read_csv(files)])

# Print the label column values
print(all_spectra['label'].values)
```

```python
'''
Step 1: Normalization
Normalize pixel intensity across all wavelength channels using the mean reflectance near 900 nm. 
The channels from 167 to 172 correspond to the wavelengths from 892.95 to 908.24 nm.
'''
# Normalize the spectral data by dividing by the mean of specific columns (167 to 173)
all_spectra = pd.concat(
    (all_spectra.iloc[:, 0:5], all_spectra.iloc[:, 5:].divide(all_spectra.iloc[:, 5+167:5+173].mean(axis=1).values, axis=0)),
    axis=1
)

# Create labels for the scatter plot
label = np.concatenate((all_spectra['label'] + "_peripheral", all_spectra['label'] + "_central"), axis=0)

# Set a variable to remove lower-wavelength channels
n = 4

# Extract specific spectral regions
whole = all_spectra.iloc[:, 5 + n:5 + 204]
peripheral = all_spectra.iloc[:, 209 + n:209 + 204]
paracentral = all_spectra.iloc[:, 413 + n:413 + 204]
central = all_spectra.iloc[:, 617 + n:617 + 204]
```

### Step 2: Singular value decomposition (SVD)
SVD is a widely used method for dimensionality reduction. Most of the spectral information spanning different wavelength channels can be projected into a small number of dimensional spaces. This method is effective in extracting non-redundant spectral features which possibly highlight anomalous leaf color patterns. 

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figures/Figure2.png" alt="Alt text" width="70%">
Figure 2. Step 1: Pixel-by-pixel normalization of leaf reflectance spectra with nIR bands.  (A, B) Reflectance spectra obtained from representative plant leaves grown under control (A) and phosphate deficiency (B) conditions. Leaf reflectance at nIR bands from 890 to 910 nm are minimally affected by nutrient deficiencies but rather affected by light conditions. Blue graphs show the original reflectance spectra from leaf pixels. Orange graphs show reflectance spectra from the same pixels after dividing the original reflectance values by the mean reflectance from 890 to 910 nm. The solid lines with transparent bands show the mean values with S.D. Red vertical bands show the wavelength from 890 to 910 nm. 


### Step 2 procedure: 
```python
'''
Step 2-1: SVD Transformation
Perform Singular Value Decomposition (SVD) transformation and save the first four SVD components. 
In Krishnamoorthi S (2024), the central and peripheral regions of CNT, 0xFe, 0xP and 0xN were utilized.
'''

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

# Plot the SVD results
for i in [0, 2]:
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=np.array(svd_values[:, i]), y=np.array(svd_values[:, i + 1]), hue=label)
    plt.xlabel('Dimension %i (%.1f %%)' % (i + 1, (100 * svd.explained_variance_ratio_[i])))
    plt.ylabel('Dimension %i (%.1f %%)' % (i + 2, (100 * svd.explained_variance_ratio_[i + 1])))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
```

### Step 3: Transformation of the representative images into SVD spaces 
The transformation matrices of SVD can be used to project high-dimensional hyperspectral data into a small-dimensional space (two-dimensional image with six wavelength channels obtained with the SVD transformation matrix). Each of major SVD channels highlights distinct spectral features contained in the original data cube. 

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figures/Figure3-1.png" alt="Alt text" width="50%">
Figure 3-1. Singular Value Decomposition. In this project, leaf reflectance spectra are represented by a matrix M, where each column corresponds to different wavelengths, and each row corresponds to individual plants. SVD decomposes this matrix A into three matrices: U, Σ, and V*. The left singular vectors U (columns of U) capture distinct patterns or characteristics present in the reflectance spectra. Specifically, the four columns of U represent values in the first four dimensions of SVD, which can be thought of as unique features extracted from the data.
Now, focus on the right singular vectors V*, or more precisely, the rows of V*. The first four rows of V* act as a weight matrix, revealing how leaf reflectance at individual wavelengths contribute to the identified patterns represented by the first four SVD modes. Each row in V* helps us understand the significance of specific wavelengths in shaping the major patterns (SVD0 – SVD3) discovered in the leaf reflectance spectra. Image source: Wikipedia (https://en.wikipedia.org/wiki/Singular_value_decomposition).<p></p>

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figures/Figure3-2.png" alt="Alt text" width="100%">
Figure 3-2. (A) SVD analysis of thallus reflectance spectra. Thallus reflectance from the whole area is plotted on the first four columns of left singular vectors (SVD 0 – SVD 3) that were calculated from the central and peripheral spectral data. Colours show different nutrient deficiency treatments. (B) Line graphs showing the right singular vector rows for the first four dimensions of SVD. Green, red and blue vertical lines show absorption wavelengths for chlorophyll, anthocyanin and water.  Image source: Krishnamoorthi S et al. (2024) [https://www.cell.com/cell-reports/home].

### Step 3 procedure:
```python
'''
Step 3: Generation of SVD pseudo-color
Identify the SVD component(s) that best highlight leaf color patterns.
The right singular vectors V* represent a weight matrix, revealing how leaf reflectance at individual wavelengths contributes to the identified patterns.
'''

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
```
```python
'''
Plot SVD components values in box plots. 
'''
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
    df_boxplot['ID'] = np.concatenate((all_spectra['label'] + '_peripheral', all_spectra['label'] + '_paracentral', all_spectra['label'] + '_central'))
    fig, _ = SVD_box_plot(df_boxplot, group_by='ID', data_column=f'SVD {i}')
```

### Step 4: Pseudo-coloring of hyperspectral leaf images
Based on the leaf images and density plots generated with different SVD channels, users select SVD channel(s) that highlight leaf patterns associated with plant nutrient stresses and save the transformation matrix in the image processing software. The transformation matrix can be applied to hyperspectral images of any other leaves to help camera users to visually assess and quantify plant stress symptoms.<p></p>

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figures/Figure4.png" alt="Alt text" width="70%">
Figure 4. (A, B) RGB and pseudo-coloured images of liverworts treated under full Yamagami (Cntl), 0 mM NO3 (0xN), 0 mM PO4 (0xP) and 0 mM Fe (0xFe) conditions. Colour bars represent pixel intensities in the pseudo-colour spaces SVD 1-3. Red and pink arrows indicate pigmented and senesced thallus areas in RGB, SVD2 and SVD3 images. The values are extracted from the peripheral, paracentral, and central areas and shown in the box plots. Scale bar represents 1 cm.  

### Step 4 procedure:
```python
'''
Step 4: Application
Apply the selected SVD component(s) to hyperspectral images, and visualize them in SVD pseudo-colors
'''
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
```

## Citation
Shalini Krishnamoorthi, Grace Zi Hao Tan, Yating Dong, Richalynn Leong, Ting-Ying Wu, Daisuke Urano (2024) [https://www.cell.com/cell-reports/home].<br>
Shalini Krishnamoorthi, Daisuke Urano (2024) Figshare .[https://figshare.com/account/projects/180352/articles/24257317]
