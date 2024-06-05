# SVD-assisted visualization and assessment of spatial hyperspectral patterns of plant leaves
This is the official implementation for Shalini Krishnamoorthi et al. (2024) [https://www.cell.com/cell-reports/home].

## Background
Leaf reflectance spectra are widely utilized for diagnosing plant stress, particularly for nutrient deficiencies and pathogen infections, which often manifest as distinct leaf color patterns. Figure 0 shows Marchantia polymorpha plants grown under various nutrient deficiencies. Nitrate deficiency (0xN) and phosphate deficiency (0xP) result in early senescence and the accumulation of purple pigment in the central area, respectively. Iron deficiency (0xFe) induces chlorosis of the thallus starting from the peripheral growing area, while calcium deficiency (0.05xCa) leads to irreversible necrosis of the growing edges. As these leaf color changes are location-dependent, leaf reflectance spectra were separately obtained from three distinct regions: a central circle, paracentral annulus, and peripheral annulus (Figure 0A).

<p></P>

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figures/Figure0.png" alt="Alt text" width="70%">
Figure 0: (A) Illustration depicting the central, paracentral, and peripheral regions of Marchantia polymorpha plants. (B-F) Representative images and corresponding leaf reflectance spectra of M. polymorpha plants cultivated under various nutrient deficiencies. The graphs present mean values (solid lines) with standard deviation (translucent bands). Dashed lines in green, red, and blue denote absorption wavelengths for chlorophyll, anthocyanin, and water, respectively.

## Project Overview
Hyperspectral cameras capture the reflectance of light with high spectral resolution, storing this information in a data cube with x, y, and λ dimensions (two-dimensional images with multiple wavelength channels). This project provides Python code for visualizing spatial leaf color patterns using pseudo-color spaces created via singular value decomposition (SVD) of normalized hyperspectral images. The procedure consists of four steps:

1. Normalization: Normalize pixel intensity across all wavelength channels using the mean reflectance near 900 nm (bands from 875 to 925 nm are used in this step).
2. SVD Transformation: Perform SVD transformation and save the first five SVD spaces.
3. Pseudo-Color Generation: Generate pseudo-colored images and create density plots along the five SVD color spaces. The user then selects and saves the SVD color space(s) that effectively represent leaf color patterns.
4. Application: Apply the selected SVD space(s) to hyperspectral images of other leaves. In our publication, these pseudo-colored images were used to diagnose plant nutrient stresses in Marchantia polymorpha (liverwort) and Lactuca sativa (lettuce).

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figures/Figure1.png" alt="Alt text" width="70%">
Figure 1 Diagram summarizing this project. 

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
Before proceeding with Singular Value Decomposition, background masking, and extraction of leaf pixels from the three distinct areas are required. In this GitHub repository, we provide a simple GUI that assists users in selecting plant leaves and extracting spectral information. The following code obtains the mean reflectance spectra from pixels within the central, paracentral, and peripheral areas, as well as from the entire leaf, and saves them in CSV format. Sample hyperspectral images, control, phosphate deficiency, nitrate deficiency and iron deficiency, are provided at [https://github.com/dr-daisuke-urano/PlantHyperspectralSVD/tree/main/SPECIM_sample_images]

'''python
"""
i. Download and import PlantHyperspectralSVD.py 
ii. Divide pixel values at all wavelength channels by the respective mean nIR reflectance.
iii. 
"""
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
'''

Note: The images for both control and stress conditions should be of the same species and at similar developmental stages (e.g., avoid comparing leaf reflectance patterns of rice with those of maize).


### Step 1: Pixel-by-pixel normalization of leaf reflectance spectra with nIR bands.  
Plant leaves show high reflectance of near-infrared (nIR) light. While the reflectance at far-red wavelengths (700 – 780 nm) was greatly reduced under various stress conditions, the leaf reflectance at longer wavelengths is highly independent of plant growing conditions. First, we identified the wavelength bands that showed the smallest “coefficient of variations (CoV)” among the wavelength range from 400 to 1000 nm, which can then be used as the reference band to normalize the leaf reflectance spectra. The bands near 900 nm (890 – 910 nm) were originally selected from more than 100 spectral data obtained from control, nitrate deficiency, phosphate deficiency, iron deficiency, magnesium deficiency and calcium deficiency conditions in M. polymorpha. The small CoV at ~900 nm was consistently observed in L. sativa plants. 

### Step 1 procedure: 
```python
"""
i. For each pixel, calculate the mean nIR reflectance value from 890 to 910 nm.
ii. Divide pixel values at all wavelength channels by the respective mean nIR reflectance.
"""
```

### Step 2: Singular value decomposition (SVD)
SVD is a widely used method for dimensionality reduction. Most of the spectral information spanning different wavelength channels can be projected into a small number of dimensional spaces. This method is effective in extracting non-redundant spectral features which possibly highlight anomalous leaf color patterns. 

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figures/Figure2.png" alt="Alt text" width="70%">
Figure 2. Step 1: Pixel-by-pixel normalization of leaf reflectance spectra with nIR bands.  (A, B) Reflectance spectra obtained from representative plant leaves grown under control (A) and phosphate deficiency (B) conditions. Leaf reflectance at nIR bands from 890 to 910 nm are minimally affected by nutrient deficiencies but rather affected by light conditions. Blue graphs show the original reflectance spectra from leaf pixels. Orange graphs show reflectance spectra from the same pixels after dividing the original reflectance values by the mean reflectance from 890 to 910 nm. The solid lines with transparent bands show the mean values with S.D. Red vertical bands show the wavelength from 890 to 910 nm. 


### Step 2 procedure: 
```python
"""
i. Extract spectral information from all pixels into a data frame 
ii. Run singular value decomposition algorithm with seven iterations, and save the transformation matrices for six major SVD dimensional spaces. 
"""
```

### Step 3: Transformation of the representative images into SVD spaces 
The transformation matrices of SVD can be used to project high-dimensional hyperspectral data into a small-dimensional space (two-dimensional image with six wavelength channels obtained with the SVD transformation matrix). Each of major SVD channels highlights distinct spectral features contained in the original data cube. 

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figures/Figure3-1.png" alt="Alt text" width="50%">
Figure 3-1. Singular Value Decomposition. In this project, leaf reflectance spectra are represented by a matrix M, where each column corresponds to different wavelengths, and each row corresponds to individual plants. SVD decomposes this matrix A into three matrices: U, Σ, and V*. The left singular vectors U (columns of U) capture distinct patterns or characteristics present in the reflectance spectra. Specifically, the four columns of U represent values in the first four dimensions of SVD, which can be thought of as unique features extracted from the data.
Now, focus on the right singular vectors V*, or more precisely, the rows of V*. The first four rows of V* act as a weight matrix, revealing how leaf reflectance at individual wavelengths contribute to the identified patterns represented by the first four SVD modes. Each row in V* helps us understand the significance of specific wavelengths in shaping the major patterns (SVD0 – SVD3) discovered in the leaf reflectance spectra. Image source: Wikipedia (https://en.wikipedia.org/wiki/Singular_value_decomposition).<p></p>

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figure3-2.png" alt="Alt text" width="100%">
Figure 3-2. (A) SVD analysis of thallus reflectance spectra. Thallus reflectance from the whole area is plotted on the first four columns of left singular vectors (SVD 0 – SVD 3) that were calculated from the central and peripheral spectral data. Colours show different nutrient deficiency treatments. (B) Line graphs showing the right singular vector rows for the first four dimensions of SVD. Green, red and blue vertical lines show absorption wavelengths for chlorophyll, anthocyanin and water.  Image source: Krishnamoorthi S et al. (2024) [https://www.cell.com/cell-reports/home].

### Step 3 procedure:
```python
"""
i. Reduce spectral dimension of hyperspectral data cube using the SVD transformation matrix. 
ii. Show leaf images in the five SVD channels
iii. Show the distribution of pixel intensity in the five SVD channels using density plot
iv. Show the values in transformation matrix together with the density plot.  
"""

import numpy as np
import matplotlib.pyplot as plt

# Sample Python code
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave')
plt.show()
```

### Step 4: Pseudo-coloring of hyperspectral leaf images
Based on the leaf images and density plots generated with different SVD channels, users select SVD channel(s) that highlight leaf patterns associated with plant nutrient stresses and save the transformation matrix in the image processing software. The transformation matrix can be applied to hyperspectral images of any other leaves to help camera users to visually assess and quantify plant stress symptoms.<p></p>

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figures/Figure4.png" alt="Alt text" width="70%">
Figure 4. (A) RGB and pseudo-coloured images of M. polymorpha plants treated under full Yamagami (Cntl), 0 mM NO3 (0xN), 0 mM PO4 (0xP) and 0 mM Fe (0xFe) conditions. Colour bars represent pixel intensities in the pseudo-colour spaces SVD 1-3. Red and pink arrows indicate pigmented and senesced thallus areas in RGB, SVD2 and SVD3 images. The values were extracted from the peripheral, paracentral, and central areas and shown in the box plots D – F below. Scale bar represents 1 cm. (B) The box plots show the 25th, 50th and 75th percentiles with whiskers showing max and min values within 1.5 x IQR. Coloured dots in the boxplots represent the raw data from individual plants (n = 50 plants). 

### Step 4 procedure:
```python
"""
i. Normalize hyperspectral leaf images and mask background (see step 1) 
ii. Apply a selected-SVD transformation matrix to the normalized hyperspectral images. 
iii. Display the transformed image with a pseudo-color scale together with a density plot
"""

```

## Citation
Shalini Krishnamoorthi, Grace Zi Hao Tan, Yating Dong, Richalynn Leong, Ting-Ying Wu, Daisuke Urano (2024) [https://www.cell.com/cell-reports/home].<br>
Shalini Krishnamoorthi, Daisuke Urano (2024) Figshare .[https://figshare.com/account/projects/180352/articles/24257317]
