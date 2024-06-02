# SVD-assisted visualization and assessment of spatial hyperspectral patterns of plant leaves

This is the official implementation for Shalini Krishnamoorthi et al. (2024) [https://www.cell.com/cell-reports/home].



## Project Summary
Leaf reflectance spectrum is widely used for plant stress diagnostics, especially for nutrient stresses and pathogen infections that cause distinct leaf color patterns. Hyperspectral cameras capture the reflectance of light with high spectral resolution (spectral resolution ranging from sub-nanometer to a few nanometers) and store this information in a data cube of x, y, and λ dimensions (two-dimensional images with multiple wavelength channels). This project describes Python code for visualizing spatial leaf color patterns using pseudo-color spaces created via singular value decomposition (SVD) of normalized hyperspectral images. 

The procedure consists of four steps:

1. Normalization: Normalize pixel intensity across all wavelength channels using the mean reflectance near 900 nm (bands from 875 to 925 nm are used in this step).
2. SVD Transformation: Perform SVD transformation and save the first five SVD spaces.
3. Pseudo-Color Generation: Generate pseudo-colored images and create density plots along the five SVD color spaces. The user then selects and saves the SVD color space(s) that effectively represent leaf color patterns.
4. Application: Apply the selected SVD space(s) to hyperspectral images of other leaves. In our publication, these pseudo-colored images were used to diagnose plant nutrient stresses in Marchantia polymorpha (liverwort) and Lactuca sativa (lettuce).

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figure1.png" alt="Alt text" width="70%">
Figure 1 Diagram summarizing this project. 

## Dependencies
To create a Conda environment with the dependencies used in Krishmoorthi S (2024), use the following command:

```bash
conda env create -f environment.yml
```

- Python 3.6+
- NumPy
- Matplotlib
- OpenCV
- x
- x
- x
- scikit-learn


## Sample Usage

### Prerequisites: 
The following code requires two or more hyperspectral images of representative leaves obtained from control and stress conditions. The control and stress condition images should be from the same species and at the same developmental stages (e.g., do not compare leaf reflectance patterns of rice with those of maize).

Before proceeding with step 1, Gaussian smoothing filter with the kernel of (2,2) was applied to the hyperspectral cube. In addition, background masking was performed to extract plant pixels from the hyperspectral cube. 

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

<img src="https://github.com/dr-daisuke-urano/Hyperspectral_Imaging/blob/main/Figure2.png" alt="Alt text" width="70%">
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

Figure 3-1. 

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
Based on the leaf images and density plots generated with different SVD channels, users select SVD channel(s) that highlight leaf patterns associated with plant nutrient stresses and save the transformation matrix in the image processing software. The transformation matrix can be applied to hyperspectral images of any other leaves to help camera users to visually assess and quantify plant stress symptoms.

### Step 4 procedure:
```python
"""
i. Normalize hyperspectral leaf images and mask background (see step 1) 
ii. Apply a selected-SVD transformation matrix to the normalized hyperspectral images. 
iii. Display the transformed image with a pseudo-color scale together with a density plot
"""

```

## Citation
Shalini Krishnamoorthi, Grace Zi Hao Tan, Yating Dong, Richalynn Leong, Ting-Ying Wu, Daisuke Urano (2024) [https://www.cell.com/cell-reports/home].
