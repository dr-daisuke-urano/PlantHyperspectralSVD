# -*- coding: utf-8 -*-
"""
Updated on March 9th 2021


@author: Daisuke
"""

#%%
# Load a hyperspectral cube
# path: absolute path to .hdr file 

def hsi_loading (path):
    import spectral as spy
    import numpy as np
    
    ID = path.split('\\')[-1]
    print(ID)

    data_ref = spy.io.envi.open(path+'\\capture\\' + ID+'.hdr', path + '\\capture\\' + ID + '.raw')
    white_ref = spy.io.envi.open(path+'\\capture\\' +'WHITEREF_'+ ID+'.hdr', path+'\\capture\\' +'WHITEREF_'+ ID+'.raw')
    dark_ref = spy.io.envi.open(path+'\\capture\\' +'DARKREF_'+ ID+'.hdr', path+'\\capture\\' +'DARKREF_'+ ID+'.raw')

    white = np.array(white_ref.load())
    dark = np.array(dark_ref.load())
    cube = np.array(data_ref.load())
    cube = (cube - dark) / (white - dark)
    
    return cube

#%%
# Reconstruct RGB image from hyperspectral cube
# RGB reference (.csv file) was made using the ColorChecker sheet. 

def RGB(cube):
    
    import numpy as np

    # R, G, and B reference reflectance values from the Channel 6 to 98
    B_reference = np.array([0.28462822, 0.28685858, 0.28758628, 0.29093164, 0.29571122,
           0.30220778, 0.30079957, 0.27963749, 0.29555491, 0.31906699,
           0.32516116, 0.32451853, 0.32559147, 0.32121282, 0.31593104,
           0.31127376, 0.30159306, 0.29403532, 0.28518392, 0.27803837,
           0.26438833, 0.25220742, 0.23785448, 0.22080147, 0.20234738,
           0.18885382, 0.1813013 , 0.17122129, 0.16100587, 0.1516296 ,
           0.14161461, 0.13246509, 0.12350839, 0.11398222, 0.1053238 ,
           0.09502631, 0.08695949, 0.0798408 , 0.07264553, 0.06786152,
           0.06351456, 0.0600972 , 0.05584277, 0.04966747, 0.04722196,
           0.04584859, 0.04574296, 0.04914031, 0.04966885, 0.0486881 ,
           0.0484957 , 0.04782599, 0.04706948, 0.04643554, 0.04711054,
           0.0459301 , 0.04562978, 0.04483982, 0.04478453, 0.04492691,
           0.04528737, 0.04501297, 0.044905  , 0.04501652, 0.0449133 ,
           0.04555969, 0.04434773, 0.03968953, 0.03880938, 0.0427795 ,
           0.04565767, 0.04564508, 0.04515549, 0.04575521, 0.04612792,
           0.04700006, 0.04855985, 0.04939982, 0.04975982, 0.05118658,
           0.05134406, 0.05192206, 0.0528893 , 0.05424882, 0.05466969,
           0.05568517, 0.05646897, 0.05624799, 0.05528999, 0.05502877,
           0.05431273, 0.0536918 , 0.05313713])
    
    G_reference = np.array([0.21133903, 0.19229854, 0.18078605, 0.16417557, 0.15436052,
           0.14840434, 0.12556966, 0.09313943, 0.0970141 , 0.12014251,
           0.11981628, 0.11819124, 0.12000915, 0.11835837, 0.1193578 ,
           0.12030675, 0.1218589 , 0.12349002, 0.12709269, 0.13230053,
           0.13579399, 0.14191656, 0.14937976, 0.15476981, 0.15940244,
           0.16737199, 0.18270389, 0.2020068 , 0.22317632, 0.24598215,
           0.27072405, 0.29505249, 0.31961973, 0.34010537, 0.35724534,
           0.36693443, 0.3737123 , 0.37546906, 0.37373522, 0.37074009,
           0.3664101 , 0.35928211, 0.3499074 , 0.33562034, 0.32473127,
           0.31567165, 0.31262132, 0.30839412, 0.29890602, 0.28856173,
           0.27746994, 0.26626604, 0.25537294, 0.24514435, 0.2363445 ,
           0.22501424, 0.21382301, 0.20501528, 0.19609266, 0.18763979,
           0.17928106, 0.17153051, 0.16436592, 0.15626209, 0.1498779 ,
           0.1446759 , 0.13768314, 0.12677298, 0.12379434, 0.12577324,
           0.12516931, 0.12377166, 0.12184466, 0.12027606, 0.11963645,
           0.11984991, 0.11964515, 0.11898388, 0.11754042, 0.11713352,
           0.11657382, 0.11545898, 0.11547345, 0.11564586, 0.11568949,
           0.11692526, 0.11748922, 0.11804433, 0.11938791, 0.12119119,
           0.12227899, 0.1234448 , 0.12521574])
    
    R_reference = np.array([0.19379772, 0.18053199, 0.16449151, 0.14728372, 0.13217046,
           0.12471709, 0.09912372, 0.06362731, 0.06501805, 0.08884007,
           0.0872441 , 0.0855771 , 0.0840513 , 0.0828051 , 0.08234076,
           0.08084493, 0.08133215, 0.08087183, 0.08254146, 0.08292572,
           0.08340879, 0.08386889, 0.08278485, 0.08019788, 0.07456487,
           0.06964235, 0.07022374, 0.0718186 , 0.07329241, 0.07471453,
           0.07527669, 0.07545765, 0.07577908, 0.07413419, 0.07420735,
           0.07298076, 0.07303222, 0.07371355, 0.07331879, 0.07373113,
           0.07325049, 0.07174468, 0.06877615, 0.06185071, 0.05729419,
           0.05601269, 0.05861221, 0.06474009, 0.06926641, 0.07156813,
           0.07453253, 0.07746038, 0.07975167, 0.08327926, 0.08758171,
           0.08938982, 0.09140255, 0.09479274, 0.09979963, 0.10750622,
           0.11896181, 0.13528612, 0.15544309, 0.18053507, 0.20842546,
           0.24427374, 0.28137826, 0.31276189, 0.34233328, 0.4002486 ,
           0.45385175, 0.49514783, 0.53280182, 0.56352838, 0.58944713,
           0.61350888, 0.63375058, 0.64685355, 0.65512809, 0.66278691,
           0.66776043, 0.67205826, 0.67632037, 0.68149331, 0.68275553,
           0.68573297, 0.68618982, 0.68956697, 0.68951365, 0.69225473,
           0.69300172, 0.69386459, 0.69621948])
        
    B = (cube[:,:,6:99] * B_reference[np.newaxis, np.newaxis, :]).sum(axis = 2)
    G = (cube[:,:,6:99] * G_reference[np.newaxis, np.newaxis, :]).sum(axis = 2)
    R = (cube[:,:,6:99] * R_reference[np.newaxis, np.newaxis, :]).sum(axis = 2)

    img = np.array([B/np.percentile(B,90), G/np.percentile(G, 90), R/np.percentile(R, 90)])
    img = img.transpose(1,2,0)
    img[img > 1] = 1
    img = (200*img).astype(np.uint8)
    img = adjust_brightness(img, 50)
    
    return img

#%%
# Adjust brightness of single image
# img:      cv2 image 
# value:    integer from -255 to +255. 

def adjust_brightness(img, value = 0):
    
    import cv2
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if 0 <= value <= 255:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value 
 
    else:
        raise ValueError(f"Value {value} is outside the range 0 to 255.")
        
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

#%%
# Select ROIs (plants) from a hysperspectral cube, and returns a dataframe and an image  
# The dataframe contains 4 columns 'ID', 'label', 'xy, 'Refectance'

class CoordinateStore:   
    def __init__(self):
        self.points = []

    def select_point(self,event,x,y,flags,param):
            import cv2
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(img,(x,y),3,(0,0,0),-1)
                cv2.putText(img, str(len(self.points)), (x+5,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                self.points.append([x,y])
        
def hsi_selection (cube):
    
    import cv2
    import pandas as pd
    import numpy as np
    global img

    #instantiate class
    coordinateStore1 = CoordinateStore()
    img = RGB(cube)
    
    cv2.namedWindow('Left Click: select plants, q: quit', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Left Click: select plants, q: quit', coordinateStore1.select_point)
    
    while True:
        cv2.imshow('Left Click: select plants, q: quit', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    spectrum_data = []

    for roi in coordinateStore1.points:
        spectrum = np.mean(cube[roi[1]-2:roi[1]+3, roi[0]-2:roi[0]+3,:], axis = (0,1))
        spectrum_data.append(spectrum)

    img = img.astype(np.uint8)
    df = pd.DataFrame([range(len(coordinateStore1.points)), [" "]*len(coordinateStore1.points), coordinateStore1.points, spectrum_data], index = ['ID', 'label', 'xy', 'ROI Reflectance']).transpose()
    return df, img
    
#%%
def spectral_comparison_SR (df, path = None):
    import matplotlib.pyplot as plt
    import numpy as np
        
    label = df.index.unique()
    bands = np.linspace(400, 1000, 204)
    fig = plt.figure()
    
    for name in label:
        plt.plot(bands, df[df.index == name].mean(axis = 0), label = name)
        plt.fill_between(bands, df[df.index == name].mean(axis = 0)-df[df.index == name].std(axis = 0), df[df.index == name].mean(axis = 0)+df[df.index == name].std(axis = 0), alpha = 0.2)
    
    plt.axvline(x = 428, color = 'palegreen', linestyle = '--', zorder=-1)
    plt.axvline(x = 660, color = 'palegreen', linestyle = '--', zorder=-1)
    plt.axvline(x = 546, color = 'lightpink', linestyle = '--', zorder=-1)
    plt.axvline(x = 970, color = 'lightblue', linestyle = '--', zorder=-1)
    plt.title('Mean reflectance ± SD, image %s' % path.split('\\')[-1])
    plt.xlabel('Wavelength[nm]')
    plt.ylabel('Reflectance')
    plt.legend(loc='lower right') 
    plt.show()
    fig.savefig(path + r'\%s_spectral_comparison.svg' % path.split('\\')[-1])
    
    return fig

#%%
def hsi_svd (cube, bands, dim = 10, path = None):
    import sklearn.decomposition as skd
    import matplotlib.pyplot as plt
    
    x,y,lamda = cube[:,:,:].shape 
    hsi2D = cube[:,:,:].reshape(x*y,lamda)
        
    model_svd = skd.TruncatedSVD(dim)
    hsi2D_svd = model_svd.fit_transform(hsi2D)
    hsi3D_svd = hsi2D_svd.reshape(x,y,dim)
            
    for n in range (dim):
        fig = plt.figure()
        fig.suptitle('SVD component %i' %n + ' \n Explained Variance Ratio: %.3f' % model_svd.explained_variance_ratio_[n])
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(bands, model_svd.components_[n])
        ax1.set_xlabel('Wavelength[nm]')
        ax1.set_ylabel('Reflectance')
        ax2.imshow(hsi3D_svd[:,:,n], cmap = 'bwr')
        ax2.axis('off') 
        if not path is None:
            fig.savefig(path + r'\SVD_component_%i.svg' %n)
        
    return model_svd

#%%
def hsi_nmf (cube, bands, dim = 10, path = None):
    import sklearn.decomposition as skd
    import matplotlib.pyplot as plt
    
    x,y,lamda = cube[:,:,:].shape 
    hsi2D = cube[:,:,:].reshape(x*y,lamda)
    
    model_nmf = skd.NMF(n_components=dim)
    hsi2D_nmf = model_nmf.fit_transform(hsi2D)
    hsi3D_nmf = hsi2D_nmf.reshape(x,y,dim)
    
    for n in range (dim):
        fig = plt.figure()
        fig.suptitle('NMF component %i' %n)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(bands, model_nmf.components_[n])
        ax1.set_xlabel('Wavelength[nm]')
        ax1.set_ylabel('Reflectance')
        ax2.imshow(hsi3D_nmf[:,:,n], cmap = 'bwr')
        ax2.axis('off')
        if not path is None:
            fig.savefig(path + r'\NMF_component_%i.svg' %n)
        
    return model_nmf

#%%
def three_areas (mask, x, y, r):
    
    import numpy as np
    import cv2
    
    outer = np.zeros(mask.shape[:2], np.uint8)  
    cv2.circle(outer,(int(x),int(y)), int(r), 1, thickness=-1) 
    outer = cv2.bitwise_and(mask, outer)

    middle = np.zeros(mask.shape[:2], np.uint8)  
    cv2.circle(middle,(int(x),int(y)), int(2*r/3), 1, thickness=-1)
    middle = cv2.bitwise_and(mask, middle)

    central = np.zeros(mask.shape[:2], np.uint8)  
    cv2.circle(central,(int(x),int(y)), int(r/3), 1, thickness=-1)
    central = cv2.bitwise_and(mask, central)

    outer = cv2.bitwise_and(outer, cv2.bitwise_not(middle))
    middle = cv2.bitwise_and(middle, cv2.bitwise_not(central))

    return central, middle, outer


#%%
def plant_masking(cube, threshold=2.2, kn=1):
    import cv2
    import pandas as pd
    import numpy as np
    
    # Convert the hyperspectral cube to RGB image
    img = RGB(cube)

    # Mask white pixels (reflection from a plate) using a brightness threshold
    _, white = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 235, 1, cv2.THRESH_BINARY_INV)

    # Load the SVD model
    model = pd.read_csv(r'C:/Users/Admin/OneDrive - Temasek Life Sciences Laboratory/Experimental_Data/Hyperspectral_Imaging/model_svd.csv').iloc[1, 1:]

    # Apply SVD model to hyperspectral data
    svd_pic = np.dot(cube[:, :, 10:200], model)

    # Threshold the SVD image to create a mask
    _, mask = cv2.threshold(svd_pic, threshold, 1, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up the mask
    kernel = np.ones((kn, kn), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = mask * white

    return img, mask, svd_pic

#%%
def data_extraction_SR (cube, df_loc, threshold = 2.2, kn = 1, dist = 15, path = None):  
    import cv2
    import pandas as pd
    import numpy as np
    import scipy.spatial as spatial  
    from matplotlib import pyplot as plt
    import seaborn as sns

    # Get the mask and related variables
    img, mask, svd_pic = plant_masking(cube, threshold, kn)
        
    # Initialize the data list with column names
    data = [[
        'ImageID', 'ID', 'label', 'leaf area [px]', 'radius [px]', 
        'reflectance', 'peripheral', 'paracentral', 'central'
    ]]
    
    # Find contours in the mask
    contours, _ = cv2.findContours(
        cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8), 3).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    for c in contours:
        if cv2.contourArea(c) > 25:  # Minimum contour area in pixels
            cimg = np.zeros(img.shape[:2])
            cv2.drawContours(cimg, [c], -1, color=255, thickness=-1)
            coordinates = np.argwhere(cimg)[:, ::-1]
            for i in range(len(df_loc['xy'])):
                ID = df_loc['ID'][i]
                xy = np.asarray(df_loc['xy'][i]).reshape(1, -1)
                if np.min(spatial.distance.cdist(np.asarray(xy), np.array(coordinates[:]))) < dist:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    
                    # Extract areas using the three_areas function
                    inner, middle, outer = three_areas(mask, x, y, radius)
                    outer_cube = cube * cimg[:, :, np.newaxis] * outer[:, :, np.newaxis]
                    middle_cube = cube * cimg[:, :, np.newaxis] * middle[:, :, np.newaxis]
                    inner_cube = cube * cimg[:, :, np.newaxis] * inner[:, :, np.newaxis]

                    # Draw the contour and circle
                    cv2.circle(img, (int(x), int(y)), int(radius), (255, 0, 255), 1)
                    
                    # Calculate leaf area
                    area = np.sum(cimg * mask) / 255
                    plant_cube = cube * cimg[:, :, np.newaxis] * mask[:, :, np.newaxis]

                    # Append data to the list
                    data.append([
                        path.split('\\')[-1], df_loc['ID'][i], df_loc['label'][i], area, radius,
                        np.nansum(plant_cube, axis=(0, 1)) / np.sum(cimg * mask),
                        np.nansum(outer_cube, axis=(0, 1)) / np.sum(cimg * outer),
                        np.nansum(middle_cube, axis=(0, 1)) / np.sum(cimg * middle),
                        np.nansum(inner_cube, axis=(0, 1)) / np.sum(cimg * inner)
                    ])
                    print(f"ID {ID} Leaf area = {area} px")

    for i in range(len(df_loc['xy'])):
        if df_loc['label'][i] is not None:
            cv2.putText(
                img, f"{i}:{df_loc['label'][i]}", 
                (df_loc['xy'][i][0] - int(dist), df_loc['xy'][i][1] + int(dist)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(255, 0, 255), thickness=1
            )

    # Show Masked image
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1 = sns.heatmap(
        svd_pic[::-1, :], cmap='rainbow', vmin=-4, vmax=8, square=True, cbar=True, cbar_kws={"shrink": 0.5}
    )
    ax1.invert_yaxis()
    ax1.axis('off')
    ax1.set_title(f'Threshold val: {threshold:.1f}')
    
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(np.flip(RGB(cube) * mask[:, :, np.newaxis],2))
    ax2.axis('off')
    ax2.set_title(f'Masked image {path.split("\\")[-1]}')
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(np.flip(img, 2))
    ax3.axis('off')
    ax3.set_title(f'Contour image {path.split("\\")[-1]}')
    
    fig.set_size_inches(18, 6)
    fig.savefig(path + rf'\{path.split("\\")[-1]}_masked_image.png')
    fig.show()

    # Create DataFrame and flatten data
    masked_cube = cube * mask[:, :, np.newaxis]
    df = pd.DataFrame(data[1:], columns=data[0]).sort_values('ID')
    df_flatten = pd.concat([
        df.iloc[:, 3:5], 
        pd.DataFrame(df['reflectance'].apply(pd.Series), columns=None), 
        pd.DataFrame(df['peripheral'].apply(pd.Series), columns=None), 
        pd.DataFrame(df['paracentral'].apply(pd.Series), columns=None), 
        pd.DataFrame(df['central'].apply(pd.Series), columns=None)
    ], axis=1)
    
    df_flatten.index = pd.MultiIndex.from_frame(df.iloc[:, :3])
    df_flatten.columns = [
        'leaf area [px]', 'radius [px]'
    ] + [f"whole_{a}" for a in np.arange(204)] + [f"peripheral_{a}" for a in np.arange(204)] + [f"paracentral_{a}" for a in np.arange(204)] + [f"central_{a}" for a in np.arange(204)]
    
    df_flatten.to_csv(path + rf'\{path.split("\\")[-1]}_spectrum.csv', index=True)
                
    return df_flatten, img, masked_cube


#%%
"""
Anthocyanin reflectance index ARI:   1/550nm−1/700nm
Modified ARI mARI: (1/550nm−1/700nm)⋅NIR  

Carotenoid reflectance index CRI550: 1/510nm-1/550nm
Carotenoid reflectance index CRI700: 1/510nm-1/700nm
new Carotenoid Index nCI: 720nm/521nm - 1

Leaf    Chlorophyll Index LCI: ([850]−[710])/([850]+[680])
MERIS Terrestrial chlorophyll index MTCI: 754nm−709nm/709nm−681nm
Modified Chlorophyll Absorption Ratio Index 710 	MCARI710 	((750nm−710nm)−0.2(750nm−550nm))*(750nm/710nm)

Water Index WI: 900nm/970nm

Single bands
    Chlorophyll a: 428nm, 660nm
    Chlorophyll b: 453nm, 645nm
    Anthocyanin (pH 1): 546nm
    Water : 970nm, 1450nm, 1930nm
"""
def spectral_overview_SR(df, path = None):
    import numpy as np
    import pandas as pd
    
    df.columns = np.linspace(400, 1000, 204)
    
    ARI = np.reciprocal(df.iloc[:, abs(df.columns - 550) <= 5].mean(axis =1)) - np.reciprocal(df.iloc[:, abs(df.columns - 700) <= 5].mean(axis =1))
    mARI = ARI * df.iloc[:, abs(df.columns - 800) <= 50].mean(axis =1) 
    
    CRI550 = np.reciprocal(df.iloc[:, abs(df.columns - 510) <= 5].mean(axis =1)) - np.reciprocal(df.iloc[:, abs(df.columns - 550) <= 5].mean(axis =1))
    CRI700 = np.reciprocal(df.iloc[:, abs(df.columns - 510) <= 5].mean(axis =1)) - np.reciprocal(df.iloc[:, abs(df.columns - 700) <= 5].mean(axis =1))
    CARI = df.iloc[:, abs(df.columns - 720) <= 5].mean(axis =1) / df.iloc[:, abs(df.columns - 521) <= 5].mean(axis =1) - 1
    
    LCI = (df.iloc[:, abs(df.columns - 850) <= 5].mean(axis =1) - df.iloc[:, abs(df.columns - 710) <= 5].mean(axis =1)) / (df.iloc[:, abs(df.columns - 850) <= 5].mean(axis =1) + df.iloc[:, abs(df.columns - 680) <= 5].mean(axis =1))
    MTCI = (df.iloc[:, abs(df.columns - 754) <= 5].mean(axis =1) - df.iloc[:, abs(df.columns - 709) <= 5].mean(axis =1)) / (df.iloc[:, abs(df.columns - 709) <= 5].mean(axis =1) - df.iloc[:, abs(df.columns - 681) <= 5].mean(axis =1))
    MCARI710 = ((df.iloc[:, abs(df.columns - 750) <= 5].mean(axis =1) - df.iloc[:, abs(df.columns - 710) <= 5].mean(axis =1)) - 0.2*(df.iloc[:, abs(df.columns - 750) <= 5].mean(axis =1) - df.iloc[:, abs(df.columns - 550) <= 5].mean(axis =1))) / (df.iloc[:, abs(df.columns - 750) <= 5].mean(axis =1) - df.iloc[:, abs(df.columns - 710) <= 5].mean(axis =1))
    
    WI = df.iloc[:, abs(df.columns - 900) <= 5].mean(axis =1) / df.iloc[:, abs(df.columns - 970) <= 5].mean(axis =1) 
    WI880 = (df.iloc[:, abs(df.columns - 970) <= 5].mean(axis =1) - df.iloc[:, abs(df.columns - 880) <= 5].mean(axis =1)) / (df.iloc[:, abs(df.columns - 970) <= 5].mean(axis =1) + df.iloc[:, abs(df.columns - 880) <= 5].mean(axis =1))
    
    R428 = df.iloc[:, abs(df.columns - 428) <= 5].mean(axis =1)
    R546 = df.iloc[:, abs(df.columns - 546) <= 5].mean(axis =1)
    R660 = df.iloc[:, abs(df.columns - 660) <= 5].mean(axis =1)
    R970 = df.iloc[:, abs(df.columns - 970) <= 5].mean(axis =1)
    
    df_overview = pd.DataFrame({'ARI(1R550-1R700)':ARI,'mARI':mARI,'LCI(chl)':LCI, 'MTCI(chl)':MTCI, 'MCARI710(chl)':MCARI710,
                          'CRI550':CRI550, 'CRI700':CRI700, 'CARI':CARI, 'WI(R900 R970 ratio)':WI, 'WI(R970-R880 R970+R880 ratio)':WI880,
                          'R428':R428, 'R660':R660, 'R546':R546, 'R970':R970})
    df_overview.to_csv(path + r'\%s_overview.csv' % path.split('\\')[-1], index= True)
    return df_overview
