# -*- coding: utf-8 -*-
"""
Toolbox for Shalini's hyperspectral analysis, version 8
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
    
    import pandas as pd
    import numpy as np
    
    RGB_reference = pd.read_csv(r'S:/GraceT/python_code/RGB_reference.csv')
    
    B_channel = (cube[:,:,6:109] * RGB_reference['B'].values[np.newaxis, np.newaxis, 6:109]).sum(axis = 2)
    G_channel = (cube[:,:,6:109] * RGB_reference['G'].values[np.newaxis, np.newaxis, 6:109]).sum(axis = 2)
    R_channel = (cube[:,:,6:109] * RGB_reference['R'].values[np.newaxis, np.newaxis, 6:109]).sum(axis = 2)

    B_channel = (cube[:,:,6:99] * RGB_reference['B'].values[np.newaxis, np.newaxis, 6:99]).sum(axis = 2)
    G_channel = (cube[:,:,6:99] * RGB_reference['G'].values[np.newaxis, np.newaxis, 6:99]).sum(axis = 2)
    R_channel = (cube[:,:,6:99] * RGB_reference['R'].values[np.newaxis, np.newaxis, 6:99]).sum(axis = 2)

    img = np.array([B_channel/np.percentile(B_channel,90), G_channel/np.percentile(G_channel,90), R_channel/np.percentile(R_channel,90)])
    img = img.transpose(1,2,0)
    img[img > 1] = 1
    img = (200*img).astype(np.uint8)
    img = adjust_brightness(img, 50)
    
    return img
#%%
# Return a masked cube

def hsi_masked(cube, threshold_leaf = 0.6):
    
    import numpy as np
    import cv2
    
    img = RGB(cube)
    _, mask = cv2.threshold(1 - img[:,:,0]/255, threshold_leaf, 1, cv2.THRESH_BINARY) 
    masked_cube = cube * mask[:,:, np.newaxis]
    
    return masked_cube

#%%
# Adjust brightness of single image
# img:      cv2 image 
# value:    integer from -255 to +255. 

def adjust_brightness(img, value = 0):
    
    import cv2
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value 
 
    else:
        value = -value
        lim = value
        v[v < lim] = 0
        v[v >= lim] -= value         
    
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
    
    cv2.namedWindow('select plants manually', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('select plants manually', coordinateStore1.select_point)
    
    while True:
        cv2.imshow('select plants manually', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    spectrum_data = []

    for roi in coordinateStore1.points:
        spectrum = np.mean(cube[roi[1]-2:roi[1]+3, roi[0]-2:roi[0]+3,:], axis = (0,1))
        spectrum_data.append(spectrum)

    img = img.astype(np.uint8)
#    df = pd.DataFrame(np.transpose([range(len(coordinateStore1.points)), [" "]*len(coordinateStore1.points), coordinateStore1.points, spectrum_data]), columns = ['ID', 'label', 'xy', 'ROI Reflectance'])
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
def two_areas (mask, x, y, r):
    
    import numpy as np
    import cv2
    
    outer = np.zeros(mask.shape[:2], np.uint8)  
    cv2.circle(outer,(int(x),int(y)), int(r), 1, thickness=-1) 
    outer = cv2.bitwise_and(mask, outer)

    central = np.zeros(mask.shape[:2], np.uint8)  
    cv2.circle(central,(int(x),int(y)), int(r/2), 1, thickness=-1)
    central = cv2.bitwise_and(mask, central)

    outer = cv2.bitwise_and(outer, cv2.bitwise_not(central))

    return central, outer

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
'''
Note
cube: hyperspectral cube data from Specim camera
df_loc: dataframe 

'''
def data_extraction_SR (cube, df_loc, threshold = 2.2, kn = 1, dist = 15, path = None):
    
    import cv2
    import pandas as pd
    import numpy as np
    import scipy.spatial as spatial  
    from matplotlib import pyplot as plt
    import seaborn as sns
        
    data = [('ImageID', 'ID', 'label', 'leaf area [px]', 'radius [px]', 'reflectance', 'peripheral', 'paracentral', 'central')]
    img = RGB(cube)

    # Mask white pixels (reflection from a plate) using a brightness of 235
    _, white = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 235, 1, cv2.THRESH_BINARY_INV) 

    # The SVD model was generated from one of Grace's screening plate (Image ID: 272)
    model = pd.read_csv(r'C:/Users/kshalini/OneDrive - Temasek Life Sciences Laboratory/Python code/model_svd.csv').iloc[1, 1:]
 #   model = pd.read_csv(r'C:/Users/daisuke/OneDrive - Temasek Life Sciences Laboratory/Experimental_Data/Hyperspectral_Imaging/model_svd.csv').iloc[1, 1:]
    svd_pic = np.dot(cube[:,:,10:200], model)
    _, mask = cv2.threshold(svd_pic, threshold, 1, cv2.THRESH_BINARY_INV) 

    kernel = np.ones((kn,kn), dtype=np.uint8)
#    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = mask * white
    contours, _ = cv2.findContours(cv2.dilate(mask, np.ones((7,7), dtype=np.uint8), 3).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        if cv2.contourArea(c) > 25: # number of pixels 
            cimg = np.zeros(img.shape[:2])
            cv2.drawContours(cimg, [c], -1, color=255, thickness=-1)
            coordinates = np.argwhere(cimg)[:,::-1]
            for i in range(len(df_loc['xy'])):
                ID = df_loc['ID'][i]
                xy = (np.asarray(df_loc['xy'][i])).reshape(1,-1)
                if np.min(spatial.distance.cdist(np.asarray(xy), np.array(coordinates[:]))) < dist:

                    (x,y), radius = cv2.minEnclosingCircle(c)
            
                    inner, middle, outer = three_areas(mask, x, y, radius)
                    outer_cube = cube * cimg[:,:, np.newaxis] * outer[:,:, np.newaxis]
                    middle_cube = cube * cimg[:,:, np.newaxis] * middle[:,:, np.newaxis]
                    inner_cube = cube * cimg[:,:, np.newaxis] * inner[:,:, np.newaxis]

#                    cv2.drawContours(img, c, -1, color = (255, 0, 0), thickness = 1)
                    cv2.circle(img, (int(x),int(y)), int(radius), (255, 0, 255), 1)                        
                    area = np.sum(cimg * mask)/255    #This is more accurate than using area enclosed in a contour.
                    plant_cube = cube * cimg[:,:, np.newaxis] * mask[:,:, np.newaxis]
                    data.append([path.split('\\')[-1], df_loc['ID'][i], df_loc['label'][i], area, radius, np.nansum(plant_cube, axis = (0,1))/np.sum(cimg*mask), np.nansum(outer_cube, axis = (0,1))/np.sum(cimg*outer), np.nansum(middle_cube, axis = (0,1))/np.sum(cimg*middle), np.nansum(inner_cube, axis = (0,1))/np.sum(cimg*inner)])
                    print("ID " + str(ID) + " Leaf area = %s px" % str(np.sum(cimg*mask)/255))
                    
    for i in range(len(df_loc['xy'])):
        if df_loc['label'][i] != None:
            cv2.putText(img, str(i) + ':' + df_loc['label'][i], (df_loc['xy'][i][0]-int(dist), df_loc['xy'][i][1]+int(dist)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(255,0,255), thickness=1)
                                                                             
    # Show Masked image
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1 = sns.heatmap(svd_pic[::-1,:], cmap = 'rainbow', vmin = -4, vmax = 8, square = True, cbar_kws={"shrink": 0.5})
    ax1.invert_yaxis()
    ax1.axis('off')
    ax1.set_title('Threshold val: %.1f' % threshold)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(RGB(cube) * mask[:,:, np.newaxis])
    ax2.axis('off')    
    ax2.set_title('Masked image %s' % path.split('\\')[-1])
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(img)
    ax3.axis('off')
    ax3.set_title('Contour image %s' % path.split('\\')[-1])
    fig.set_size_inches(18, 6)
    fig.savefig(path + r'\%s_masked_image.png' % path.split('\\')[-1])
    fig.show()
    
    masked_cube = cube * mask[:,:, np.newaxis]
    df = pd.DataFrame(data[1:], columns = data[0]).sort_values('ID')
    df_flatten = pd.concat([df.iloc[:,3:5], pd.DataFrame(df['reflectance'].apply(pd.Series), columns= None), pd.DataFrame(df['peripheral'].apply(pd.Series), columns= None), pd.DataFrame(df['paracentral'].apply(pd.Series), columns= None), pd.DataFrame(df['central'].apply(pd.Series), columns= None)], axis = 1)  
    df_flatten.index = pd.MultiIndex.from_frame(df.iloc[:,:3])
    df_flatten.columns = ['leaf area [px]', 'radius [px]'] + ["whole_" +str(a) for a in np.arange(204)] + ["peripheral_" +str(a) for a in np.arange(204)] + ["paracentral_" +str(a) for a in np.arange(204)] + ["central_" +str(a) for a in np.arange(204)]
    df_flatten.to_csv(path + r'\%s_spectrum.csv' % path.split('\\')[-1], index = True)
                
    return df_flatten, img, masked_cube

#%%
def SQL_write (df, path):
    import sqlite3
    
    try:
        conn = sqlite3.connect(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Experimental_Data\Hyperspectral_Imaging\hyperspec_SQL.sqlite')
        df.to_sql('hyperspec'+path.split('\\')[-1], conn, if_exists='replace', index=False)
        cursor = conn.cursor()
        print("")
        print("Database created and Successfully Connected to SQLite")
    
        sqlite_select_Query = "select sqlite_version();"
        cursor.execute(sqlite_select_Query)
        record = cursor.fetchall()
        print("SQLite Database Version is: ", record)
        cursor.close()
    
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if (conn):
            conn.close()
            print("The SQLite connection is closed")    
            
#%%
def SQL_read(table_name):
    import sqlite3
    import pandas as pd
    
    try:
        conn = sqlite3.connect(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Experimental_Data\Hyperspectral_Imaging\hyperspec_SQL.sqlite')
        cursor = conn.cursor()
        df = pd.read_sql('select * from hyperspec%s' % table_name, conn)
        print("")
        print("Database successfully connected to SQLite")
    
        sqlite_select_Query = "select sqlite_version();"
        cursor.execute(sqlite_select_Query)
        record = cursor.fetchall()
        print("SQLite Database Version is: ", record)
        cursor.close()
    
    except sqlite3.Error as error:
        print("Error while connecting to SQLite", error)
    finally:
        if (conn):
            conn.close()
            print("The SQLite connection is closed")   
        
    return df

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

#%%
def violin_plot (df, group_by = 'ID', data_column = None, cnt = None):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # if group_by not in df.columns:
    #     print ('no %s column found.' %group_by)
    
    if data_column not in df.columns:
        print ('data not found. Check the header name in spreadsheet')
       
    mean_df = df.groupby(group_by, as_index = True).mean()
    df = df.set_index(group_by)    
    print(mean_df[data_column])
    plt.figure()

    ax = sns.violinplot(x= df.index, y= data_column, data=df, color="0.8")
    ax = sns.stripplot(x= df.index, y= data_column, data=df, jitter=True)
    ax.set_position([0.25,0.3,0.65,0.6])
    xlabels = ax.xaxis.get_ticklabels(minor=False)
    
    for i in range(len(xlabels)):
        if xlabels[i].get_text() == cnt:
            ci95 = ax.get_lines()[2*i].get_data()[1]
            ax = plt.axhline(y = mean_df.loc[cnt, data_column], color = 'gold', linestyle = '-', zorder=-1)
            ax = plt.axhspan(ci95[0], ci95[1], facecolor= 'yellow', alpha = 0.25, zorder=-1)
            print (cnt + ' is defined as the control group.')

    
    plt.xticks(rotation = 90)     
    fig = ax.get_figure()
    fig.set_size_inches(mean_df.count()[0]/2 + 1, 4)
    
    return fig, ax


#%%
def ANOVA_data (df, data_column = None):
    
    import scipy.stats as ss
    import pandas as pd
    import researchpy as rp
#    import statsmodels.stats.multicomp as multi
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    if data_column not in df.columns:
        print('data not found. Check the header name in DataFrame')
    
    summary = rp.summary_cont(df[data_column].groupby(df['ID']))
    summary.index.name = data_column
    samples = []
    for group in df['ID'].unique():
        samples.append(df[data_column][df['ID'] == group].values)
    f_val, p_val = ss.f_oneway(*samples)
    if p_val < 0.05:        
        results = pairwise_tukeyhsd(endog = df[data_column], groups = df['ID'])
        results = pd.DataFrame(data=results._results_table.data[1:], columns=results._results_table.data[0])
        results.index.name = data_column
        print(results)
    else:
        print(data_column + ': There is no statistical difference.')
        results = pd.DataFrame()

    return summary, results

#%%

"""
Oct 25th
Reference to reconstruct RGB images

import numpy as np
import pandas as pd

path = r'C:/Users/Daisuke/OneDrive - Temasek Life Sciences Laboratory/Experimental_Data/Hyperspectral_Imaging/2020-10-14/509_ColorChecker/capture/509.hdr'

cube = hsi_loading(path)
df = pd.DataFrame()
df['wavelength'] = np.linspace(400, 1000, 204)
bands = np.linspace(400, 1000, 204)

locs_B, img_B = hsi_selection(cube)
locs_G, img_G = hsi_selection(cube)
locs_R, img_R = hsi_selection(cube)

ref_B, normalized_ref = hsi_spectrum(cube, bands, locs_B['xy'])
ref_G, normalized_ref = hsi_spectrum(cube, bands, locs_G['xy'])
ref_R, normalized_ref = hsi_spectrum(cube, bands, locs_R['xy'])

df['B'] = ref_B.mean().values[1:]
df['G'] = ref_G.mean().values[1:]
df['R'] = ref_R.mean().values[1:]

df.to_csv(r'C:/Users/Daisuke/OneDrive - Temasek Life Sciences Laboratory/Experimental_Data/Hyperspectral_Imaging/RGB_reference.csv', index = False)
"""