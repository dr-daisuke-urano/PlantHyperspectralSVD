# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 20:59:03 2022
Codes used for publication figures. 

@author: Daisuke
"""
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from glob import glob
from scipy.stats import zscore
from sklearn.preprocessing import scale

#%%
'''
Load all speral files 
The spectral files contain 204 spectal bands from central, paracentral and peripheral regions
'''

df = pd.DataFrame()
for files in glob(r'C:\Users\kshalini\OneDrive - Temasek Life Sciences Laboratory\Auronidin-Mass data\Excel files-SR8\*spectrum_SR8.csv'):
    df = pd.concat([df, pd.read_csv(files)])

# One excel file lacks label column. It's likely 0xFe
df = df.replace(' ', '0xFe')
print(df['label'].values)


#%%
'''
Building of SVD model from all plant images
'''
df_all = pd.concat((df.iloc[:,0:5], df.iloc[:,5:].divide(df.iloc[:,167:173].mean(axis = 1).values, axis = 0)), axis = 1)

whole = df_all.iloc[:, 5+4:5+204]
peripheral = df_all.iloc[:, 209+4:209+204]
paracentral = df_all.iloc[:, 413+4:413+204]
central = df_all.iloc[:, 617+4:617+204]
concat_all = np.concatenate((peripheral, paracentral, central), axis = 0)
print(concat_all.shape)

svd = TruncatedSVD(n_components=6, n_iter=7, random_state=42)
svd.fit(concat_all)
print(svd.singular_values_)
print(svd.explained_variance_ratio_)

'''
Show SVD map of whole plant spectra
'''
svd_values = svd.transform(whole)

for i in [0,2,4]:
    plt.figure(figsize = (4,4))
    sns.scatterplot(np.array(svd_values[:,i]), np.array(svd_values[:,i+1]), hue = df_all['label'])
    plt.xlabel('Dimension %i (%.1f %%)' % (i+1, (100*svd.explained_variance_ratio_[i])))
    plt.ylabel('Dimension %i (%.1f %%)' % (i+2, (100*svd.explained_variance_ratio_[i+1])))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(r'C:\Users\kshalini\OneDrive - Temasek Life Sciences Laboratory\Auronidin-Mass data\SVD_wholeplant_Dim%i_Dim%i.svg' % (i+1, i+2))
    plt.show()

#%%
'''
Build SVD model from selected spectral data
Selected central and peripheral regions of CNT, 0xFe, 0xP and 0xN

I decided to use the SVD model with selected spectra for publication (See Figure 2) 
'''

df_svd = df_all.loc[(df['label'] == '0xFe') | (df['label'] == '0xN') | (df['label'] == '0xP') | (df['label'] == 'CNT')]
label = np.concatenate((df_svd.iloc[:,2]+"_peripheral", df_svd.iloc[:,2]+"_central"), axis = 0)

whole = df_svd.iloc[:, 5+4:5+204]
peripheral = df_svd.iloc[:, 209+4:209+204]
paracentral = df_svd.iloc[:, 413+4:413+204]
central = df_svd.iloc[:, 617+4:617+204]
concat = np.concatenate((peripheral, central), axis = 0)
print(concat.shape)

svd = TruncatedSVD(n_components=6, n_iter=7, random_state=42)
svd.fit(concat)
svd_values = svd.transform(concat)

for i in [0,2,4]:
    plt.figure(figsize = (4,4))
    sns.scatterplot(np.array(svd_values[:,i]), np.array(svd_values[:,i+1]), hue = label)
    plt.xlabel('Dimention %i (%.1f %%)' % (i+1, (100*svd.explained_variance_ratio_[i])))
    plt.ylabel('Dimention %i (%.1f %%)' % (i+2, (100*svd.explained_variance_ratio_[i+1])))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(r'C:\Users\kshalini\OneDrive - Temasek Life Sciences Laboratory\Auronidin-Mass data\SVD_selected_Dim%i_Dim%i.svg' % (i+1, i+2))
    plt.show()


#%%
'''
Box plots of SVD values (Figure 2D)

This box_plot2 function is basically same to the original box_plot function, except for the color palette used. 
'''

def box_plot2 (df, group_by = 'ID', data_column = None):

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    colors = np.repeat(sns.color_palette()[0:4], repeats = 1, axis = 0)
           
    mean_df = df.groupby(group_by, as_index = True).mean()
    df = df.set_index(group_by)    
    print(mean_df[data_column])
    plt.figure()

    ax = sns.boxplot(x= df.index, y= data_column, data=df, color="0.8")
    ax = sns.stripplot(x= df.index, y= data_column, data=df, jitter=True, size = 4, palette=colors)
    ax.set_position([0.25,0.3,0.65,0.6])
    
    plt.xticks(rotation = 90)     
    fig = ax.get_figure()
    fig.set_size_inches(mean_df.count()[0]/2 + 1, 4)
    plt.xlabel(None)

    return fig, ax

for i in [0,1,2,3]:
    df_boxplot = pd.DataFrame(np.dot(np.concatenate((peripheral, paracentral, central), axis = 0), svd.components_[i]), columns= ['SVD %i' % i])
    df_boxplot['ID'] = np.concatenate((df_svd['label'] +'_peripheral', df_svd['label'] +'_paracentral', df_svd['label'] +'_central'))
    fig, _ = box_plot2 (df_boxplot, group_by = 'ID', data_column = 'SVD %i' % i)
    fig.savefig(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\boxplot_SVD%i.svg' % i)


#%%
'''
Line graph of SVD components (Figure 2B)
'''

plt.plot(specim_wavelength[4:], svd.components_[0], label='Dimension 1')
plt.plot(specim_wavelength[4:], svd.components_[1], label='Dimension 2')
plt.plot(specim_wavelength[4:], svd.components_[2], label='Dimension 3')
plt.plot(specim_wavelength[4:], svd.components_[3], label='Dimension 4')
plt.xlabel('Wavelength [nm]')
plt.ylabel('values')
plt.legend()
plt.savefig(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\SVD_selected_components.svg')
np.savetxt(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\SVD_selected_components.csv', svd.components_, delimiter=",")

#%%
'''
Show images using SVD components 1 to 4 (Figure 2C) 
Note: Pixel-wise normalization may be necessary before showing this data. 3x3 bin then normalize?
'''

cubes = np.empty(shape = [512,0,204])
for folder in glob(r'C:\Users\kshalini\OneDrive - Temasek Life Sciences Laboratory\Revision_Exp\Dorsal view\Day-14\Hyperspec1\*'):
#    path = r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\Representative_Hypserspectral_images\323'
    cube = hsi_loading(folder)
    normalized_cube = cube / np.mean(cube[:,:,167-173])
    cubes = np.concatenate((cubes, normalized_cube), axis =1)
    
'''
Masking background white pixels
'''

img = RGB(cubes)

# Mask white pixels (reflection from a plate) using a brightness of 235
_, white = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 235, 1, cv2.THRESH_BINARY_INV) 

# The SVD model was generated from one of Grace's screening plate (Image ID: 272)
model = pd.read_csv(r'C:/Users/kshalini/OneDrive - Temasek Life Sciences Laboratory/tak1-w10-mybpromoter/model_svd.csv').iloc[1, 1:]
svd_pic = np.dot(cubes[:,:,10:200], model)
_, mask = cv2.threshold(svd_pic, 1.5, 1, cv2.THRESH_BINARY_INV) 

kernel = np.ones((5,5), dtype=np.uint8)
#    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
mask = mask * white
masked_cube = cubes * mask[:,:,np.newaxis]
plt.imshow(mask )
plt.axis('off')
plt.show()


#%%
'''
SVD images
'''

for i in [0,1,2,3]:
    svd_pic = np.dot(masked_cube[:,:,4:204], svd.components_[i])
    '''
    show histograms
    '''
    # sns.histplot(np.nonzero(svd_pic.flatten()), label = 'SVD %i' % i)
    # plt.legend()
    # plt.show()

    '''
    SVD images
    '''
    svd_pic[np.nonzero(svd_pic)] += 10
    plt.figure(figsize = (210,30))
    plt.imshow(svd_pic, vmax = np.percentile(svd_pic[np.nonzero(svd_pic)], 99.9), vmin = np.percentile(svd_pic[np.nonzero(svd_pic)], 0.01)*0.95, cmap = 'nipy_spectral') # 'gist_stern' 'nipy_spectral'
    plt.axis('off')
#    plt.colorbar()
    plt.title('SVD %i' %i,size=100)
    plt.savefig(r'C:\Users\kshalini\OneDrive - Temasek Life Sciences Laboratory\Revision_Exp\Dorsal view\Day-14\Hyperspec{}'.format(i))
    plt.show()


#%%
'''
Box plots of SVD data (wrky mutants)


Load all speral files of mutants
The spectral files contain 204 spectal bands from central, paracentral and peripheral regions
'''

df_mutant = pd.DataFrame()
for files in glob(r'C:\Users\kshalini\OneDrive - Temasek Life Sciences Laboratory\Auronidin Mutant experiments\Batch-4(gprotein)\Day14- Excel files\*spectrum_SR8.csv'):
    df_mutant = pd.concat([df_mutant, pd.read_csv(files)])

print(np.unique(df_mutant['label'].values))


#%%
'''
Normalization of data using wavelength bands 167 - 172
'''
df_mutant_all = pd.concat((df_mutant.iloc[:,0:5], df_mutant.iloc[:,5:].divide(df_mutant.iloc[:,167:173].mean(axis = 1).values, axis = 0)), axis = 1)
label = np.concatenate((df_mutant_all['label']+"_whole", df_mutant_all['label']+"_peripheral",df_mutant_all['label']+"_paracentral",df_mutant_all['label']+"_central",), axis = 0)

all_data = pd.DataFrame()
for genotype in ['Tak1', 'gcr1-1', 'gcr1-2', 'gpa1-2', 'gpa1-3', 'gpb1-2', 'gpb1-3', 'gpg1-1', 'gpg1-2', 'w10-10', 'w10-18', 'xlg1-2', 'xlg1-3']:
    mutant_treatment = df_mutant_all[df_mutant_all['label'].str.contains(genotype)]

    for treatment in ['cntl', 'XFe', 'XP', 'XCa', 'XMg']:
        mutant = mutant_treatment[mutant_treatment['label'].str.contains(treatment)]
        whole_mutant = mutant.iloc[:, 5+4:5+204]
        peripheral_mutant = mutant.iloc[:, 209+4:209+204]
        paracentral_mutant = mutant.iloc[:, 413+4:413+204]
        central_mutant = mutant.iloc[:, 617+4:617+204]
        
        for i in [1,2,3]:
            df_boxplot = pd.DataFrame(np.dot(np.concatenate((whole_mutant, peripheral_mutant, paracentral_mutant, central_mutant), axis = 0), svd.components_[i]), columns= ['SVD %i' % i])
            df_boxplot['ID'] = np.concatenate((mutant['label'] +'_whole', mutant['label'] +'_peripheral', mutant['label'] +'_paracentral', mutant['label'] +'_central'))
            # Concatenate the data for each iteration to the all_data DataFrame
            all_data = pd.concat([all_data, df_boxplot], axis=0)
            fig, _ = box_plot2(df_boxplot, group_by = 'ID', data_column = 'SVD %i' % i)
            fig.savefig(r'C:\Users\kshalini\OneDrive - Temasek Life Sciences Laboratory\Auronidin Mutant experiments\Batch-4(gprotein)\Box plot\boxplot_%s_%s_SVD%i.svg' % (genotype, treatment, i))
            csv_file_path = r'C:\Users\kshalini\OneDrive - Temasek Life Sciences Laboratory\Auronidin Mutant experiments\Batch-4(gprotein)\Box plot\all_genotype_treatment_data.csv' 
            all_data.to_csv(csv_file_path, index=False)

#%%
'''
Heatmap (Figure 5A)
'''

mutant_svd = df_mutant_all.loc[:, ['label', 'leaf area [px]']] 

for i in [1,2,3]:
    mutant_svd['peripheral_SVD%i' % i] = np.dot(df_mutant_all.iloc[:, 209+4:209+204], svd.components_[i])
    mutant_svd['paracentral_SVD%i' % i] = np.dot(df_mutant_all.iloc[:, 413+4:413+204], svd.components_[i])
    mutant_svd['central_SVD%i' % i] = np.dot(df_mutant_all.iloc[:, 617+4:617+204], svd.components_[i])

#mutant_svd = pd.DataFrame(scale(mutant_svd.iloc[:,1:]), index = mutant_svd['label'], columns = mutant_svd.columns[1:])
mutant_svd = pd.DataFrame(zscore(mutant_svd.iloc[:,1:]), index = mutant_svd['label'], columns = mutant_svd.columns[1:])
mutant_svd_mean = mutant_svd.groupby(mutant_svd.index).mean()

for treatment in [('cntl', 'Oranges'), ('0XN', 'Purples'), ('0XP', 'Blues'), ('0XFe', 'Greens'), ('5XCa', "Wistia")]:    
    heatmap = mutant_svd_mean[mutant_svd_mean.index.str.contains(treatment[0])]
    hm = sns.clustermap(heatmap, z_score=None, cmap = treatment[1], col_cluster= False, method = 'average', dendrogram_ratio=0.1, figsize=(4,4)
                        , cbar_pos=(0.7, 0, 0.035, 0.35))
    hm.savefig(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\mutant_boxplots\heatmap_%s.svg' % treatment[0])
    plt.show()


#%%
'''
Agricultural index values
'''

overview = pd.DataFrame()
for files in glob(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\spectral_Excel_Files\Excel files-SR8 - CaFilesFiexed\*overview_SR8.csv'):
#for files in glob(r'C:\Users\Daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\Excel files-SR8\*overview_SR8.csv'):
    overview = pd.concat([overview, pd.read_csv(files)])
index_df = pd.DataFrame(list(overview["Unnamed: 0"].str.split('_')), columns = (['label', 'area']))
index_df = index_df.replace(' ', '0xFe')
overview.drop(columns = "Unnamed: 0", inplace = True)
overview.insert(0, column = 'label', value = index_df['label'].values)
overview.insert(0, column = 'area', value = index_df['area'].values)
overview.set_index(['label','area'], inplace = True)

ref = overview[overview.index.get_level_values('area') == 'peripheral']
ref.index = ref.index.get_level_values('label')
ref = ref.iloc[:, :-4].apply(zscore)

'''
heatmap of agricultural index values. 
Shalini made a similar heatmap by herself, so this won't be used for publicaiton. 
'''

hm = sns.clustermap(ref, cmap = 'nipy_spectral', col_cluster= False, method = 'average')


#%% 
'''
Spectra graphs (Figure 1)
'''

for condition in df['label'].unique():
    df2 = df.loc[df['label'] == condition]
#    df2.iloc[:,5:] = df2.iloc[:,5:].divide(df2.iloc[:, 167:173].mean(axis = 1).values, axis = 0)
    spectral_df = pd.DataFrame(np.concatenate((df2.iloc[:,5+4:209], df2.iloc[:,209+4:413], df2.iloc[:,413+4:617],df2.iloc[:,617+4:821]), axis = 0))
    spectral_df.columns = specim_wavelength[4:] #np.linspace(400, 1000, 204)
    
    name = []
    for area in ['whole','peripheral','paracentral','central']:
        for label in df2['label']:
            name.append(label + '_' + area)
    
    spectral_df.index = name
    graph = spectral_comparison_SR(spectral_df, bands = specim_wavelength[4:])
    graph.savefig(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\spectral_graphs\spectrum_%s.svg' % condition)
    df_ov = spectral_overview_GT(spectral_df)
    df_ov.to_csv(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\agricultural_ind\agricultural_index_%s.csv' %condition)
    df_ov['ID'] = df_ov.index
    mARI, _ = box_plot (df_ov, group_by = 'ID', data_column = 'mARI')
    mARI.savefig(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\agricultural_ind\mARI_%s.svg' %condition)
    LCI, _ = box_plot (df_ov, group_by = 'ID', data_column = 'LCI(chl)')
    LCI.savefig(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\agricultural_ind\LCI_%s.svg' %condition)
    CARI, _ = box_plot (df_ov, group_by = 'ID', data_column = 'CARI')
    CARI.savefig(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\agricultural_ind\CARI_%s.svg' %condition)
    WI, _ = box_plot (df_ov, group_by = 'ID', data_column = 'WI(R970-R880 R970+R880 ratio)')    
    WI.savefig(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\agricultural_ind\WI_%s.svg' %condition)
       

#%%
'''
Boxplots for leaf area and radius (Figure 1)
'''
area, _ = box_plot(df, group_by = 'label', data_column = 'leaf area [px]')
radius, _ = box_plot(df, group_by = 'label', data_column = 'radius [px]')
area.savefig(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\agricultural_ind\leaf_area.svg')
radius.savefig(r'C:\Users\daisuke\OneDrive - Temasek Life Sciences Laboratory\Manuscript\SR_Auronidin\spectrum_graphs\agricultural_ind\radius_area.svg')

#%% to get svd values 
svd1_data = pd.DataFrame()
svd2_data = pd.DataFrame()
svd3_data = pd.DataFrame()
for genotype in ['Tak1', 'gcr1-1', 'gcr1-2', 'gpa1-2', 'gpa1-3', 'gpb1-2', 'gpb1-3', 'gpg1-1', 'gpg1-2', 'w10-10', 'w10-18', 'xlg1-2', 'xlg1-3']:
    mutant_treatment = df_mutant_all[df_mutant_all['label'].str.contains(genotype)]

    for treatment in ['cntl', 'XFe', 'XP', 'XCa', 'XMg']:
        mutant = mutant_treatment[mutant_treatment['label'].str.contains(treatment)]
        whole_mutant = mutant.iloc[:, 5+4:5+204]
        peripheral_mutant = mutant.iloc[:, 209+4:209+204]
        paracentral_mutant = mutant.iloc[:, 413+4:413+204]
        central_mutant = mutant.iloc[:, 617+4:617+204]

        for i in [1, 2, 3]:
            df_boxplot = pd.DataFrame(np.dot(np.concatenate((whole_mutant, peripheral_mutant, paracentral_mutant, central_mutant), axis=0), svd.components_[i]), columns=['SVD %i' % i])
            df_boxplot['ID'] = np.concatenate((mutant['label'] + '_whole', mutant['label'] + '_peripheral', mutant['label'] + '_paracentral', mutant['label'] + '_central'))

            if i == 1:
                svd1_data = pd.concat([svd1_data, df_boxplot], axis=0)
            elif i == 2:
                svd2_data = pd.concat([svd2_data, df_boxplot], axis=0)
            elif i == 3:
                svd3_data = pd.concat([svd3_data, df_boxplot], axis=0)

            # Save individual box plots (optional)
            fig, _ = box_plot2(df_boxplot, group_by='ID', data_column='SVD %i' % i)
            fig.savefig(f'boxplot_{genotype}_{treatment}_SVD{i}.svg')

# Save the SVD1, SVD2, and SVD3 data as separate Excel files
svd1_data.to_excel('SVD1_data.xlsx', index=False)
svd2_data.to_excel('SVD2_data.xlsx', index=False)
svd3_data.to_excel('SVD3_data.xlsx', index=False)   