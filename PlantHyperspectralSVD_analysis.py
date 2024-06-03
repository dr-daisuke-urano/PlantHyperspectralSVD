# -*- coding: utf-8 -*-
"""
Analysis of Shalini's hyperspectral data
Use threshold of 1 for control and Ca def pics, while of 5 for nitrate def
'Cntl_a1-1','Cntl_b1-25','Cntl_a1-8','Cntl_Tak1','Cntl_a1-13'
'0XP_a1-1','0XP_b1-25','0XP_a1-8','0XP_Tak1','0XP_a1-13'
0XN_a1-1','0XN_b1-25','0XN_a1-8','0XN_Tak1','0XN_a1-13
'5XCa_a1-1','5XCa_b1-25','5XCa_a1-8','5XCa_Tak1','5XCa_a1-13'
'0XFe_a1-1','0XFe_b1-25','0XFe_a1-8','0XFe_Tak1','0XFe_a1-13'
0XMg_a1-1','0XMg_1-25','0XMg_a1-8','0XMg_Tak1','0XMg_a1-13
'Cntl_b1-29','Cntl_wrky6-8','Cntl_b1-70','Cntl_Tak1','Cntl_wrky6-9'
Cntl_Tak1','Cntl_wrky10','Cntl_Tak1','Cntl_wrky10
'0XP_Tak1','0XP_wrky10','0XP_Tak1','0XP_wrky10
0XN_Tak1','0XN_wrky10','0XN_Tak1','0XN_wrky10
@author: daisuke
"""


#%% 
from hyperspec_SR8_ZY import *
import cv2
import pandas as pd
import numpy as np

#%%
path = r'C:\Users\kshalini\OneDrive - Temasek Life Sciences Laboratory\Revision_Exp\14D-oldplants\Day-14\Hyperspec\6116'
cube = hsi_loading(path)
loc, _ = hsi_selection(cube) #Press Q after selection

#%%

loc['label'] = ['0XN_Tak1','0XN_wrky10','0XN_Tak1','0XN_wrky10'] 


#loc['label'] = ['a', 'b', 'c', 'd', 'e']   ##Labelling of the Plants  #047 

#%%
df, img, masked = data_extraction_SR(cube, loc, threshold =1, path = path) #Masking 



#%%
spectral_df = pd.DataFrame(np.concatenate((df.iloc[:,2:206], df.iloc[:,206:410], df.iloc[:,410:614],df.iloc[:,614:820]), axis = 0))
spectral_df.columns = np.linspace(400, 1000, 204)

name = []
for area in ['whole','peripheral','paracentral','central']:
    for label in loc['label']:
        name.append(label + '_' + area)

spectral_df.index = name
spectral_comparison_SR(spectral_df, path)
df_ov = spectral_overview_SR(spectral_df, path) 