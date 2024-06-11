# -*- coding: utf-8 -*-

def specim_loading(path):
    """
    Load a hyperspectral data cube from the SPECIM IQ image directory.

    Parameters:
    path (str): The absolute path to the directory containing .hdr and .raw files.

    Returns:
    np.ndarray: The calibrated hyperspectral data cube.
    
    Raises:
    ValueError: If the provided path is not a valid directory.
    FileNotFoundError: If the required files are not found in the directory.
    """
    from pathlib import Path
    import spectral as spy
    import numpy as np
    
    # Convert the path to a Path object
    path = Path(path)
    
    # Check if the provided path exists and is a directory
    if not path.exists() or not path.is_dir():
        raise ValueError(f"The path {path} is not a valid directory")
    
    # Extract the directory name as the ID
    ID = path.stem
    print(f"Loading hyperspectral data for ID: {ID}")
 
    try:
        # Open the hyperspectral data files
        data_ref = spy.io.envi.open(str(f'{path}\\capture\\{ID}.hdr'), str(f'{path}\\capture\\{ID}.raw'))
        white_ref = spy.io.envi.open(str(f'{path}\\capture\\WHITEREF_{ID}.hdr'), str(f'{path}\\capture\\WHITEREF_{ID}.raw'))
        dark_ref = spy.io.envi.open(str(f'{path}\\capture\\DARKREF_{ID}.hdr'), str(f'{path}\\capture\\DARKREF_{ID}.raw'))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}")

    # Load the data into numpy arrays
    white = np.array(white_ref.load())
    dark = np.array(dark_ref.load())
    cube = np.array(data_ref.load())

    # Perform calibration by subtracting dark reference and dividing by the white reference
    cube = (cube - dark) / (white - dark)
    
    return cube

#%%
# Reconstruct RGB image from hyperspectral cube
# RGB reference (.csv file) was generated using the ColorChecker sheet. 

def specim_RGB(cube):
    
    import numpy as np

    # R, G, and B reference reflectance values from the Specim wavelength channels 6 to 98
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
def adjust_brightness(img, value = 0):
    """
    Adjust the brightness of an image.
    
    Parameters:
    img (np.ndarray): The input image.
    value (int): The value to adjust brightness by, ranging from -255 to +255.
    
    Returns:
    np.ndarray: The brightness-adjusted image.
    """
    import cv2
    import numpy as np
    
    # Ensure the value is within the range
    value = max(-255, min(255, value))
    
    # Convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Perform brightness adjustment
    if value >= 0:
        v = np.clip(v + value, 0, 255)
    else:
        v = np.clip(v + value, 0, 255)  # value is negative
    
    # Merge the channels back
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    return img

#%%
class CoordinateStore:
    """
    Class to store coordinates selected by the user.
    """
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        """
        Callback function to handle mouse events. Stores points on double left-click.
        
        Parameters:
        event: The type of mouse event.
        x, y: The coordinates of the mouse event.
        flags: Any relevant flags passed by OpenCV.
        param: Any extra parameters supplied by OpenCV.
        """
        import cv2
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # Draw a circle at the selected point
            cv2.circle(img, (x, y), 3, (0, 0, 0), -1)
            # Annotate the point with its index
            cv2.putText(img, str(len(self.points)), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            # Store the point coordinates
            self.points.append([x, y])

#%%        
def plant_selection(cube):
    """
    Select Regions of Interest (ROIs) from a hyperspectral cube and return a DataFrame and an image.
    
    Parameters:
    cube (np.ndarray): Hyperspectral data cube.
    
    Returns:
    pd.DataFrame: DataFrame containing the ROIs and their reflectance spectra.
    np.ndarray: Image with annotated ROIs.
    """
    import cv2
    import pandas as pd
    import numpy as np   
    global img
    
    # Instantiate the CoordinateStore class
    coordinateStore1 = CoordinateStore()
    # Convert the hyperspectral cube to an RGB image
    img = specim_RGB(cube)
    
    # Set up the OpenCV window and mouse callback
    window_title = 'Double-click to select plants, press "q" to finish'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, coordinateStore1.select_point)
    
    while True:
        # Display the image and wait for user interaction
        cv2.imshow(window_title, img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Exit loop if 'q' is pressed
            break

    # Close the OpenCV window
    cv2.destroyAllWindows()
    
    # List to store the reflectance spectra for each ROI
    spectrum_data = []

    for roi in coordinateStore1.points:
        # Calculate the mean spectrum for the 5x5 region around the selected point
        spectrum = np.mean(cube[roi[1]-2:roi[1]+3, roi[0]-2:roi[0]+3, :], axis=(0, 1))
        spectrum_data.append(spectrum)

    # Ensure image is in uint8 format for display purposes
    img = img.astype(np.uint8)
    
    # Create a DataFrame with ROI information
    roi_information = pd.DataFrame({
        'ID': range(len(coordinateStore1.points)),
        'label': [""]*len(coordinateStore1.points),
        'xy': coordinateStore1.points,
        'ROI Reflectance': spectrum_data
    })
    
    return roi_information, img

#%%
def specim_plot(df, path=None, band_range=range(0,204)):
    """
    Plot spectral comparison of different samples with mean reflectance and standard deviation.

    Parameters:
        df (DataFrame): DataFrame containing spectral reflectance data.
        path (str): Path to save the output plot.

    Returns:
        Figure: Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
        
    # Get unique labels from DataFrame index
    label = df.index.unique()

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
    
    specim_wavelength = [specim_wavelength[i] for i in band_range]
    # Create a new figure
    fig = plt.figure()

    # Plot mean reflectance with standard deviation for each sample
    for name in label:
        plt.plot(specim_wavelength, df[df.index == name].mean(axis=0), label=name)
        plt.fill_between(specim_wavelength,
                         df[df.index == name].mean(axis=0) - df[df.index == name].std(axis=0),
                         df[df.index == name].mean(axis=0) + df[df.index == name].std(axis=0),
                         alpha=0.2)

    # Add vertical lines to mark specific wavelengths
    plt.axvline(x=428, color='palegreen', linestyle='--', zorder=-1)
    plt.axvline(x=660, color='palegreen', linestyle='--', zorder=-1)
    plt.axvline(x=546, color='lightpink', linestyle='--', zorder=-1)
    plt.axvline(x=970, color='lightblue', linestyle='--', zorder=-1)

    # Add ptitle, labels and legend
    if path != None:
        plt.title('Mean reflectance ± SD, image %s' % path.split('\\')[-1])
    plt.xlabel('Wavelength[nm]')
    plt.ylabel('Reflectance')
    plt.legend(loc='lower right')
    plt.show()

    # # Save the figure as PDF file if path is provided
    if path:
        fig.savefig(rf'{path}\%s_spectral_comparison.pdf' % path.split('\\')[-1])

    return fig

#%%
def three_areas(mask, x, y, r):
    """
    Select central, paracentral, and periferal areas.

    Parameters:
    mask (np.ndarray): The input mask image (binary or grayscale).
    x (int): The x-coordinate of the circle's center.
    y (int): The y-coordinate of the circle's center.
    r (int): The radius of the outermost circle.
    """
    
    import numpy as np
    import cv2
    
    # The outer circle mask
    outer = np.zeros(mask.shape[:2], np.uint8)  
    cv2.circle(outer, (int(x), int(y)), int(r), 1, thickness=-1) 
    outer = cv2.bitwise_and(mask, outer)

    # The middle circle mask
    middle = np.zeros(mask.shape[:2], np.uint8)  
    cv2.circle(middle, (int(x), int(y)), int(2*r/3), 1, thickness=-1)
    middle = cv2.bitwise_and(mask, middle)

    # The central circle mask
    central = np.zeros(mask.shape[:2], np.uint8)  
    cv2.circle(central, (int(x), int(y)), int(r/3), 1, thickness=-1)
    central = cv2.bitwise_and(mask, central)

    # Remove the middle circle area from the outer circle area
    periferal = cv2.bitwise_and(outer, cv2.bitwise_not(middle))
    # Remove the central circle area from the middle circle area
    paracentral = cv2.bitwise_and(middle, cv2.bitwise_not(central))

    return central, paracentral, periferal

#%%
def plant_masking(cube, threshold=2.2, kn=1):
    import cv2
    import numpy as np
    
    # Convert the hyperspectral cube to RGB image
    img = specim_RGB(cube)

    # Mask white pixels (reflection from a plate) using a brightness threshold
    _, white = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 235, 1, cv2.THRESH_BINARY_INV)

    # A representative leaf reflectance generated with SVD. SPECIM IQ wavelength channels [10:200]
    plant_reference_spectrum = np.array([0.04069669,  0.04079603,  0.04126037,  0.04213693,  0.04273207,
            0.04379434,  0.04518137,  0.04713085,  0.04874695,  0.05029235,
            0.0516203 ,  0.05295029,  0.05294916,  0.05332283,  0.05327264,
            0.05344182,  0.05390424,  0.05425218,  0.0543171 ,  0.05522408,
            0.05585662,  0.05727984,  0.05921757,  0.06182781,  0.06514108,
            0.0688416 ,  0.07287744,  0.07792781,  0.08274431,  0.08712468,
            0.09143544,  0.09475504,  0.09722056,  0.09856757,  0.0991911 ,
            0.09913175,  0.09923179,  0.09917054,  0.09974652,  0.10164138,
            0.10276858,  0.1031924 ,  0.1027126 ,  0.10187102,  0.10124572,
            0.10113701,  0.10179041,  0.10246365,  0.10329962,  0.10449796,
            0.10565817,  0.10722926,  0.10852164,  0.10937907,  0.11002037,
            0.11090264,  0.11098943,  0.11108233,  0.11116369,  0.11111724,
            0.11098415,  0.11097883,  0.11096041,  0.11322217,  0.11394354,
            0.11286716,  0.11229089,  0.11243878,  0.11269937,  0.11351002,
            0.11409797,  0.11430819,  0.1129232 ,  0.11037677,  0.10717013,
            0.10334962,  0.10013911,  0.09814503,  0.09737767,  0.09646666,
            0.09434346,  0.09063417,  0.08613345,  0.08249818,  0.07965832,
            0.07827442,  0.0788685 ,  0.08293296,  0.09276   ,  0.10782823,
            0.12418226,  0.13724419,  0.14330781,  0.14223865,  0.13543469,
            0.12344085,  0.10910331,  0.09239325,  0.07515585,  0.05786373,
            0.04189477,  0.02756335,  0.01521532,  0.00479685, -0.00368386,
           -0.01059016, -0.01556718, -0.01969922, -0.02287875, -0.02538634,
           -0.02725541, -0.02868451, -0.0298151 , -0.03078676, -0.03144858,
           -0.03190495, -0.03239354, -0.03274344, -0.03315384, -0.0335644 ,
           -0.03404963, -0.03421362, -0.03422848, -0.03380797, -0.03416794,
           -0.03555574, -0.03529606, -0.03537379, -0.03648132, -0.03648571,
           -0.03692753, -0.03719202, -0.03735861, -0.037777  , -0.03793817,
           -0.03801186, -0.0384836 , -0.03849779, -0.03870448, -0.03879738,
           -0.03916825, -0.03950635, -0.039951  , -0.04001093, -0.04007773,
           -0.04053165, -0.04100606, -0.04129533, -0.04136556, -0.0415791 ,
           -0.04163111, -0.04166479, -0.04189004, -0.04211083, -0.04238162,
           -0.04291235, -0.04348892, -0.04388314, -0.04394482, -0.04384172,
           -0.04397907, -0.0437822 , -0.04379905, -0.04419531, -0.04406442,
           -0.04378627, -0.04406204, -0.04387016, -0.04338229, -0.04324095,
           -0.04314747, -0.04305496, -0.04254075, -0.04202947, -0.04179743,
           -0.04191031, -0.04088917, -0.0412918 , -0.04073383, -0.04114633,
           -0.04026839, -0.0399152 , -0.04082155, -0.04049295, -0.04036159,
           -0.03976451, -0.0397357 , -0.04025013, -0.0402305 , -0.03912869])
                     
    # Apply the representative leaf reflectance to hyperspectral cube
    svd_pic = np.dot(cube[:, :, 10:200], plant_reference_spectrum)

    # Threshold the SVD image to create a mask
    _, mask = cv2.threshold(svd_pic, threshold, 1, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up the mask
    kernel = np.ones((kn, kn), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = mask * white

    return img, mask, svd_pic

#%%
def data_extraction(cube, df_loc, threshold = 2.2, kn = 1, dist = 15, path = None):  
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
                    central, paracentral, periferal = three_areas(mask, x, y, radius)
                    periferal_cube = cube * cimg[:, :, np.newaxis] * periferal[:, :, np.newaxis]
                    paracentral_cube = cube * cimg[:, :, np.newaxis] * paracentral[:, :, np.newaxis]
                    central_cube = cube * cimg[:, :, np.newaxis] * central[:, :, np.newaxis]

                    # Draw the contour and circle
                    cv2.circle(img, (int(x), int(y)), int(radius), (255, 0, 255), 1)
                    
                    # Calculate leaf area
                    area = np.sum(cimg * mask) / 255
                    plant_cube = cube * cimg[:, :, np.newaxis] * mask[:, :, np.newaxis]

                    # Append data to the list
                    data.append([
                        path.split('\\')[-1], df_loc['ID'][i], df_loc['label'][i], area, radius,
                        np.nansum(plant_cube, axis=(0, 1)) / np.sum(cimg * mask),
                        np.nansum(periferal_cube, axis=(0, 1)) / np.sum(cimg * periferal),
                        np.nansum(paracentral_cube, axis=(0, 1)) / np.sum(cimg * paracentral),
                        np.nansum(central_cube, axis=(0, 1)) / np.sum(cimg * central)
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
    ax2.imshow(np.flip(specim_RGB(cube) * mask[:, :, np.newaxis],2))
    ax2.axis('off')
    ax2.set_title('Masked image %s' % path.split('\\')[-1])
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(np.flip(img, 2))
    ax3.axis('off')
    ax3.set_title('Contour image %s' % path.split('\\')[-1])
    
    fig.set_size_inches(18, 6)
    # # Save the figure as PNG file if path is provided
    if path:
        fig.savefig(path + r'\%s_masked_image.png' % path.split('\\')[-1])
    fig.show()

    # Create DataFrame and flatten data
    masked_cube = cube * mask[:, :, np.newaxis]
    df = pd.DataFrame(data[1:], columns=data[0]).sort_values('ID')
    df_spectrum = pd.concat([
        df.iloc[:, 3:5], 
        pd.DataFrame(df['reflectance'].apply(pd.Series), columns=None), 
        pd.DataFrame(df['peripheral'].apply(pd.Series), columns=None), 
        pd.DataFrame(df['paracentral'].apply(pd.Series), columns=None), 
        pd.DataFrame(df['central'].apply(pd.Series), columns=None)
    ], axis=1)
    
    df_spectrum.index = pd.MultiIndex.from_frame(df.iloc[:, :3])
    df_spectrum.columns = ['leaf area [px]', 'radius [px]'] + [f"whole_{a}" for a in np.arange(204)] + [f"peripheral_{a}" for a in np.arange(204)] + [f"paracentral_{a}" for a in np.arange(204)] + [f"central_{a}" for a in np.arange(204)]

    if path:    
        df_spectrum.to_csv(path + r'\%s_spectrum.csv' % path.split('\\')[-1], index=True)                

    return df_spectrum, img, masked_cube
