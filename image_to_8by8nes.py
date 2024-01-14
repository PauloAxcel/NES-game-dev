from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance
from collections import Counter
import random

def make_template(image_path,locx,locy):
    try:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGRA2RGBA)
    except:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
    ys,xs,_ = image.shape
    allc = []
    avgc = []

    for j in range(locy):
        for i in range(locx):
            allc.append(image[j*ys//locy:(j+1)*ys//locy,i*xs//locx:(i+1)*xs//locx])

    for c in allc: 
        color_counts = {}
        for pixel in c:
            for p in pixel:
                rgb = tuple(p[:3])
                color_counts[rgb] = color_counts.get(rgb, 0) + 1
            most_frequent_color = max(color_counts, key=color_counts.get)
        avgc.append(list(most_frequent_color))
    
    final = np.uint8(avgc).reshape(locy,locx,3)
    plt.imshow(final)
    plt.show()

    return final

def apply_template(test, matrix_2c02, hex_list):
    ys, xs, _ = test.shape
    reshaped_test_image = test.reshape(-1, 3)
    reshaped_matrix = matrix_2c02.reshape(-1, 3)
    distances = distance.cdist(reshaped_test_image, reshaped_matrix)
    closest_color_indices = np.argmin(distances, axis=1)
    checks = Counter(closest_color_indices)

    if len(checks.most_common())>4:
        mask = []
        for cnt, (colour,_) in enumerate(checks.most_common()):
            if cnt>=4:
                mask.append([closest_color_indices==colour])
    
        masked = np.array(list(map(bool,np.sum(mask,axis=0)[0])))        
        closest_color_indices_masked = [a if not b else np.nan for a, b in zip(closest_color_indices, masked)]

        new_test_image = np.array([reshaped_matrix[int(a)] if not np.isnan(a) else [np.nan,np.nan,np.nan] for a in closest_color_indices_masked]).reshape(ys,xs,3)

        nan_indices = np.all(np.isnan(new_test_image), axis=2)
        

        for i in range(nan_indices.shape[0]):
            for j in range(nan_indices.shape[1]):
                if nan_indices[i,j] == True:
                    neighbors = []
                    
                    if j + 1 < nan_indices.shape[1] and not np.isnan(new_test_image[i, j + 1, 0]):
                        neighbors.append(tuple(new_test_image[i, j + 1, :]))
                    if i + 1 < nan_indices.shape[0] and not np.isnan(new_test_image[i + 1, j, 0]):
                        neighbors.append(tuple(new_test_image[i + 1, j, :]))
                    if i - 1 >= 0 and not np.isnan(new_test_image[i - 1, j, 0]):
                        neighbors.append(tuple(new_test_image[i - 1, j, :]))
                    if j - 1 >= 0 and not np.isnan(new_test_image[i, j - 1, 0]):
                        neighbors.append(tuple(new_test_image[i, j - 1, :]))
                            
                    counter = Counter(x for xs in neighbors for x in set(xs)).most_common(3)
                    colour = []
                    for c,num in counter:
                        colour.append([np.any(a==c) for a in neighbors])
                    
                    new_test_image[i,j] = random.choice([c for c,a in zip(neighbors,np.sum(colour,axis=0)==max(np.sum(colour,axis=0))) if a == True])
                else:
                    continue
    else:
        new_test_image = reshaped_matrix[closest_color_indices].reshape(ys, xs, 3)

    palette = []
    for cnt, (index, _) in enumerate(checks.most_common()):
        if cnt < 4:
            palette.append(hex_list.reshape(-1, 1)[index][0])

    final_image = np.uint8(new_test_image)
    plt.imshow(final_image)
    plt.show()

    return final_image, palette

def extend_string_array(input_array, target_shape):
    input_array = np.array(input_array, dtype=str)
    target_rows, target_cols = target_shape
    reshaped_array = input_array.reshape(-1, len(input_array[0]))
    extended_array = np.full((target_rows, target_cols), '0', dtype=str)
    extended_array[:reshaped_array.shape[0], :reshaped_array.shape[1]] = reshaped_array

    return extended_array

def convert_to_hex(new_test, output_file):
        
    unique_colors = np.unique(new_test.reshape(-1, 3), axis=0)
    unique_colors_tuples = [tuple(color) for color in unique_colors]

    color_to_label = {color: i for i, color in enumerate(unique_colors_tuples)}

    label_matrix = np.zeros_like(new_test[:, :, 0], dtype=int)

    for i in range(new_test.shape[0]):
        for j in range(new_test.shape[1]):
            pixel_color = tuple(new_test[i, j, :])
            label_matrix[i, j] = color_to_label.get(pixel_color, -1)  # Use -1 for colors not in the unique_colors


    lowbits = []
    highbits = []
    for a in label_matrix.reshape(-1,1):
        if a == 0:
            lowbits.append("0")
            highbits.append("0")
        if a == 1:
            lowbits.append("1")
            highbits.append("0")
        if a == 2:
            lowbits.append("0")
            highbits.append("1")
        if a == 3:
            lowbits.append("1")
            highbits.append("1")

    lowbits = np.array(lowbits).reshape(label_matrix.shape[0],label_matrix.shape[1])
    highbits = np.array(highbits).reshape(label_matrix.shape[0],label_matrix.shape[1])

    if lowbits.shape[0] == 16 and lowbits.shape[1] == 16:
            lowbits = lowbits.reshape(8,2,16)
    if highbits.shape[0] == 16 and highbits.shape[1] == 16:
            highbits = highbits.reshape(8,2,16)
            
    if  (lowbits.shape[0] == 16 and lowbits.shape[1] == 8):
        lowbits = lowbits.reshape(8,2,8)
    if (highbits.shape[0] == 16 and highbits.shape[1] == 8):
        highbits = highbits.reshape(8,2,8)
        
    if (lowbits.shape[0] == 8 and lowbits.shape[1] == 16):
        lowbits = lowbits.reshape(8,1,16)
    if (highbits.shape[0] == 8 and highbits.shape[1] == 16):
        highbits = highbits.reshape(8,1,16)
        
        # len(lowbits.shape)==3 means that the array has already been previously changed!
    if not len(lowbits.shape)==3 and (lowbits.shape[0] < 8 or lowbits.shape[1] < 8):
            lowbits = extend_string_array(lowbits, (8,8))
    if not len(lowbits.shape)==3 and (highbits.shape[0] < 8 or highbits.shape[1] < 8):
            highbits = extend_string_array(highbits, (8,8))
        
    if not len(lowbits.shape)==3 and (lowbits.shape[0] == 8 and lowbits.shape[1] == 8):
            lowbits = lowbits.reshape(8,1,8)
    if not len(highbits.shape)==3 and (highbits.shape[0] == 8 and highbits.shape[1] == 8):
            highbits = highbits.reshape(8,1,8)

    highhexs = []

    for row in highbits:
        for r in row:    
            hex_value = ''.join([hex(int(bit, 2))[2:].zfill(1) for bit in r])
            highhexs.append(hex_value)

    lowhexs = []

    for row in lowbits:
        for r in row:   
            hex_value = ''.join([hex(int(bit, 2))[2:].zfill(1) for bit in r])
            lowhexs.append(hex_value)
            
    lowhex = []
    highhex = []

    for a in lowhexs:
        if len(a) > 8:
            # Break the string into halves
            half_length = len(a) // 2
            first_half = a[:half_length]
            second_half = a[half_length:]

            # Process each half separately
            lowhex.append([hex(int(first_half, 2)),hex(int(second_half, 2))])
        else:
            # Process the whole string
            lowhex.append(hex(int(a, 2)))

    for a in highhexs:
        if len(a) > 8:
            # Break the string into halves
            half_length = len(a) // 2
            first_half = a[:half_length]
            second_half = a[half_length:]

            # Process each half separately
            highhex.append([hex(int(first_half, 2)),hex(int(second_half, 2))])
        else:
            # Process the whole string
            highhex.append(hex(int(a, 2)))
    
    
    hext = pd.concat([pd.DataFrame(lowhex),pd.DataFrame(highhex)],axis=1)
    
    if hext.shape[0]>8:    
        hext1 = hext.iloc[:hext.shape[0]//2,:]
        hext1 = pd.concat([hext1.iloc[:,:hext1.shape[1]//2],hext1.iloc[:,hext1.shape[1]//2:]])
        hext1 = hext1.melt().iloc[:,1]
    else:
        hext1 = pd.concat([hext.iloc[:,:hext.shape[1]//2],hext.iloc[:,hext.shape[1]//2:]])
        hext1 = hext1.melt().iloc[:,1]
    
    if hext.shape[0]>8:    
        hext2 = hext.iloc[hext.shape[0]//2:,:]
        hext2 = pd.concat([hext2.iloc[:,:hext2.shape[1]//2],hext2.iloc[:,hext2.shape[1]//2:]])
        hext2 = hext2.melt().iloc[:,1]
    else:
        hext2 = pd.concat([hext.iloc[:,:hext.shape[1]//2],hext.iloc[:,hext.shape[1]//2:]])
        hext2 = hext2.melt().iloc[:,1]
    
    if np.all(hext1 == hext2):
        combined_hex = hext1.values
    else:
        combined_hex = pd.concat([hext1 ,hext2], axis=0).values
    
    binary_data = bytes(int(x, 0) for x in combined_hex)

    with open(output_file, 'wb') as f:
        f.write(binary_data)


test_image_path = r'C:\\assembly\\pythonscripts\\fireball.png'
locy = 16
locx = 16
test = make_template(test_image_path,locx,locy)

image_path = r'C:\\assembly\\pythonscripts\\nes_2c02_colour_palette.png'
locy = 4
locx = 16
hex_list = np.array([[f"${(i * locx + j):02X}" for j in range(locx)] for i in range(locy)]).reshape(locy,locx)
        
matrix_2c02 = make_template(image_path,locx,locy)

# Apply the template palette to the test image
new_test, palette = apply_template(test, matrix_2c02, hex_list)
print(palette)

output_file = r'output.chr'
convert_to_hex(new_test, output_file)



