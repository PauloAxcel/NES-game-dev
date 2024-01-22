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
    #plt.imshow(final)
    #plt.show()

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


def apply_hex(hex_palette_values, matrix_2c02, hex_list):
    data_lines = [line.strip().replace(".byte", "").replace(" ", "").split(",") for line in hex_palette_values]
    hex_values_clean = [[entry.upper() for entry in line] for line in data_lines]
 
    colorv = []
    for h in hex_values_clean:
        for hh in h:    
            for c, h2 in zip(matrix_2c02.reshape(matrix_2c02.shape[0]*matrix_2c02.shape[1],3),hex_list.flatten()):
                if hh == h2:
                    colorv.append(list(c))

    
    final_image = np.uint8(np.array(colorv)).reshape(len(colorv)//4,4,3)
    num_blocks = final_image.shape[0] // 4
    fig, axs = plt.subplots(num_blocks, 1, figsize=(6, 2*num_blocks))
    titles = ["Background", "Sprites"]
    for i in range(num_blocks):
        axs[i].imshow(final_image[i*4:(i+1)*4])
        axs[i].axis('off')  # Turn off axis for cleaner display
        axs[i].set_title(titles[i])

    plt.show()
    
    return final_image



def plot_sprites(raw_data, xsize, ysize, bits, sprites_palettes):

    data_lines = [line.strip().replace(".byte", "").split(",") for line in raw_data]
    flags = [a[2].split('$')[1] for a in  data_lines]

    binary_entries = [bin(int(entry, 16))[2:].zfill(8) for entry in flags]
    flags_list = []
    for binary_entry in binary_entries:
        flags = {
            'vflip': bool(int(binary_entry[7], 2)),
            'hflip': bool(int(binary_entry[6], 2)),
            'priority': bool(int(binary_entry[5], 2)),
            'palette': int(binary_entry[1:3], 2)
        }
        flags_list.append(flags)
    
    palette_entries = [entry['palette'] for entry in flags_list]

    processed_data = [[int(value.split('$')[1], 16) for value in line] for line in data_lines]
    
    sprites = np.uint8(np.zeros((ysize * bits, xsize * bits, 3)))
    
    hb = bits//2
    
    for line, entry in zip(processed_data, palette_entries):
        x = line[-1]
        y = line[0]
        p = sprites_palettes[entry]

        sprites[y:y+hb, x:x+hb] = p[0]
        sprites[y+hb:y+bits, x:x+hb] = p[1]
        sprites[y:y+hb, x+hb:x+bits] = p[2]
        sprites[y+hb:y+bits, x+hb:x+bits] = p[3]
        
        
    plt.imshow(sprites, cmap='gray', interpolation='none')
    plt.title("Sprite Screen location + Palette")
    plt.show()
    
    return sprites, flags_list

def generate_background(initial_nametable, background_addresses, xsize, ysize, bits, attribute_table, background_palettes):
        
    nametable_address = int(initial_nametable.split('$')[1], 16)  # Assuming a default nametable address of $2000
    offsets = [int(address.split('$')[1], 16) - nametable_address for address in background_addresses]

    background = np.uint8(np.zeros((ysize * bits, xsize * bits,3)))
    
    hb = bits//2

    for offset in offsets:
        for i in range(background.shape[0]//bits):
            for j in range(background.shape[1]//bits):
                spot = j + i * background.shape[1]//bits
                if spot == offset:
                    p = background_palettes[attribute_table[i//2,j//2]]
                    background[i*bits:i*bits+hb, j*bits:j*bits+hb] = p[0]
                    background[i*bits+hb:i*bits+bits, j*bits:j*bits+hb] = p[1]
                    background[i*bits:i*bits+hb, j*bits+hb:j*bits+bits] = p[2]
                    background[i*bits+hb:i*bits+bits, j*bits+hb:j*bits+bits] = p[3]
        
    plt.imshow(background, cmap='gray', interpolation='none')
    plt.title("Background Nametable + Attribute table")
    plt.show()
    
    return background

def convert_and_reverse(binary_str):
    binary_chunks = [binary_str[i:i+2] for i in range(0, len(binary_str), 2)]
    decimal_values = [int(chunk, 2) for chunk in binary_chunks[::-1]]
    return decimal_values

def process_attributes(initial_attribute, at, coloring):
    attribute_table = np.zeros((8, 8))

    attribute_address = int(initial_attribute.split('$')[1], 16)  # Assuming a default nametable address of $2000
    offsets = [int(a.split('$')[1], 16) - attribute_address for a in at]

    coloring_without_percent = [color[1:] for color in coloring]

    organized_colors = [convert_and_reverse(binary_str) for binary_str in coloring_without_percent]

    for offset in offsets:
        x_offset = offset // 8
        y_offset = offset % 8
        attribute_table[x_offset:(x_offset + 1), y_offset:(y_offset + 1)] = 1

    # Resize the array
    scaled = np.array(cv2.resize(np.uint8(attribute_table*255.0), (0,0), fx=2, fy=2) > 100) * 1
    soffsets = [a * 2 if 16 * 2 * (a // 8) == 0 else 16 * 2 * (a // 8) for a in offsets]

    for so, c in zip(soffsets, organized_colors):
        for i in range(scaled.shape[0]):
            for j in range(scaled.shape[1]):
                spot = j + i * scaled.shape[0]
                if spot == so:
                    scaled[i, j] = c[0]
                    scaled[i, j+1] = c[1]
                    scaled[i+1, j] = c[2]
                    scaled[i+1, j+1] = c[3]

    #plt.imshow(scaled, cmap='gray', interpolation='none')
    #plt.show()
    
    resized = scaled[:-1,:]
    
    return resized


#test_image_path = r'C:\\assembly\\pythonscripts\\fireball.png'
#locy = 32
#locx = 30
#test = make_template(test_image_path,locx,locy)

image_path = r'C:\assembly\NES-game-dev-main\\nes_2c02_colour_palette.png'
locy = 4
locx = 16
hex_list = np.array([[f"${(i * locx + j):02X}" for j in range(locx)] for i in range(locy)]).reshape(locy,locx)
        
matrix_2c02 = make_template(image_path,locx,locy)

# Apply the template palette to the test image
#new_test, palette = apply_template(test, matrix_2c02, hex_list)
#print(palette)

#output_file = r'output.chr'
#convert_to_hex(new_test, output_file)


hex_palette_values =   [".byte $0f, $12, $23, $27",
                        ".byte $0f, $2b, $3c, $39",
                        ".byte $0f, $0c, $07, $13",
                        ".byte $0f, $19, $09, $29",
                        ".byte $0f, $2d, $10, $15",
                        ".byte $0f, $19, $09, $29",
                        ".byte $0f, $19, $09, $29",
                        ".byte $0f, $19, $09, $29"]

#convert hex values into colour palettes
fpalettes = apply_hex(hex_palette_values, matrix_2c02, hex_list)
background_palettes = fpalettes[:4]
sprites_palettes = fpalettes[4:]


sprite_info = [
    ".byte $70, $05, $00, $80",
    ".byte $70, $06, $00, $88",
    ".byte $78, $07, $00, $80",
    ".byte $78, $08, $00, $88"
]
#shows where the background/sprites are being drawn
xsize=32
ysize=30
bits=8

sprites, flags_list = plot_sprites(sprite_info, xsize, ysize, bits, sprites_palettes)



at = ["$23C2", "$23E0"]
coloring = ["%01000000", "%00001100"]
initial_attribute = "$23C0"
attribute_table = process_attributes(initial_attribute, at, coloring)


addresses = ["$206b", "$2157", "$2223", "$2352"]
addresses2 = ["$2074", "$2143" ,"$215d" ,"$2173", "$222f" ,"$22f7"]
addresses3 = ["$20f1", "$21a8", "$227a", "$2344", "$237c"]

background_addresses = addresses + addresses2 + addresses3

initial_nametable = "$2000"
background = generate_background(initial_nametable, background_addresses, xsize, ysize, bits, attribute_table, background_palettes)



alltogether = sprites + background
plt.imshow(alltogether)
plt.title("Game Screen!")
plt.show()
