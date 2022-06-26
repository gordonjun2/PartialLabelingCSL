from cmath import nan
import splitfolders
import pandas as pd
import os
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)

# # Examine duplicated data (to edit/remove manually)
# df = pd.read_csv('attributes_original.csv')

# print('\nPlease edit/remove the duplicated data below manually...\n')
# print(df[df['filename'].duplicated(keep=False)])

# Refer to https://github.com/jfilter/split-folders on how the folders should be set up (If the images are not separated by classes, then just simply 
# put them into another folder. Eg. ./image_original/image/<images here>)

splitfolders.ratio('./images_cleaned', output="./images_split", seed=1337, ratio=(0.8, 0.1,0.1))        # Split the dataset into train-val-test

# Retrieve all available training images as a list
train_images_list = os.listdir('./images_split/train/images/')                              # Get training images as list first

# Create training dataframe
df = pd.read_csv('attributes_cleaned.csv')                                                  

print('Original training dataframe.\n')
print(df)

# Remove all non-training and missing images
df = df[df['filename'].isin(train_images_list)].reset_index(drop=True)                      # Cleans up the df according to what is actually available in training image folder

print('Dataframe with non-training and missing data removed.\n')
print(df)

# Check distribution of data
print('\nDefault training distribution of Neck type:')
print(df['neck'].value_counts(dropna=False).sort_index())
print('\nDefault training distribution of Sleeve Length type:')
print(df['sleeve_length'].value_counts(dropna=False).sort_index())
print('\nDefault training distribution of Pattern type:')
print(df['pattern'].value_counts(dropna=False).sort_index())

# Data augmentation

# The idea is to sum the weighting for each data. E.g. data 1 has Neck 4, Sleeve Length 2, and Pattern N/A, so the total sum is 
# 7.37272727 + 9.44827586 + 0 = 16.82100313. Data augmentation multiplier has 3 category: If sum >= 20 and no more than 2 common classes 
# are present in the other attributes, x40 per image. If sum < 20 and sum >= 15 and no more than 2 common classes are present in the 
# other attributes, x30 per image. If sum >= 5 and no more than 2 common classes are present in the other attributes, x20 per image. 
# Else, no augmentation for that image.

neck_count_list = []
sleeve_len_count_list = []
pattern_count_list = []

for attr in ['neck', 'sleeve_length', 'pattern']:
    for i in range(len(df[attr].value_counts(dropna=False))):
        if i != len(df[attr].value_counts(dropna=False))-1:
            if attr == 'neck':
                neck_count_list.append(df[attr].value_counts()[i])
            elif attr == 'sleeve_length':
                sleeve_len_count_list.append(df[attr].value_counts()[i])
            else:
                pattern_count_list.append(df[attr].value_counts()[i])
        else:
            if attr == 'neck':
                neck_count_list.append(df[attr].isna().sum())
            elif attr == 'sleeve_length':
                sleeve_len_count_list.append(df[attr].isna().sum())
            else:
                pattern_count_list.append(df[attr].isna().sum())

neck_count_array = np.array(neck_count_list)
sleeve_len_count_array = np.array(sleeve_len_count_list)
pattern_count_array = np.array(pattern_count_list)

# Undersample common data (HAVENT TEST)

# for pattern_type, aug_multiplier in zip(pattern_type_list, aug_multiplier_list):
#     if aug_multiplier == 0:
#         resampled_df = df[np.int8(df['pattern'])==pattern_type].sample(n=num_data_per_pattern)
#         df = df[np.int8(df['pattern']) != pattern_type]
#         df = df.append(resampled_df, ignore_index=True)

neck_aug_weight_array = np.append(np.max(neck_count_array[:-1])/neck_count_array[:-1], 0)
sleeve_len_aug_weight_array = np.append(np.max(sleeve_len_count_array[:-1])/sleeve_len_count_array[:-1], 0)
pattern_count_aug_weight_array = np.append(np.max(pattern_count_array[:-1])/pattern_count_array[:-1], 0)

print('\nAugmentation weightage for attributes:')             
print('Neck:', neck_aug_weight_array)
print('Sleeve Length:', sleeve_len_aug_weight_array)
print('Pattern:', pattern_count_aug_weight_array)

syn_df = pd.DataFrame(columns=['filename', 'neck', 'sleeve_length', 'pattern'])

print('\nInitialize new synthetic dataframe to contain augmented data.\n')
# print(syn_df)

if not os.path.isdir('./images_augmented/train/images/'):
    os.makedirs('./images_augmented/train/images/')

# Declare an augmentation pipeline

transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, p=0.5),
    A.HorizontalFlip(p=0.75),
    A.ChannelShuffle(p=0.5),
    A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
    A.Perspective(p=0.75),
    A.RandomToneCurve(scale=0.2, p=0.5)
])

for row_count in range(df.shape[0]):

    #print(df['filename'][row_count])

    # Read an image with OpenCV
    image = cv2.imread('./images_split/train/images/' + df['filename'][row_count])
    cv2.imwrite('./images_augmented/train/images/' + df['filename'][row_count], image)              

    try:
        neck_index = int(df['neck'][row_count])
    except:
        neck_index = -1

    try:
        sleeve_len_index = int(df['sleeve_length'][row_count])
    except:
        sleeve_len_index = -1

    try:
        pattern_index = int(df['pattern'][row_count])
    except:
        pattern_index = -1

    weight_sum = neck_aug_weight_array[neck_index] + sleeve_len_aug_weight_array[sleeve_len_index] + pattern_count_aug_weight_array[pattern_index]
    #print(weight_sum)

    if weight_sum >= 20 and ((pattern_count_aug_weight_array[pattern_index] != 1 and sleeve_len_aug_weight_array[sleeve_len_index] != 1) or (pattern_count_aug_weight_array[pattern_index] != 1 and neck_aug_weight_array[neck_index] != 1) or (neck_aug_weight_array[neck_index] != 1 and sleeve_len_aug_weight_array[sleeve_len_index] != 1)):
        aug_multiplier = 40         # 60
    elif weight_sum >= 15  and ((pattern_count_aug_weight_array[pattern_index] != 1 and sleeve_len_aug_weight_array[sleeve_len_index] != 1) or (pattern_count_aug_weight_array[pattern_index] != 1 and neck_aug_weight_array[neck_index] != 1) or (neck_aug_weight_array[neck_index] != 1 and sleeve_len_aug_weight_array[sleeve_len_index] != 1)):
        aug_multiplier = 30         # 50
    elif weight_sum >= 5  and ((pattern_count_aug_weight_array[pattern_index] != 1 and sleeve_len_aug_weight_array[sleeve_len_index] != 1) or (pattern_count_aug_weight_array[pattern_index] != 1 and neck_aug_weight_array[neck_index] != 1) or (neck_aug_weight_array[neck_index] != 1 and sleeve_len_aug_weight_array[sleeve_len_index] != 1)):
        aug_multiplier = 20         # 40
    else:
        continue

    # # Convert it to the RGB colorspace

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for j in range(1, aug_multiplier):

        # Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]

        syn_filename = df['filename'][row_count].split('.jpg')[0] + '_aug_' + str(j) + '.jpg'
        cv2.imwrite('./images_augmented/train/images/' + syn_filename, transformed_image)

        new_row = {'filename':syn_filename, 'neck':df['neck'][row_count], 'sleeve_length':df['sleeve_length'][row_count], 'pattern':df['pattern'][row_count]}
        #append row to the dataframe
        syn_df = syn_df.append(new_row, ignore_index=True)

df = df.append(syn_df, ignore_index=True).reset_index(drop=True)

print('Dataframe after data augmentation.\n')
print(df)

# Check distribution of training data
print('\nAugmented training distribution of Neck type:')
print(df['neck'].value_counts(dropna=False).sort_index())
print('\nAugmented training distribution of Sleeve Length type:')
print(df['sleeve_length'].value_counts(dropna=False).sort_index())
print('\nAugmented training distribution of Pattern type:')
print(df['pattern'].value_counts(dropna=False).sort_index())

neck_count_list = []
sleeve_len_count_list = []
pattern_count_list = []

for attr in ['neck', 'sleeve_length', 'pattern']:
    for i in range(len(df[attr].value_counts(dropna=False))):
        if i != len(df[attr].value_counts(dropna=False))-1:
            if attr == 'neck':
                neck_count_list.append(df[attr].value_counts()[i])
            elif attr == 'sleeve_length':
                sleeve_len_count_list.append(df[attr].value_counts()[i])
            else:
                pattern_count_list.append(df[attr].value_counts()[i])
        else:
            if attr == 'neck':
                neck_count_list.append(df[attr].isna().sum())
            elif attr == 'sleeve_length':
                sleeve_len_count_list.append(df[attr].isna().sum())
            else:
                pattern_count_list.append(df[attr].isna().sum())

train_neck_count_array = np.array(neck_count_list)
train_sleeve_len_count_array = np.array(sleeve_len_count_list)
train_pattern_count_array = np.array(pattern_count_list)

neck_df = pd.DataFrame(columns=['neck type', 'count before augmentation', 'count after augmentation', 'percentage before augmentation (%)', 'percentage after augmentation (%)'])
sleeve_len_df = pd.DataFrame(columns=['sleeve length type', 'count before augmentation', 'count after augmentation', 'percentage before augmentation (%)', 'percentage after augmentation (%)'])
pattern_df = pd.DataFrame(columns=['pattern type', 'count before augmentation', 'count after augmentation', 'percentage before augmentation (%)', 'percentage after augmentation (%)'])

for i in range(len(neck_count_array)):
    if i != len(neck_count_array)-1:
        new_row = {'neck type': i, 'count before augmentation': neck_count_array[i], 'count after augmentation': train_neck_count_array[i], 'percentage before augmentation (%)': neck_count_array[i]*100/np.sum(neck_count_array), 'percentage after augmentation (%)': train_neck_count_array[i]*100/np.sum(train_neck_count_array)}
    else:
        new_row = {'neck type': nan, 'count before augmentation': neck_count_array[-1], 'count after augmentation': train_neck_count_array[-1], 'percentage before augmentation (%)': neck_count_array[-1]*100/np.sum(neck_count_array), 'percentage after augmentation (%)': train_neck_count_array[-1]*100/np.sum(train_neck_count_array)}

    neck_df = neck_df.append(new_row, ignore_index=True)

for i in range(len(sleeve_len_count_array)):
    if i != len(sleeve_len_count_array)-1:
        new_row = {'sleeve length type': i, 'count before augmentation': sleeve_len_count_array[i], 'count after augmentation': train_sleeve_len_count_array[i], 'percentage before augmentation (%)': sleeve_len_count_array[i]*100/np.sum(sleeve_len_count_array), 'percentage after augmentation (%)': train_sleeve_len_count_array[i]*100/np.sum(train_sleeve_len_count_array)}
    else:
        new_row = {'sleeve length type': nan, 'count before augmentation': sleeve_len_count_array[-1], 'count after augmentation': train_sleeve_len_count_array[-1], 'percentage before augmentation (%)': sleeve_len_count_array[-1]*100/np.sum(sleeve_len_count_array), 'percentage after augmentation (%)': train_sleeve_len_count_array[-1]*100/np.sum(train_sleeve_len_count_array)}

    sleeve_len_df = sleeve_len_df.append(new_row, ignore_index=True)

for i in range(len(pattern_count_array)):
    if i != len(pattern_count_array)-1:
        new_row = {'pattern type': i, 'count before augmentation': pattern_count_array[i], 'count after augmentation': train_pattern_count_array[i], 'percentage before augmentation (%)': pattern_count_array[i]*100/np.sum(pattern_count_array), 'percentage after augmentation (%)': train_pattern_count_array[i]*100/np.sum(train_pattern_count_array)}
    else:
        new_row = {'pattern type': nan, 'count before augmentation': pattern_count_array[-1], 'count after augmentation': train_pattern_count_array[-1], 'percentage before augmentation (%)': pattern_count_array[-1]*100/np.sum(pattern_count_array), 'percentage after augmentation (%)': train_pattern_count_array[-1]*100/np.sum(train_pattern_count_array)}

    pattern_df = pattern_df.append(new_row, ignore_index=True)

# Check change of distribution of training data
print('\nChange of distribution of Neck type:')
print(neck_df)
print('\nChange of distribution of Sleeve Length type:')
print(sleeve_len_df)
print('\nChange of distribution of Pattern type:')
print(pattern_df)

df.to_csv('attributes_train.csv', index=False, na_rep='#N/A')                       # Save final training annotation file

# Create validation dataframe
df = pd.read_csv('attributes_cleaned.csv')                                         

# Retrieve all available training images as a list
val_images_list = os.listdir('./images_split/val/images/')                           # Cleans up the df according to what is actually available in validation image folder

# Remove all non-training and missing images
df = df[df['filename'].isin(val_images_list)].reset_index(drop=True)

print('\nDataframe with non-validation and missing data removed.\n')
print(df)

# Check distribution of data
print('\nValidation distribution of Neck type:')
print(df['neck'].value_counts(dropna=False).sort_index())
print('\nValidation distribution of Sleeve Length type:')
print(df['sleeve_length'].value_counts(dropna=False).sort_index())
print('\nValidation distribution of Pattern type:')
print(df['pattern'].value_counts(dropna=False).sort_index())

df.to_csv('attributes_val.csv', index=False, na_rep='#N/A')                         # Save final validation annotation file



