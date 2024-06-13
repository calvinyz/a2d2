import os
import random
import shutil

def split_data(source_images, source_labels, training_images, testing_images, validation_images,
              training_labels, testing_labels, validation_labels, split_size):
    images = [f[:-5] for f in os.listdir(source_images) if f.endswith('.jpeg')]
    # labels = [f for f in os.listdir(source_labels) if f.endswith('.txt')]
    
    random.shuffle(images)
    
    training_split = int(len(images) * split_size[0])
    test_split = int(len(images) * (split_size[0] + split_size[1]))
    
    training_images_list = images[:training_split]
    test_images_list = images[training_split:test_split]
    validation_images_list = images[test_split:]
    
    def copy_files(file_list, source_folder, dest_folder, file_type):
        for filename in file_list:
            source_file = os.path.join(source_folder, filename + file_type)
            dest_file = os.path.join(dest_folder, filename + file_type)
            shutil.copyfile(source_file, dest_file)
    
    copy_files(training_images_list, source_images, training_images, '.jpeg')
    copy_files(test_images_list, source_images, testing_images, '.jpeg')
    copy_files(validation_images_list, source_images, validation_images, '.jpeg')
    copy_files(training_images_list, source_labels, training_labels, '.txt')
    copy_files(test_images_list, source_labels, testing_labels, '.txt')
    copy_files(validation_images_list, source_labels, validation_labels, '.txt')

# Path to the main folders containing images and labels
source_images_folder = './datasets/annotated_hpwren_small/full/images/'
source_labels_folder = './datasets/annotated_hpwren_small/full/labels/'

# Destination folders for training, testing, and validation data for images and labels
training_images_folder = './datasets/annotated_hpwren_small/training/images/'
testing_images_folder = './datasets/annotated_hpwren_small/testing/images/'
validation_images_folder = './datasets/annotated_hpwren_small/validation/images/'

training_labels_folder = './datasets/annotated_hpwren_small/training/labels/'
testing_labels_folder = './datasets/annotated_hpwren_small/testing/labels/'
validation_labels_folder = './datasets/annotated_hpwren_small/validation/labels/'

# Define the split sizes (80% training, 10% testing, 10% validation)
split_sizes = [0.8, 0.1, 0.1]

# Split and copy the data
split_data(source_images_folder, source_labels_folder, training_images_folder, testing_images_folder,
           validation_images_folder, training_labels_folder, testing_labels_folder,
           validation_labels_folder, split_sizes)
