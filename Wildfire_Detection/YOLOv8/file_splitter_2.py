import os
import random
import shutil

def split_data(source_images, source_labels, training_images, testing_images, validation_images,
              training_labels, testing_labels, validation_labels, split_size, file_extensions):
    images = [os.path.splitext(f)[0] for f in os.listdir(source_images) if f.lower().endswith(tuple(file_extensions))]
    
    random.shuffle(images)
    
    training_split = int(len(images) * split_size[0])
    test_split = int(len(images) * (split_size[0] + split_size[1]))
    
    training_images_list = images[:training_split]
    test_images_list = images[training_split:test_split]
    validation_images_list = images[test_split:]
    
    def copy_files(file_list, source_folder, dest_folder, file_extensions):
        for filename in file_list:
            for file_extension in file_extensions:
                source_file = os.path.join(source_folder, filename + file_extension)
                if os.path.exists(source_file):
                    dest_file = os.path.join(dest_folder, filename + file_extension)
                    shutil.copyfile(source_file, dest_file)
                    break  # Break after finding the first matching file extension

    copy_files(training_images_list, source_images, training_images, file_extensions)
    copy_files(test_images_list, source_images, testing_images, file_extensions)
    copy_files(validation_images_list, source_images, validation_images, file_extensions)
    copy_files(training_images_list, source_labels, training_labels, ['.txt'])
    copy_files(test_images_list, source_labels, testing_labels, ['.txt'])
    copy_files(validation_images_list, source_labels, validation_labels, ['.txt'])

# Path to the main folders containing images and labels
source_images_folder = './datasets/smoke_and_no_smoke_hpwren/full/images/'
source_labels_folder = './datasets/smoke_and_no_smoke_hpwren/full/labels/'

# Destination folders for training, testing, and validation data for images and labels
training_images_folder = './datasets/smoke_and_no_smoke_hpwren/training/images/'
testing_images_folder = './datasets/smoke_and_no_smoke_hpwren/testing/images/'
validation_images_folder = './datasets/smoke_and_no_smoke_hpwren/validation/images/'

training_labels_folder = './datasets/smoke_and_no_smoke_hpwren/training/labels/'
testing_labels_folder = './datasets/smoke_and_no_smoke_hpwren/testing/labels/'
validation_labels_folder = './datasets/smoke_and_no_smoke_hpwren/validation/labels/'

# Define the split sizes (80% training, 10% testing, 10% validation)
split_sizes = [0.8, 0.1, 0.1]

# Define the file extensions to consider
file_extensions = ['.jpg', '.jpeg']

# Split and copy the data
split_data(source_images_folder, source_labels_folder, training_images_folder, testing_images_folder,
           validation_images_folder, training_labels_folder, testing_labels_folder,
           validation_labels_folder, split_sizes, file_extensions)
