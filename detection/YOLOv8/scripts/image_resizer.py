from PIL import Image
import os

def resize_images(input_folder, output_folder, target_width, resize_every):

    counter = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Add other image extensions if needed
            counter += 1

            if counter % resize_every == 0:
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                resize_image(input_path, output_path, target_width)

def resize_image(input_path, output_path, target_width):
    with Image.open(input_path) as img:
        width_percent = (target_width / float(img.size[0]))
        target_height = int((float(img.size[1]) * float(width_percent)))
        resized_img = img.resize((target_width, target_height), Image.ANTIALIAS)
        resized_img.save(output_path)

input_folder = './datasets/flame_aerial_dataset/images'
output_folder = './datasets/flame_aerial_dataset/resized'

resize_images(input_folder, output_folder, 640, 10)
