from PIL import Image
import os

def stretch_images(input_folder, output_folder, target_size=(640, 640)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Add other image extensions if needed
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            stretch_image(input_path, output_path, target_size)

def stretch_image(input_path, output_path, target_size):
    with Image.open(input_path) as img:
        stretched_img = img.resize(target_size, Image.ANTIALIAS)
        stretched_img.save(output_path)

input_folder = './datasets/flame_aerial_dataset/resized'
output_folder = './datasets/flame_aerial_dataset/stretched'

stretch_images(input_folder, output_folder, (640, 640))
