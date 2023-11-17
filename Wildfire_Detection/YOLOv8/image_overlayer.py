from PIL import Image
import os

def overlay_normalized_mask_on_white(mask_path, output_folder, alpha=0.5, image_size=(640, 480)):
    # Create a white image
    white_image = Image.new('RGB', image_size, 'white')

    # Open the mask and resize it to match the image size
    mask = Image.open(mask_path)
    mask = mask.resize(image_size, Image.ANTIALIAS)

    # Blend the mask on top of the white image
    blended = Image.blend(white_image, mask, alpha=alpha)

    # Save the result to the output folder
    output_filename = os.path.basename(mask_path)
    output_path = os.path.join(output_folder, output_filename)
    blended.save(output_path, format='PNG')  # Save as PNG to preserve transparency

def overlay_normalized_masks_on_white_in_folder(mask_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for mask_filename in os.listdir(mask_folder):
        if mask_filename.lower().endswith('.png'):  # Adjust if needed
            mask_path = os.path.join(mask_folder, mask_filename)

            overlay_normalized_mask_on_white(mask_path, output_folder)

image_folder = './datasets/flame_aerial_dataset/images/'
mask_folder = './datasets/flame_aerial_dataset/masks/'
output_folder = './datasets/flame_aerial_dataset/overlayed_images/'

overlay_normalized_masks_on_white_in_folder(mask_folder, output_folder)
