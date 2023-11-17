import os

def create_no_smoke_bbox_file(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Add other image extensions if needed
            image_path = os.path.join(image_folder, filename)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_filename)

            with open(txt_path, 'w') as txt_file:
                # Write YOLO format bounding box coordinates for no smoke
                txt_file.write('0 0.0 0.0 1.0 1.0')

image_folder = './datasets/unannotated_hpwren/no_smoke/images'
output_folder = './datasets/unannotated_hpwren/no_smoke/labels'

create_no_smoke_bbox_file(image_folder, output_folder)
