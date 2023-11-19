import os
import xml.etree.ElementTree as ET
import cv2

xml_folder = './datasets/annotated_hpwren_large/full/annotations/xmls'
image_folder = './datasets/annotated_hpwren_large/full/images'
output_folder = './datasets/annotated_hpwren_large/full/annotated/'

for xml_filename in os.listdir(xml_folder):
    if xml_filename.endswith('.xml'):
        xml_path = os.path.join(xml_folder, xml_filename)
        
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find image size
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)

        # Find bounding box coordinates and extract values
        coordinates = root.find('.//bndbox')
        xmin = float(coordinates.find('xmin').text) / width
        ymin = float(coordinates.find('ymin').text) / height
        xmax = float(coordinates.find('xmax').text) / width
        ymax = float(coordinates.find('ymax').text) / height

        # Load the corresponding image
        image_filename = xml_filename.split('.')[0] + '.jpeg'
        image_path = os.path.join(image_folder, image_filename)

        if os.path.exists(image_path):
            image = cv2.imread(image_path)

            if image is not None:
                # Draw bounding box on the image in blue
                xmin_pixel = int(xmin * width)
                ymin_pixel = int(ymin * height)
                xmax_pixel = int(xmax * width)
                ymax_pixel = int(ymax * height)

                cv2.rectangle(image, (xmin_pixel, ymin_pixel), (xmax_pixel, ymax_pixel), (255, 0, 0), 2)  # Blue color

                # Save annotated image to the output folder
                output_image_filename = xml_filename.split('.')[0] + '_annotated.jpeg'
                output_image_path = os.path.join(output_folder, output_image_filename)
                cv2.imwrite(output_image_path, image)
            else:
                print(f"Error: Unable to load image {image_path}")
        else:
            print(f"Error: Image file not found at {image_path}")
