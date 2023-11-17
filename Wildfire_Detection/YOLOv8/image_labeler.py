import os
import xml.etree.ElementTree as ET

folder_path = './datasets/annotated_hpwren_small/full/xmls/'  
output_path = './datasets/annotated_hpwren_small/full/labels/'

for filename in os.listdir(folder_path):
    if filename.endswith('.xml'):
        file_path = os.path.join(folder_path, filename)
        
        # Parse the XML file
        tree = ET.parse(file_path)
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

        # Write normalized coordinates to a text file for each XML file
        output_filename = filename.split('.')[-2] + '.txt'

        with open(output_path + output_filename, 'w') as file:
            file.write(f'0 {xmin} {ymin} {xmax} {ymax}')
    