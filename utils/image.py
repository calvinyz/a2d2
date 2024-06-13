from PIL import Image

def display_img(img_data, mode='RGB'):
    img = Image.fromarray(img_data, mode)
    if mode == 'BGR':
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
    img.show()

def dispaly_img_1d(img, width, height, mode='RGB'):
    # img = responses[0].image_data_uint8
    img = Image.frombytes('RGB', (width, height), img, 'raw')
    if mode == 'BGR':
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
    img.show()

def save_img(img_data, img_path, mode='RGB'):
    img = Image.fromarray(img_data, mode)
    if mode == 'BGR':
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
    img.save(img_path)
