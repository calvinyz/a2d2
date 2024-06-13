from ultralytics import YOLO
from os.path import join, abspath, dirname

# Load a model
# model = YOLO(join(dirname(abspath(__file__)), 'yolov8n_wildfire_as_e100.pt'))  # load a pretrained model

if __name__ == '__main__':
    model_name = 'yolov8n.pt'
    model = YOLO(model_name)
    # Train the model
    n_epochs = 200
    data_pathfile = join(dirname(abspath(__file__)), 'wildfire_local_yolov8/data.yaml')

    train_res = model.train(data=data_pathfile, 
                          epochs=n_epochs, 
                          imgsz=640, 
                          batch=8, 
                          name=f'{model_name[:-3]}_wildfire_e{n_epochs}_local')
    val_res = model.val()
