from ultralytics import YOLO
from ultralytics import globals

model = YOLO("yolov8n-face.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="/home/jcordero/Downloads/alcaraz.jpg", save=False) # Display preds. Accepts all YOLO predict arguments

# Set the eval case global variable
# Case 0: plain yolo using only confidence score
# Case 1: plain yolo using confidence score and bbox size
# Case 3: yolo + depth estimator
# Case 5: yolo + few shot model
globals.eval_case = 3

model.val(data="widerface_pa.yaml")