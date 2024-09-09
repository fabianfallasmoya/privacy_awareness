from ultralytics import YOLO

model = YOLO("yolov8n-face.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="/home/jcordero/Downloads/alcaraz.jpg", save=False) # Display preds. Accepts all YOLO predict arguments

model.val(data="widerface_pa.yaml")