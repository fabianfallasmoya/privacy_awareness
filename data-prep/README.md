# Data Preparation
In this markdown document we go through the steps of preparing our custom dataset for both yolo-face and for retina-face models. We're taking the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/) validation dataset as the baseline (no train/test, only val). The general idea is to create a reduced WIDERFACE dataset (around 200 images) that contains custom Privacy Awareness annotations for validation purposes.

## Yolo-Face
For this model we created the following workflow (you can skip 1 to 3 and download the dataset from this [link](https://app.roboflow.com/objectdetection-iyfmq/widerface-pa/1)):

1. The first step is to download the validation WIDERFACE dataset. The directory layput should look like this:
```
WIDER_val/
├── images
│   ├── 0--Parade
│   ├── 10--People_Marching
│   ├── 11--Meeting
│   ├── 12--Group
│   ├── 13--Interview
│   ├── 14--Traffic
│   ├── 15--Stock_Market
│   ├── 16--Award_Ceremony
│   ├── 17--Ceremony
│   ├── 18--Concerts
│   ├── 19--Couple
│   ├── 1--Handshaking
...
│   ├── 5--Car_Accident
│   ├── 61--Street_Battle
│   ├── 6--Funeral
│   ├── 7--Cheering
│   ├── 8--Election_Campain
│   └── 9--Press_Conference
└── wider_face_split
```
2. Second, create a copy called WIDER_val_PA, and reduce the number of images by running **reduce_images.sh** in the parent directory. This will just leave 5 images of each event (there are 61 events in WIDERFACE).

3. Now that we have reduced the dataset, it's time to add our Privacy Awareness (PA) annotations, to do this we're using an annotation tool, in this case [Roboflow](https://roboflow.com/). Although this tool doesn't support extra labels other than classes, we can annotate PA as 5 classes and then do some extra pre-processing to ensure that these are not perceived as classes but as a new metric label. To do so:

    * Go to the Settings tab in the Roboflow project and there should be only one existing class called face. Add 5 new classes called za_verylow, zb_low, zc_mid, zd_high, and ze_very_high. This will represent PA levels, the prefix ensures that we get those in alphabetical order which is crucial for our pre-processing scripts.
    * Then go to the Annotate tab and click the Dataset Job and for each image look for existing bounding boxes and for each box, duplicate it and change the class from face to the appropiate PA level.
    * Delete images that contain >30 bounding boxes due to time constraints.
    * After finishing the annotation process, go to Versions and create a new version with the Resize option in 640x640.
    * Finally download the version using the "Download Dataset" button and choose YOLOv8 format.
4. Rename the downloaded version to WIDERFACE_PA_yolov8 and verify it looks like:
```
WIDERFACE_PA_yolov8/
└── valid
    ├── images
    └── labels

```
5. Pre-processing: To ensure that YOLO reads the dataset as expected, we run the **formatPA.py** which convert the annotations from:
```
0 0.346875 0.18203125 0.06875 0.18828125
0 0.715625 0.18046875 0.06875 0.20703125
4 0.346875 0.18203125 0.06875 0.18828125
3 0.715625 0.18046875 0.06875 0.20703125
```
to:
```
0 0.346875 0.18203125 0.06875 0.18828125 4
0 0.715625 0.18046875 0.06875 0.20703125 3
```
Notice that for this example there are two bounding boxes which were duplicated at first with different class labels (first column), then we remove the duplicates but keep the label as an extra metric in the original bboxes.
6. At the root of WIDERFACE_PA_yolov8, create a file called "dataset.yaml" with these contents:
```
path: ../datasets/customPA/WIDERFACE_PA_yolov8
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['face']

roboflow:
  workspace: objectdetection-iyfmq
  project: widerface-pa
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/objectdetection-iyfmq/widerface-pa/dataset/1
```
NOTE: Notice that the "path" variable may change depending on the location of your yolo repository.

Done, the dataset should be ready to be used by YOLO.


## Retina-Face
For this model, we take the previous section's (Yolo dataset setup) artifacts as a baseline and followed the next steps:

1. The Retina Face project expects the input images to be present at a certain location with a directory structure, so copy the contents from WIDERFACE_val_PA to that location and delete the images to leave just the directories structure:
```
cp -r WIDER_val_PA/images/* ../privacy_awareness/retina-face/data/widerface/val/images/

cd privacy_awareness/retina-face/data/widerface/val/images

find . -maxdepth 2 -type f -delete
```
For our specific case, delete the event directory that don't have any images:
```
cd privacy_awareness
rm -r retina-face/data/widerface/val/images/10--People_Marching/
```

2. Then Retina Face expects a "wider_val.txt" that contains a list of input images, the authors provide one for the complete WIDERFACE dataset, which we then can reduce to our custom dataset by running:
```
cd data-prep

./filter_jpg_list.sh ../retina-face/data/widerface/val/wider_val.txt ../../datasets/customPA/WIDERFACE_PA_yolov8/valid/images/
```
NOTE: For some reason, the one that the authors provide is not sorted correctly, it starts with events 24 and 40, so edit it manually to get perfect ascending order.

3. Using that txt, we can now copy the images from the custom dataset to the correct directory structure and the correct name. To do this, use the **copy_images_retina.sh** :
```
sudo apt-get install fzf

./copy_images_retina.sh ../retina-face/data/widerface/val/wider_val.txt ../../datasets/customPA/WIDERFACE_PA_yolov8/valid/images/ ../retina-face/data/widerface/val/images/
```

4. Now the only file missing is the "wider_face_val_bbx_gt.txt" that contains the labels, we can obtain this by running the **copy_labels_retina.sh** and then copy the output file to the data directory.
```
./copy_labels_retina.sh ../retina-face/data/widerface/val/wider_val.txt ../../datasets/customPA/WIDERFACE_PA_yolov8/valid/labels/

mv wider_face_val_bbx_gt.txt ../retina-face/data/widerface/val/
```