# Experiment Setup

This part of the repo is meant to guide the user through the process of creating several test sets, test them and collect data that eventually could be statiscally analyzed. At this point the user has the data-prep section done and the different privacy awareness cases running, if not please go to the data-prep directory and follow the steps.

## Generating the test sets

We have 275 images available in the custom privacy-awareness dataset, the idea is to evaluate the different base-agent and support-agent combinations PA results with different inputs, in this case, several groups of 100 randomly selected images which we call "test sets". To generate this sets, run the ```sets_generator.sh```, a bash script that creates a txt file with 100 sets of combinations of sorted non-repeated numbers from 1 to 275, and each line in the txt file starts with the set number followed by the numbers combination. Run:
```
# go to the experiment_setup directory
cd experiment_setup/

# run the script to generate the test sets list
./sets_generator.sh

# this will create a text file called test_sets.txt
cat test_sets.txt

# the text file contains the test sets (first number is the set id, the rest represent the selected images), take a look at a couple of them:

1 1 4 6 7 8 9 10 11 12 15 16 20 23 24 25 27 30 31 34 35 39 42 43 51 52 54 61 63 64 65 69 70 71 75 82 86 87 89 91 95 100 101 106 108 111 112 114 121 122 123 128 132 133 135 137 138 140 143 148 149 164 165 168 171 179 181 194 197 198 199 205 207 208 212 213 214 219 220 229 232 237 240 245 247 250 252 254 255 257 260 261 262 264 265 266 268 270 271 274 275

2 1 3 4 6 7 8 10 11 12 13 14 16 19 23 26 30 35 39 41 42 44 45 47 49 51 56 60 66 70 71 82 88 89 91 93 95 98 105 107 108 113 116 122 125 127 130 132 134 136 141 144 145 148 149 158 159 161 163 165 168 169 172 175 177 179 181 184 187 190 192 198 199 200 203 204 206 207 214 216 220 221 223 227 230 232 234 236 237 238 241 244 245 250 255 256 261 265 266 267 273

```

## Backing up the datasets

The scripts to setup the selected sets depend on having a backup of the complete datasets to then restore it and change to another set. To achieve this, manually backup the following files:
```
# from the datasets directory
cp -r datasets/customPA datasets/customPA_bk

# from the root of the repo
cp -r retina-face/data/widerface/val retina-face/data/widerface/val_bk

```

## Install selected test set in YOLO and RetinaFace
In this section we use a couple of scripts to actually setup the datasets according to the generated test set list. For example, for a test set of 100 images, the rest (175 images) must be discarded from the complete dataset in order to run the evaluation on the reduced set.

### YOLO
For the case of YOLO just run:
```
# go to the experiment_setup directory
cd experiment_setup/

# run the script to select and setup a specific test set (i.e. test_set_id = 1)
./select_set_yolo.sh test_sets.txt 1

# at this point the dataset is configured for test set 1, so run the model as usual
cd ../yolo-face/
python3 run_val.py
```

### RetinaFace
For the case of RetinaFace just run:
```
# go to the experiment_setup directory
cd experiment_setup/

# run the script to select and setup a specific test set (i.e. test_set_id = 1)
./select_set_retina.sh test_sets.txt 1

# at this point the dataset is configured for test set 1, so run the model as usual
cd ../retina-face/

rm -r widerface_evaluate/widerface_txt/*
python3 test_widerface.py --trained_model Resnet50_Final.pth --network resnet50 -s
cd widerface_evaluate
python3 evaluation.py
cd ..

# or run everything in a single command

rm -r widerface_evaluate/widerface_txt/* ; python3 test_widerface.py --trained_model Resnet50_Final.pth --network resnet50 -s ; cd widerface_evaluate ; python3 evaluation.py ; cd ..
```