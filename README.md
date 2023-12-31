# Object Detection project: Apples :apple:


## Overview

This script serves as an object detector based on the latest version of YOLO, specifically YOLOv5. 
You can find the YOLO repository [here](https://github.com/ultralytics/yolov5) 
and the official YOLO documentation [here](https://docs.ultralytics.com/yolov5/).

The default configuration is tailored for apple detection, yet it's easily adaptable to any subset of YOLO detection classes. 
The script outputs a JSON file containing the positions of detected objects. 
The information is organized by image and class, providing a structure like the following:

```
{
  "IMG_6081.JPG": {
    "path": "data/test/IMG_6081.JPG",
    "position": {
      "apples": [[x1, y1], [x2, y2], ...],
      "car": [...]
    }
  },
  // ...
}
```

Additionally, in the same folder, the script saves input images with bounding boxes drawn around detected objects. 
Each box is labeled with the corresponding class and confidence score.

## Table of Contents

- [Installation](#Installation)
- [Usage](#Usage)
  - [Examples](#Examples)
- [Configuration](#configuration)
  - [YOLO Model Initialization Parameters](#YOLO-Model-Initialization-Parameters)
  - [Model Prediction Parameters](#Model-Prediction-Parameters)


## Installation
Clone the repository and install the requirements.

```bash
# Clone the repository
git clone https://github.com/DavideSangiorgi/apple_detection.git
# or
git clone git@github.com:DavideSangiorgi/apple_detection.git

# Change directory
cd apple_detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

To run the object detection script, use the following command:

```bash
python src/object_detection.py [--config CONFIG_PATH]
```

- `--config`: Path to the configuration JSON file. (Default: 'configs/default.json')

Note: The `--config` argument is optional, and if not provided, 
the script will use the default configuration specified in 'configs/default.json'.

### Examples

1. Run object detection with the default configuration:

```bash
python src/object_detection.py
```

2. Run object detection with a custom configuration:

```bash
python src/object_detection.py --config configs/custom.json
```

## Configuration

This section outlines the configurable parameters in the `configs` folder. Adjust these parameters according to your requirements.

### YOLO Model Initialization Parameters

- **`YOLO_model`**: Specifies the YOLO model to use for detection. 
Options, from smallest/less performant to heavier/most performant: "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt", "yolov5n6.pt", "yolov5s6.pt", "yolov5m6.pt", "yolov5l6.pt", "yolov5x6.pt". 
Refer to the [YOLO documentation](https://docs.ultralytics.com/yolov5/) for details.

- **`device`**: Sets the device to run the model. 
Options: "cpu" or "cuda:n" (replace n with the index of the available CUDA device). 
Check CUDA device availability with `python3 -c "import torch; print(torch.cuda.is_available())"` and the number of available devices with `python3 -c "import torch; print(torch.cuda.device_count())"`.

### Model Prediction Parameters

- **`confidence_threshold`**: Float between 0 and 1, setting the confidence score threshold for filtering predictions.

- **`iou_threshold`**: Float between 0 and 1, setting the IOU threshold for filtering predictions.

- **`augment`**: Boolean determining whether to perform augmentation during prediction, potentially improving performance.

- **`classes`**: List of classes to detect. If set to `null`, all YOLO classes will be detected.

- **`box_line_width`**: Line width of bounding boxes drawn on output images. If set to `null`, the value is automatically determined.

- **`results_path`**: Output path where images with bounding boxes and the JSON file with object positions are saved. If the folder exists, results will be overwritten.


