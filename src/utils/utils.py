import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from torch.cuda import device_count

from .storage import LocalStorageManager

YOLO_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

DEVICES = ["cpu"] + [f"cuda:{n}" for n in range(device_count())]


def load_config(local_storage: LocalStorageManager, config_path: str) -> dict:
    """
    Check that configuration parameters are in compliance with the corresponding restictions
    and converting, when necessary, raw values to more convenient class instances

    Attributes:
        local_storage: LocalStorageManager instance of the project to access project directories
        config_path: str path of the JSON configuration file
    """
    ### Load JSON
    raw_config = local_storage.load_json(config_path)

    ### Check model
    assert (
        raw_config["YOLO_model"] in YOLO_MODELS
    ), f'{raw_config["YOLO_model"]} not in {YOLO_MODELS}'

    raw_config["YOLO_model"] = local_storage.dirs.models.joinpath(
        raw_config["YOLO_model"]
    )

    ### Check device
    assert raw_config["device"] in DEVICES, f'{raw_config["device"]} not in {DEVICES}'

    ### Check confidence_threshold
    assert isinstance(raw_config["confidence_threshold"], float) or isinstance(
        raw_config["confidence_threshold"], int
    ), f'confidence_threshold is of type {type(raw_config["confidence_threshold"])}'

    assert (
        0 <= raw_config["confidence_threshold"] <= 1
    ), f'confidence_threshold = {raw_config["confidence_threshold"]}'

    ### Check iou_threshold
    assert isinstance(raw_config["iou_threshold"], float) or isinstance(
        raw_config["iou_threshold"], int
    ), f'iou_threshold is of type {type(raw_config["iou_threshold"])}'

    assert (
        0 <= raw_config["iou_threshold"] <= 1
    ), f'iou_threshold = {raw_config["iou_threshold"]}'

    ### Check augmentation
    assert isinstance(
        raw_config["augment"], bool
    ), f'augment is of type {type(raw_config["augment"])}'

    ### Check classes
    ### convert into list if necessary
    if isinstance(raw_config["classes"], str):
        raw_config["classes"] = [raw_config["classes"]]

    assert isinstance(
        raw_config["classes"], list
    ), f'classes is of type {type(raw_config["classes"])}'

    assert all(
        [isinstance(c, str) for c in raw_config["classes"]]
    ), f'{raw_config["classes"]}'

    ### uniform string case
    raw_config["classes"] = [c.title() for c in raw_config["classes"]]

    ### Check box_line_width
    assert raw_config["box_line_width"] is None or isinstance(
        raw_config["box_line_width"], int
    ), f'box_line_width is of type {type(raw_config["box_line_width"])}'

    ### Check results_path
    assert isinstance(raw_config["results_path"], str)
    ### convert into path
    raw_config["results_path"] = Path(raw_config["results_path"])
    ### make only parent directory, result directory will be created by YOLO
    raw_config["results_path"].parent.mkdir(exist_ok=True)
    ### if directory already existing, remove it
    if raw_config["results_path"].exists():
        shutil.rmtree(str(raw_config["results_path"]))

    return raw_config


def get_object_positions(
    predictions,
    idx_cls_map: dict,
) -> dict:
    """
    Extracts from YOLO bounding boxes the centre of detected objects
    and groups them by input image and detection class.

    Attributes:
        predictions: YOLO predict() output
        local_storage: LocalStorageManager instance of the project to access project directories
        idx_cls_map: dictionary mapping YOLO class indexes to corresponding class names
        results_path: output path where to save the JSON file containing object positions
    """
    position_map = {}

    ### Iterate over the input images
    for pred in predictions:
        ### Get input image path
        img_path = Path(pred.path)
        img_name = img_path.stem

        ### Use image name as first grouping key of the dictionary
        position_map[img_name] = {}

        position_map[img_name]["path"] = str(img_path)

        position_map[img_name]["position"] = {}

        ### Iterate over the objects detected in a single image
        for cls, xywh in zip(pred.boxes.cls, pred.boxes.xywh):
            ### Get position and add to the corresponding class position list
            position = [int(xywh[0]), int(xywh[1])]

            cls_name = idx_cls_map[int(cls)]
            position_map[img_name]["position"][cls_name] = position_map[img_name][
                "position"
            ].get(cls_name, []) + [position]

        # ### Save positions into JSON file
        # position_map_path = results_path.joinpath("positions.json")

        # local_storage.store_json(path_raw=str(position_map_path), data=position_map)

    return position_map


def draw_positions(position_map: dict, results_path: Path, center_rad: int) -> None:
    """
    Draws the centre of bounding boxes into corresponding images
    """
    ### Iterate over all images
    for img_dict in position_map.values():
        ### Get output image
        input_img_path = Path(img_dict["path"])
        output_img_path = results_path.joinpath(input_img_path.name)
        img = Image.open(output_img_path)

        ### Iterate over all detected class positions
        for cls_positions in img_dict["position"].values():
            ### Iterate over all detected objects of the class
            for center_xy in cls_positions:
                ### Get centre bounding bos for drawing
                xy = [
                    center_xy[0] - center_rad,
                    center_xy[1] - center_rad,
                    center_xy[0] + center_rad,
                    center_xy[1] + center_rad,
                ]

                ### Draw centre
                ImageDraw.Draw(img).ellipse(xy=xy, fill="blue", outline="blue")

        ### Save updated image
        img.save(output_img_path)
