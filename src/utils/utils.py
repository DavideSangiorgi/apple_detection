from enum import Enum

from .storage import LocalStorageManager
from torch.cuda import device_count

from pathlib import Path
import shutil


YOLO_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

DEVICES = ["cpu"] + [f"cuda:{n}" for n in range(device_count())]


def load_config(local_storage: LocalStorageManager, config_path: str) -> dict:
    raw_config = local_storage.load_json(config_path)

    ### check selected version
    assert (
        raw_config["YOLO_model"] in YOLO_MODELS
    ), f'{raw_config["YOLO_model"]} not in {YOLO_MODELS}'

    ### check device
    assert raw_config["device"] in DEVICES, f'{raw_config["device"]} not in {DEVICES}'

    ### check confidence threshold
    assert isinstance(
        raw_config["confidence_threshold"], float
    ), f'confidence_threshold is of type {type(raw_config["confidence_threshold"])}'

    assert (
        0 <= raw_config["confidence_threshold"] <= 1
    ), f'confidence_threshold = {raw_config["confidence_threshold"]}'

    ### check augmentation
    assert isinstance(
        raw_config["augment"], bool
    ), f'augment is of type {type(raw_config["augment"])}'

    ### check classes
    if isinstance(raw_config["classes"], str):
        raw_config["classes"] = [raw_config["classes"]]

    assert isinstance(
        raw_config["classes"], list
    ), f'classes is of type {type(raw_config["classes"])}'

    assert all(
        [isinstance(c, str) for c in raw_config["classes"]]
    ), f'{raw_config["classes"]}'

    raw_config["classes"] = [c.title() for c in raw_config["classes"]]

    ### check box_line_width
    assert raw_config["box_line_width"] is None or isinstance(
        raw_config["box_line_width"], int
    ), f'box_line_width is of type {type(raw_config["box_line_width"])}'

    ### check results_path
    assert isinstance(raw_config["results_path"], str)
    raw_config["results_path"] = Path(raw_config["results_path"])
    raw_config["results_path"].parent.mkdir(exist_ok=True)
    if raw_config["results_path"].exists():
        shutil.rmtree(str(raw_config["results_path"]))

    return raw_config


def save_object_positions(
    predictions,
    local_storage: LocalStorageManager,
    idx_cls_map: dict,
    results_path: Path,
) -> None:
    position_map = {}
    for pred in predictions:
        img_path = Path(pred.path)
        img_name = img_path.stem
        position_map[img_name] = {}
        position_map[img_name]["path"] = str(img_path)

        position_map[img_name]["position"] = {}

        for cls, xywh in zip(pred.boxes.cls, pred.boxes.xywh):
            position = [
                int(xywh[0] + (xywh[2] / 2)),
                int(xywh[1] + (xywh[3] / 2)),
            ]

            cls_name = idx_cls_map[int(cls)]
            position_map[img_name]["position"][cls_name] = position_map[img_name][
                "position"
            ].get(cls_name, []) + [position]

        position_map_path = results_path.joinpath("positions.json")

        local_storage.store_json(path_raw=str(position_map_path), data=position_map)
