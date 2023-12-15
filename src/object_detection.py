from pathlib import Path

from ultralytics import YOLO

from utils import draw_positions
from utils import get_object_positions
from utils import load_config
from utils import parse_args
from utils.storage import LocalStorageManager

###


LocalStorage = LocalStorageManager()


###


def main(config_path: Path):
    ### Load configuration
    config = load_config(local_storage=LocalStorage, config_path=config_path)

    yolo_model: str = config["YOLO_model"]
    device: str = config["device"]
    class_names: list = config["classes"]
    results_path: Path = config["results_path"]
    line_width: int = config["box_line_width"]

    ### Load pre-trained model
    model = YOLO(yolo_model)
    model.to(device)

    ### Map selected classes into corresponding idxs
    if class_names is not None:
        yolo_classes = {name.title(): idx for idx, name in model.names.items()}
        class_idxs = [yolo_classes[c_name] for c_name in class_names]
    else:
        class_idxs = None

    ### Prepare list of images for evaluation
    img_paths = list(LocalStorage.dirs.data_test.iterdir())

    ### Run model prediction
    predictions = model.predict(
        source=img_paths,
        conf=config["confidence_threshold"],
        augment=config["augment"],
        classes=class_idxs,
        save=True,
        line_width=line_width,
        project=str(results_path.parent),
        name=str(results_path.name),
    )

    ### Save objects position
    position_map = get_object_positions(
        predictions=predictions,
        idx_cls_map=model.names
    )

    ### Draw positions
    draw_positions(
        position_map=position_map,
        results_path=results_path,
        center_rad=line_width,
    )

    ### Save positions into JSON file
    position_map_path = results_path.joinpath("positions.json")
    LocalStorage.store_json(path_raw=str(position_map_path), data=position_map)


###


if __name__ == "__main__":
    _args = parse_args()

    main(config_path=_args.config)
