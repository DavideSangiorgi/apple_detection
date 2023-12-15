from enum import Enum
from pathlib import Path

from ultralytics import YOLO

from utils import load_config, parse_args, save_object_positions
from utils.storage import LocalStorageManager

###


LocalStorage = LocalStorageManager()


###

"""
TODO: 
    - add some image preprocesing
    - comment everything
    - organize imports
    - localstorage deve essere 1!
    - make all comments start with upper case
    - replace config params that go directly to evaluate with the dict call directly
"""


###


def main(config_path: Path):
    ### Load configuration
    config = load_config(local_storage=LocalStorage, config_path=config_path)

    yolo_model: str = config["YOLO_model"]
    device: str = config["device"]
    class_names: list = config["classes"]
    results_path: Path = config["results_path"]

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
        line_width=config["box_line_width"],
        project=str(results_path.parent),
        name=str(results_path.name),
    )

    ### Save objects position
    save_object_positions(
        predictions=predictions,
        local_storage=LocalStorage,
        idx_cls_map=model.names,
        results_path=results_path,
    )


###


if __name__ == "__main__":
    _args = parse_args()

    main(config_path=_args.config)
