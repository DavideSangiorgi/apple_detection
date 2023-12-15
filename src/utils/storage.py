import json
import os
from pathlib import Path
from typing import Union

###

T_JsonData = Union[dict, list]

###


class LocalStorageDirectoryManager:
    '''
    Initializes and validates directories used for the project.
    Provides a single point of access for key directories, 
    reducing ambiguity and enhancing codebase clarity.
    '''
    def __init__(self) -> None:
        self.root: Path = None
        self.data: Path = None
        self.data_test: Path = None

        self.__initialize()
        self.__mkdirs()
        self.__validate()

    #

    def __initialize(self) -> None:
        self.root = Path(__file__).parent.parent.parent
        self.data = self.root.joinpath("data")
        self.data_test = self.data.joinpath("test")

    def __mkdirs(self) -> None:
        self.data.mkdir(exist_ok=True)
        self.data_test.mkdir(exist_ok=True)

    def __validate(self) -> None:
        assert self.root.is_dir(), f"{self.root} is not a directory"
        assert self.data.is_dir(), f"{self.data} is not a directory"
        assert self.data_test.is_dir(), f"{self.data_test} is not a directory"


###


class LocalStorageManager:
    '''
    Initializes and validates directories used for the project.
    Provides a single point of access for key directories, 
    reducing ambiguity and enhancing codebase clarity.
    Integrates handling methods to LocalStorageDirectoryManager.
    '''
    def __init__(self) -> None:
        self.dirs = LocalStorageDirectoryManager()

    #

    @staticmethod
    def is_empty(directory: Path) -> bool:
        return len(os.listdir(str(directory))) < 1

    #

    @staticmethod
    def load_json(path_raw: str) -> T_JsonData:
        path = Path(path_raw)
        assert path.exists(), f"{path} does not exist"
        assert path.is_file(), f"{path} is not a file"
        assert path.suffix == ".json", f"{path} is not a .json file"

        data: T_JsonData = None

        with open(path_raw, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict) or isinstance(
            data, list
        ), f"data is of type {type(data)} and not dict or list"

        return data

    @staticmethod
    def store_json(path_raw: str, data: T_JsonData) -> None:
        assert isinstance(data, dict) or isinstance(
            data, list
        ), f"data is of type {type(data)} and not dict or list"

        with open(path_raw, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        path = Path(path_raw)
        assert path.exists(), f"{path} does not exist"
        assert path.is_file(), f"{path} is not a file"
