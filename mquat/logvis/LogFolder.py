import numpy as np
import json
import os

from logvis.Log import Log


class LogFolder:
    def __init__(self, folderpath:str):
        self.folderpath = folderpath
        self.foldername = os.path.splitext(os.path.basename(folderpath))[0]
        self.sub_folders = []
        if not os.path.exists(folderpath):
            raise Exception("Folder does not exists!" + str(folderpath))

        # Read meta data
        meta_filepath = os.path.join(self.folderpath, self.foldername + ".meta")
        if not os.path.exists(meta_filepath):
            raise Exception("Meta file does not exists!" + str(meta_filepath))
        f = open(meta_filepath, "r")
        meta = json.load(f)
        f.close()
        self.classname = meta["class"]
        self.name_structured = meta["name_structured"]
        self.name = meta["name"]
        self.depth = meta["depth"]
        self.propname = meta["propname"]
        self.timestamp = meta["timestamp"]
        self.quantclass = meta["quantclass"] if "quantclass" in meta else None

        # Read all Logs # todo do not read the whole thing. Your memory ...
        all_log_files = [f.path for f in os.scandir(folderpath) if not f.is_dir() and f.path.endswith("npz")]
        self.logs = []
        for log_filepath in all_log_files:
            self.logs.append(Log(log_filepath))

        # Read all sub folders
        all_sub_folders = [f.path for f in os.scandir(folderpath) if f.is_dir()]
        for sub_folderpath in all_sub_folders:
            self.sub_folders.append(LogFolder(sub_folderpath))


