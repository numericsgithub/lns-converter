import numpy as np
import json
import os

class Log:
    def __init__(self, filepath:str):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise Exception("Log file does not exist" + str(filepath))
        if not filepath.endswith(".npz"):
            raise Exception("Log file does not end with .npz" + str(filepath))
        self.data = np.load(filepath, allow_pickle=True)
        meta_filepath = filepath[:-4] + ".meta"
        print(meta_filepath)
        f = open(meta_filepath, "r")
        meta_data = json.load(f)
        f.close()
        self.name_structured = meta_data["name_structured"]
        self.name = meta_data["name"]
        self.logger_class = meta_data["class"]
        self.quantizer_class = meta_data["quantclass"]
        self.depth = meta_data["depth"]
        self.propname = meta_data["propname"]
        self.timestamp = meta_data["timestamp"]
        data = np.load(filepath)
        self.data = data
        self.global_step_map = {}
        self.local_step_map = {}
        self.global_step_map_max = 0
        self.local_step_map_max = 0
        for key in data.keys():
            cur_key = key
            key = str(key)[len("gs_"):]
            cur_gs = int(key[0: key.index("_")])
            key = key[key.rindex("_") + 1:]
            cur_ls = int(key)

            if cur_ls > self.local_step_map_max:
                self.local_step_map_max = cur_ls
            if cur_gs > self.global_step_map_max:
                self.global_step_map_max = cur_gs

            if cur_gs not in self.global_step_map:
                self.global_step_map[cur_gs] = [cur_key]
            else:
                self.global_step_map[cur_gs].append(cur_key)
            if cur_ls not in self.local_step_map:
                self.local_step_map[cur_ls] = cur_key
            else:
                raise Exception("Duplicate local steps. This is impossible!")

    def __getStepIter(self, map, map_max, null_value, start=0, step_count_limit=None):
        step_counter = 0
        cur_step = start
        print("map_max", map_max)
        while True:
            if step_count_limit != None and step_counter >= step_count_limit:
                return
            if cur_step > map_max:
                return
            if cur_step not in map:
                yield cur_step, null_value
            else:
                yield cur_step, map[cur_step]
            step_counter += 1
            cur_step += 1

    def __stepIterSaturator(self, step_iter):
        for cur_step, keys in step_iter:
            if keys is None or keys == []: # just return yourself when empty
                yield cur_step, keys
            else:
                if isinstance(keys, str): # saturate data. Just replace the keys with the data
                    keys = self.data[keys] # for local steps
                else:
                    for i in range(len(keys)): # for global steps
                        keys[i] = self.data[keys[i]]
                yield cur_step, keys


    def getGlobalStepIter(self, start=0, step_count_limit=None):
        key_iterator = self.__getStepIter(self.global_step_map, self.global_step_map_max, [], start=start, step_count_limit=step_count_limit)
        return self.__stepIterSaturator(key_iterator)

    def getLocalStepIter(self, start=0, step_count_limit=None):
        key_iterator = self.__getStepIter(self.local_step_map, self.local_step_map_max, None, start=start, step_count_limit=step_count_limit)
        return self.__stepIterSaturator(key_iterator)

