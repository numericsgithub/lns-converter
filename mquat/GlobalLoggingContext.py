import json
import os


class GlobalLoggingContext:

    def __init__(self, folder="logs"):
        self.folder = folder
        self.all_loggers = []

    def close_all(self):
        print("Close all called", self.all_loggers)
        for logger in self.all_loggers:
            print("Close all called for ", logger.name)
            logger.close()

    def save_model_structure(self, model):
        model_map = self.map_model_structure(model)
        with open(os.path.join(self.folder, "structure.json"), 'w') as f:
            json.dump(model_map, f)
        return model_map

    def flush_all(self):
        for logger in self.all_loggers:
            logger.flush()

    def map_model_structure(self, model):
        model_map = []
        for layer in model.getLayers():
            if hasattr(layer, "create_logger_mapping"):
                layer_map = layer.create_logger_mapping()
                model_map.append(layer_map)
        return model_map
