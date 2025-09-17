# ALWAYS end this with a "/"
MAIN_DATA_FOLDER_PATH = "data/"

import zlib
from json import JSONEncoder

class SimpleJSONEncoder(JSONEncoder):
    def default(self, o):
        if hasattr(o, "__dict__"):
            return o.__dict__
        else:
            return str(o)

def compress_and_save(filename: str, text: str) -> None:
    """
    Compresses text and saves it to a file.

    Args:
        filename (str): The name of the file to save the compressed data.
        text (str): The text to compress and save.
    """
    # Compress the text using zlib
    compressed_data = zlib.compress(text.encode('utf-8'))

    # Write the compressed data to the file in binary mode
    with open(filename, 'wb') as file:
        file.write(compressed_data)


def read_and_decompress(filename: str) -> str:
    """
    Reads compressed data from a file and decompresses it.

    Args:
        filename (str): The name of the file to read the compressed data from.

    Returns:
        str: The decompressed text.
    """
    # Read the compressed data from the file in binary mode
    with open(filename, 'rb') as file:
        compressed_data = file.read()

    # Decompress the data using zlib
    decompressed_text = zlib.decompress(compressed_data).decode('utf-8')
    return decompressed_text

def datpat(path:str):
    path = path.replace("\\", "/")
    if path.startswith("/"):
        path = path[1:]
    return MAIN_DATA_FOLDER_PATH + path

def get_model_base_path(args):
    model_name = args["model-name"]
    train_type = args["train-type"]
    quant_depth = args["quant-depth"]
    bias_bits_total = args["bias-bits"]
    weights_bits_total = args["weight-bits"]
    activation_bits_total = args["activation-bits"]
    adder = args["adder-type"]
    general_desc = args["desc"]

    # ["float", "fixed", "adder"]
    SAVE_PREFIX = {-1: lambda: None,
                   "float": lambda: f"data/training/{model_name}/float/{general_desc}/" + "{}",
                   "fixed": lambda: f"data/training/{model_name}/{quant_depth}_fixed/{weights_bits_total}w_{bias_bits_total}b_{activation_bits_total}a/{general_desc}/" + "{}",
                   "adder": lambda: f"data/training/{model_name}/{quant_depth}_adder/{adder}_adder_{weights_bits_total}w_{bias_bits_total}b_{activation_bits_total}a/{general_desc}/" + "{}",
                   "lns": lambda: f"data/training/{model_name}/{quant_depth}_lns/{weights_bits_total}w_{bias_bits_total}b_{activation_bits_total}a/{general_desc}/" + "{}"}

    model_base_path = SAVE_PREFIX[train_type]()
    return model_base_path

def get_model_paths(args):
    model_base_path = get_model_base_path(args)
    MODEL_SAVEPATH = model_base_path.format("model.npz")
    MODEL_SAVEPATH_Q = model_base_path.format("model_q.npz")
    MODEL_SAVEPATH_BEST = model_base_path.format("model_best.npz")
    MODEL_SAVEPATH_Q_BEST = model_base_path.format("model_best_q.npz")
    MODEL_SAVEPATH_Q_FUSEB = model_base_path.format("model_q_fusebatch.npz")
    return model_base_path, MODEL_SAVEPATH, MODEL_SAVEPATH_Q, MODEL_SAVEPATH_BEST, MODEL_SAVEPATH_Q_BEST, MODEL_SAVEPATH_Q_FUSEB
