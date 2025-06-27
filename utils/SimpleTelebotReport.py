import requests as req
import os

# Create this file and put the bot credentials in here
# The file contains a single line of text with the telegram credentials like:
# bot9519501293:AAFi7VSvlQMMIlv7-ABC123Fi7VSv12
try:
    with open("data/telegram.txt", "r") as f:
        creds = f.read()
except:
    print("data/telegram.txt not found! So you will not get any updates in telegram!")
    creds = None

def sendModelReport(model_name, args, message:str):
    try:
        prefix = f"{model_name}_{args['desc']} {args['train-type']}"
        if args['train-type'] == "fixed":
            prefix += f" {args['weight-bits']}"
        if args['train-type'] == "adder":
            prefix += f" {args['adder-type']} {args['weight-bits']}"
        if model_name == "MobileNetV1":
            prefix += f" {args['mobnet_alpha']}"
        sendReport(f"{prefix}: " + message)
    except:
        pass

def sendReport(message:str):
    try:
        message = str(message)
        if len(message) > 4000:
            message = message[0:4000]
        if creds is not None:
            req.get("https://api.telegram.org/"+creds+"/sendMessage", params={"text": message, "chat_id":"@cluster40channel"})
    except:
        pass

def sendImageReport(message:str, image_path):
    try:
        data = {"chat_id": "@cluster40channel", "caption": message}
        url = f"https://api.telegram.org/{creds}/sendPhoto"
        with open(image_path, "rb") as image_file:
            ret = req.post(url, data=data, files={"photo": image_file})
        return ret.json()
    except:
        pass
        return None