import tensorflow as tf
import numpy as np
import json
import datetime


class Logging:
    milestone_name = "default"
    __current_log = {}

    @staticmethod
    def SaveResults():
        Logging.__SaveResults()

    @staticmethod
    def AddMilestone(milestone_name):
        Logging.__SaveResults()
        Logging.milestone_name = milestone_name

    @staticmethod
    def Log(name, object):
        tf.py_function(func=Logging.__Log(name, object), inp=[name, object], Tout=tf.float32)

    @staticmethod
    def __Log(name, scalar):
        if name not in Logging.__current_log:
            Logging.__current_log[name] = []
        Logging.__current_log[name].append(scalar.numpy().tolist())
        return 0.0

    @staticmethod
    def __SaveResults():
        with open(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S") + Logging.milestone_name + '.json', 'w') as fp:
            json.dump(Logging.__current_log, fp)
        Logging.__current_log = {}
        pass