from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import json
import numpy as np
from logvis.Log import Log

class LogVisBase:
    __ID_COUNTER = 0

    def _id(self, id):
        return str(id)+"_uid"+str(self.__uid)

    def __init__(self):
        self.__uid = LogVisBase.__ID_COUNTER
        LogVisBase.__ID_COUNTER += 1

