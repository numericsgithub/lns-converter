from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import json
import numpy as np
from logvis.Log import Log
from logvis.LogVisBase import LogVisBase

class LogVisScalar(LogVisBase):

    def __init__(self, app, log: Log):
        super().__init__()
        self.log = log
        self.app = app
        self.x = []
        self.y = []
        for step, x in log.getLocalStepIter():
            self.y.append(np.average(x))
            self.x.append(step)
        self.fig = px.line(
                x=self.x, y=self.y, # replace with your own data source
                title=self.log.name + "." + self.log.propname, height=325)
        # self.fig = px.scatter(
        #     x=self.x, y=self.y,  # replace with your own data source
        #     title="sample figure", height=325)
        self.fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=1.,
                dtick=1.
            ))
        self.html = html.Div([
            html.H4('Displaying figure structure as JSON'),
            dcc.Graph(id=self._id("graph"), figure=self.fig),
            dcc.Clipboard(target_id=self._id("structure")),
            # dcc.Dropdown(
            #     ["Test1", "Test2"],
            #     "Test2",
            #     id='layer'
            # ),
            # html.Pre(
            #     id=self._id("structure"),
            #     style={
            #         'border': 'thin lightgrey solid',
            #         'overflowY': 'scroll',
            #         'height': '275px'
            #     }
            # ),
        ])
        # @self.app.callback(
        #     Output(self._id("structure"), "children"),
        #     Input(self._id("graph"), "figure"))
        # def display_structure(fig_json):
        #     return json.dumps(fig_json, indent=2)
