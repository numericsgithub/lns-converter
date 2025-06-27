from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import json
import numpy as np
from logvis.Log import Log
from logvis.LogVisBase import LogVisBase
import plotly.graph_objects as go

class LogVisHist(LogVisBase):

    def __init__(self, app, log: Log):
        super().__init__()
        self.log = log
        self.app = app
        self.x = []
        self.y = []
        self.z = []
        for step, x in log.getLocalStepIter():
            #self.y.append(np.ndarray.flatten(x))
            bins = 5000
            hist = np.histogram(np.ndarray.flatten(x), bins=bins)
            self.y.append(np.log10(hist[0]))
            self.x.append(hist[1])
            self.z.append(np.ones_like(hist[0]) * step)
        self.fig = go.Figure()
        for x, y, z in zip(self.x, self.y, self.z):
            self.fig.add_trace(go.Scatter3d(x = x, y = y, z = z))
        # self.fig = px.histogram(
        #         x=self.y[0], nbins=5000, log_y=True, # replace with your own data source
        #         title=self.log.name + "." + self.log.propname, height=325)
        # self.fig = px.scatter(
        #     x=self.x, y=self.y,  # replace with your own data source
        #     title="sample figure", height=325)
        # self.fig.update_layout(
        #     xaxis=dict(
        #         tickmode='linear',
        #         tick0=1.,
        #         dtick=1.
        #     ))
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
