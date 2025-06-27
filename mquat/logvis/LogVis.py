# from dash import Dash, dcc, html, Input, Output
# import plotly.express as px
# import json
# from LogVisScalar import LogVisScalar
# from LogVisHist import LogVisHist
#
# from logvis.LogFolder import LogFolder
#
# mainLogs = LogFolder("C:\\Users\\fdai0217\\Documents\\mquat projekt\\mquat\\examples\\LeNet5\\logs\\conv1")
#
# app = Dash(__name__)
#
#
# all_vis = []
# all_vis.append(LogVisScalar(app, mainLogs.sub_folders[2].sub_folders[1].logs[0]))
# all_vis.append(LogVisHist(app, mainLogs.sub_folders[2].sub_folders[0].logs[0]))
#
# all_vis_html = []
# for vis in all_vis:
#     all_vis_html.append(vis.html)
#
# app.layout = html.Div(all_vis_html)
#
# app.run_server(debug=True)
import os
from dash import Dash, html, dcc, Input, Output
import dash
from LogVisScalar import LogVisScalar
from LogVisHist import LogVisHist
from logvis.LogFolder import LogFolder
from logvis.LogVisFolderPage import LogVisFolderPage
import dash_cytoscape as cyto
app = Dash(__name__, use_pages=True)

# LogVisScalar(app, mainLogs.sub_folders[2].sub_folders[1].logs[0])
#LogVisScalar(app, mainLogs.sub_folders[1].sub_folders[0].logs[0])
# LogVisHist(app, mainLogs.sub_folders[2].sub_folders[0].logs[0])

folderpath = "C:\\Users\\fdai0217\\Documents\\mquat projekt\\mquat\\examples\\LeNet5\\logs"

# for sub_folder in os.listdir(folderpath):
#     log_folder = LogFolder(os.path.join(folderpath, sub_folder))
#     LogVisFolderPage(app, log_folder)

#LogVisFolderPage(app, mainLogs.sub_folders[4].sub_folders[0])

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


nodes = [
    {
        'data': {'id': short, 'label': label},
    }
    for short, label, long, lat in (
        ('la', 'Los Angeles', 0.0, 0.0),
        ('nyc', 'New York', 40.71, -74),
        ('to', 'Toronto', 43.65, -79.38),
        ('mtl', 'Montreal', 45.50, -73.57),
        ('van', 'Vancouver', 49.28, -123.12),
        ('chi', 'Chicago', 41.88, -87.63),
        ('bos', 'Boston', 42.36, -71.06),
        ('hou', 'Houston', 29.76, -95.37)
    )
]

edges = [
    {'data': {'source': source, 'target': target}}
    for source, target in (
        ('van', 'la'),
        ('la', 'chi'),
        ('hou', 'chi'),
        ('to', 'mtl'),
        ('mtl', 'bos'),
        ('nyc', 'bos'),
        ('to', 'hou'),
        ('to', 'nyc'),
        ('la', 'nyc'),
        ('nyc', 'bos')
    )
]


default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'background-color': '#BFD7B5',
            'label': 'data(label)'
        }
    }
]


app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape-event-callbacks-2',
        layout={'name': 'breadthfirst'},
        elements=edges+nodes,
        stylesheet=default_stylesheet,
        style={'width': '100%', 'height': '450px'}
    ),
    html.P(id='cytoscape-tapNodeData-output'),
    html.P(id='cytoscape-tapEdgeData-output'),
    html.P(id='cytoscape-mouseoverNodeData-output'),
    html.P(id='cytoscape-mouseoverEdgeData-output')
])


@app.callback(Output('cytoscape-tapNodeData-output', 'children'),
              Input('cytoscape-event-callbacks-2', 'tapNodeData'))
def displayTapNodeData(data):
    if data:
        return "You recently clicked/tapped the city: " + data['label']


@app.callback(Output('cytoscape-tapEdgeData-output', 'children'),
              Input('cytoscape-event-callbacks-2', 'tapEdgeData'))
def displayTapEdgeData(data):
    if data:
        return "You recently clicked/tapped the edge between " + \
               data['source'].upper() + " and " + data['target'].upper()


@app.callback(Output('cytoscape-mouseoverNodeData-output', 'children'),
              Input('cytoscape-event-callbacks-2', 'mouseoverNodeData'))
def displayTapNodeData(data):
    if data:
        return "You recently hovered over the city: " + data['label']


@app.callback(Output('cytoscape-mouseoverEdgeData-output', 'children'),
              Input('cytoscape-event-callbacks-2', 'mouseoverEdgeData'))
def displayTapEdgeData(data):
    if data:
        return "You recently hovered over the edge between " + \
               data['source'].upper() + " and " + data['target'].upper()


# app.layout = html.Div([
# 	html.H1('Multi-page app with Dash Pages'),
#
#     html.Div(
#         [
#             # html.Div(
#             #     dcc.Link(
#             #         f"{page['name']} - {page['path']}", href=page["relative_path"]
#             #     )
#             # )
#             # for page in dash.page_registry.values()
# cyto.Cytoscape(
#         id='cytoscape',
#         elements=[
#             {'data': {'id': 'ca', 'label': 'Canada'}},
#             {'data': {'id': 'on', 'label': 'Ontario'}},
#             {'data': {'id': 'qc', 'label': 'Quebec'}},
#             {'data': {'source': 'ca', 'target': 'on'}},
#             {'data': {'source': 'ca', 'target': 'qc'}}
#         ],
#         layout={'name': 'breadthfirst'},
#         style={'width': '400px', 'height': '500px'}
#     )
#         ]
#     ),
#
# 	dash.page_container
# ])

if __name__ == '__main__':
	app.run_server(debug=True)