# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    html.Img(
                id="body-image",
                className="three columns"
    ),
    dcc.Dropdown(
                id='test-input',
                options=[{'label': 0, 'value': 0},
                         {'label': 1, 'value': 1}
                        ],
                value='Life expectancy at birth, total (years)'
    ),
])

@app.callback(Output("body-image", "src"),
             [Input('test-input', 'value')])
def update_body_image(hover_data):
    src = "https://www.w3schools.com/images/picture.jpg"
    return src



if __name__ == '__main__':
    app.run_server(debug=True)