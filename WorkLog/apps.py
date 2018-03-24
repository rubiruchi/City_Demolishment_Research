import dash
import dash_core_components as dcc
import dash_html_components as html

import glob
import os
import base64
import pandas as pd

image_directory = 'combineplot/expmaxdifference/'
list_of_images = [os.path.basename(x)[:-4] for x in glob.glob('{}*p1.png'.format(image_directory))]

improvementdict = pd.read_csv(image_directory + 'improvementdict',index_col=0)

app = dash.Dash()
app.layout = html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='file',
                        options=[{'label': i, 'value': i} for i in list_of_images ],
                        )
                    ],
                    style={'width': '48%', 'display': 'inline-block'} ),


                html.Div([
                html.Div([
                    html.Img(id = 'image', style = {'width' : '100%'}),
                    dcc.Slider(
                    id='slider',
                    step=None ,
                    min = 0.5, max = 2.5,
                    marks={2:"maxdif",
                           1:"normal"},
                    value = 1
                    )], style = {'width' : '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph( id='graph')], style = {'width' : '45%', 'float': 'right', 'display': 'inline-block'})
                ])
            ])

"""
@app.callback(
        dash.dependencies.Output('table','children'),
        [dash.dependencies.Input('file','value')])
def update_table(value):

    dataframe  = pd.read_csv(value,index_col=0)

    return [html.Tr([html.Th(col) for col in dataframe.columns[:4]])] + \
            [html.Tr([ html.Td(dataframe.iloc[i][col]) for col in dataframe.columns[:4]
                                                   ]) for i in range(len(dataframe))]
"""

@app.callback(
            dash.dependencies.Output('graph','figure'),
            [dash.dependencies.Input('file','value')])
def update_graph(value):
    dis = value.split('-')[1][2:-1]
    p = value.split('-')[2][1:]
    model = "modelexp-d{}-p{}".format(dis,p)
    return { 'data' : [
        {'x': range(1,10) , 'y': improvementdict[(improvementdict.model == model) & (improvementdict.obj == "maxdif")].ObjValForOccupied, 'type': 'line', 'name': 'maxdif'},
        {'x': range(1,10) , 'y': improvementdict[(improvementdict.model == model) & (improvementdict.obj == "normal")].ObjValForOccupied, 'type': 'line', 'name': 'normal'}
        ],
        'layout': { 'title': 'Object Value', 'height': '30%' }
        }



@app.callback(
            dash.dependencies.Output('image', 'src'),
            [dash.dependencies.Input('file', 'value'),
             dash.dependencies.Input('slider','value')])
def update_image_src(file_name,slider_value):
    if slider_value == 2:
        image_filename = image_directory + file_name + '.png'
    else:
        image_filename = image_directory + file_name + '-n.png'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image)



if __name__ == '__main__':
        app.run_server(debug=True)
