# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:33:38 2022

@author: JC056455
"""
import base64
import datetime
import io
import math
import pandas as pd

from plotly.offline import plot
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shapely
from shapely.geometry import LineString, Point


### user inputs
#amb_p = 14.7 #user input
#amb_temp = 100 # user input
#inl_p = 14.496 #user ipnut
Cp = 0.2403 #potential User input. could be calculated from graph and pressure but these are typically constant in most applications.
Cv = 0.1714 #potential User input. could be calculated from graph and pressure but these are typically constant in most applications.
#pipe_dia = 48 ## user input
#fluid_p = 34.5 ## user input
#effcncy = 0.6 ## user input


## constants
R_i = 1545.33
M = 28.967
R = R_i/M
K = Cp/Cv
n = (K-1)/K
pipe_vel = {3:{"low":1200, "high":1800},10:{"low":1800, "high":3000}, 24:{"low":2700, "high":4000}, 60:{"low":3800, "high":6500}}


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
server = app.server

Amb_Card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Ambient Conditions", className="card-title"),
                    html.P("Pressure (psi)",
                        className="card-text",
                    ),
                    dcc.Input(id="ambient-pressure", type="number", value=14.7, step=0.01),
                    html.P("Temperature (°F)",
                        className="card-text",
                    ),
                    dcc.Input(id="ambient-temp", type="number", value=100, step=0.1)
                ]
            )
        )

Press_Card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Pressure Conditions", className="card-title"),
                    html.P("Inlet Fluid Pressure (psi)",
                        className="card-text",
                    ),
                    dcc.Input(id="inlet-pressure", type="number", value=14.5, step=0.01),
                    html.P("Outlet Fluid Pressure (psi)",
                        className="card-text",
                    ),
                    dcc.Input(id="outlet-pressure", type="number", value=34.5, step=0.01)
                ]
            )
        )

Sys_Card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("System Conditions", className="card-title"),
                    html.P("Pipe Diameter (inches)",
                        className="card-text",
                    ),
                    dcc.Input(id="diameter", type="number", value=60, step=1, min=1, max=60),
                    html.P("Efficiency (%)",
                        className="card-text",
                    ),
                    dcc.Input(id="efficiency", type="number", value=60, step=1, min=60, max=100)
                ]
            )
        )
Inl_Conv_Card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Inlet Flow Conversion", className="card-title"),
                    html.P("FLOW (SCFM)",
                        className="card-text",
                    ),
                    dcc.Input(id="inl-Q-SCFM", type="number", value = 0),
                    html.P("FLOW (ACFM)",
                        className="card-text",
                    ),
                    dcc.Input(id="inl-Q-ACFM", type="number", value = 0)
                ]
            )
        )
Out_Conv_Card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Outlet Flow Conversion", className="card-title"),
                    html.P("FLOW (SCFM)",
                        className="card-text",
                    ),
                    dcc.Input(id="out-Q-SCFM", type="number", value = 0),
                    html.P("FLOW (ACFM)",
                        className="card-text",
                    ),
                    dcc.Input(id="out-Q-ACFM", type="number", value = 0)
                ]
            )
        )

inl_graph_Card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Inlet Velocity Versus Flow (SCFM) Colored by Velocity Classifier Graph", className="card-title"),
                    dcc.Graph(id='inl-my-vel_graph')
                    #dcc.Input(id="min-SCFM", type="number", value = 0, disabled = True),
                    #dcc.Input(id="max-SCFM", type="number", value = 0, disabled = True),
                ]
            )
        )

out_graph_Card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Outlet Velocity Versus Flow (SCFM) Colored by Velocity Classifier Graph", className="card-title"),
                    dcc.Graph(id='out-my-vel_graph')
                    #dcc.Input(id="min-SCFM", type="number", value = 0, disabled = True),
                    #dcc.Input(id="max-SCFM", type="number", value = 0, disabled = True),
                ]
            )
        )



cards = html.Div([
     dbc.Row([
         dbc.CardGroup([Amb_Card,Press_Card,Sys_Card,Inl_Conv_Card,Out_Conv_Card])
         ]),
     dbc.Row([
         dbc.Col(inl_graph_Card, width = 6),
         dbc.Col(out_graph_Card, width = 6),
         ])
     ])


app.layout = html.Div(cards)




def calc_acfm(scfm,dnsty):
    lbmin = 0.0751294*scfm
    acfm = lbmin/dnsty
    return acfm

def calc_scfm(acfm,dnsty):
    lbmin = acfm*dnsty
    scfm = lbmin/0.0751294
    return scfm


def calc_vel_class(dia,dnsty):
    for dict_dia in pipe_vel.keys():
        if dia <= dict_dia:
            low_vel = pipe_vel[dict_dia]["low"]
            high_vel = pipe_vel[dict_dia]["high"]
            break
    scfm = 0
    vel_fpm = 0
    DIAMs = list()
    SCFMs = list()
    ACFMs = list()
    VEL_FPSs = list()
    VEL_FPMs = list()
    VEL_ranks = list()
    while vel_fpm < high_vel+1500:
        lbmin = 0.0751294*scfm
        acfm = lbmin/dnsty
        vel_fps = (acfm*4)/(math.pi*((dia/12)**2)*60)
        vel_fpm = (acfm*4)/(math.pi*((dia/12)**2))
        DIAMs.append(dia)
        SCFMs.append(scfm)
        ACFMs.append(acfm)
        VEL_FPSs.append(vel_fps)
        VEL_FPMs.append(vel_fpm)
        if vel_fpm<low_vel:
            rank = "Low"
        elif vel_fpm>high_vel:
            rank = "High"
        else:
            rank = "Acceptable"
        VEL_ranks.append(rank)
        if dia < 20:
            scfm = scfm + dia/20
        else:
            scfm = scfm + dia
    pipe_vels = pd.DataFrame(data = {"Diameter (in)":DIAMs,"Flow (SCFM)": SCFMs, "Flow (ACFM)": ACFMs, "Velocity (FPS)": VEL_FPSs, "Velocity (FPM)": VEL_FPMs, "Velocity Class": VEL_ranks})
    return pipe_vels

def generate_plot(dia, dnsty):
    velocity_data = calc_vel_class(dia,dnsty)
    accptbl_velocity_data = velocity_data[velocity_data["Velocity Class"] == "Acceptable"]
    lwr_accptbl = accptbl_velocity_data[accptbl_velocity_data["Velocity (FPM)"] == min(accptbl_velocity_data["Velocity (FPM)"])]
    upr_accptbl = accptbl_velocity_data[accptbl_velocity_data["Velocity (FPM)"] == max(accptbl_velocity_data["Velocity (FPM)"])]
    
    fig = px.line(velocity_data, x="Flow (SCFM)", y = "Velocity (FPM)", color = "Velocity Class", template='seaborn', hover_data={"Diameter (in)":':.0f',"Flow (SCFM)":':.0f',"Flow (ACFM)":':.0f',"Velocity (FPS)":':.0f',"Velocity (FPM)":':.2f',"Velocity Class": True})
    
    #x = [lwr_accptbl["Flow (SCFM)"].iloc[0],lwr_accptbl["Flow (SCFM)"].iloc[0]]
    #y = [(lwr_accptbl["Velocity (FPM)"].iloc[0])-250.0,(lwr_accptbl["Velocity (FPM)"].iloc[0])+250.0]
    
    fig.add_scatter(x= [lwr_accptbl["Flow (SCFM)"].iloc[0],lwr_accptbl["Flow (SCFM)"].iloc[0]],y=[lwr_accptbl["Velocity (FPM)"].iloc[0]-250,lwr_accptbl["Velocity (FPM)"].iloc[0]+250], name= "Lowest Acceptable SCFM", mode="lines", line=dict(color="black"),hovertemplate = 'Lowest Acceptable Flow (SCFM):%{x:.0f}')
        
    fig.add_scatter(x= [upr_accptbl["Flow (SCFM)"].iloc[0],upr_accptbl["Flow (SCFM)"].iloc[0]],y=[upr_accptbl["Velocity (FPM)"].iloc[0]-250,upr_accptbl["Velocity (FPM)"].iloc[0]+250], name= "Highest Acceptable SCFM", mode="lines", line=dict(color="black"),hovertemplate = 'Highest Acceptable Flow (SCFM):%{x:.0f}')
    
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
        ))
    
    return fig

#plot(fig)


@app.callback(
    Output(component_id='inl-Q-SCFM', component_property='value'),
    Output(component_id='inl-Q-ACFM', component_property='value'),
    [Input(component_id='inl-Q-SCFM', component_property='value'),
     Input(component_id='inl-Q-ACFM', component_property='value'),
     Input(component_id='ambient-temp', component_property='value'),
     Input(component_id='inlet-pressure', component_property='value'),
     Input(component_id='ambient-pressure', component_property='value'),],
    prevent_initial_call=False
)
def inl_ACFM_SCFM(SCFM, ACFM, amb_temp, inl_p, amb_p):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    inl_temp = ((amb_temp + 460)*(inl_p/amb_p)**n)-460
    inl_fld_dnsty = (144/R)*(inl_p/(inl_temp+460))
    if trigger_id == "inl-Q-ACFM":
        SCFM_val = round(calc_scfm(ACFM,inl_fld_dnsty))
        ACFM_val = ACFM
    else:
        ACFM_val = round(calc_acfm(SCFM,inl_fld_dnsty))
        SCFM_val = SCFM
    print(round(calc_scfm(ACFM,inl_fld_dnsty),2))
    return SCFM_val, ACFM_val


@app.callback(
    Output(component_id='out-Q-SCFM', component_property='value'),
    Output(component_id='out-Q-ACFM', component_property='value'),
    [Input(component_id='out-Q-SCFM', component_property='value'),
     Input(component_id='out-Q-ACFM', component_property='value'),
     Input(component_id='ambient-temp', component_property='value'),
     Input(component_id='inlet-pressure', component_property='value'),
     Input(component_id='outlet-pressure', component_property='value'),
     Input(component_id='efficiency', component_property='value'),
     Input(component_id='ambient-pressure', component_property='value'),],
    prevent_initial_call=False
)
def out_ACFM_SCFM(SCFM, ACFM, amb_temp, inl_p, out_p, effcncy, amb_p):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    inl_temp = ((amb_temp + 460)*(inl_p/amb_p)**n)-460
    fluid_temp = (inl_temp+460)*((1/effcncy/100)*(out_p/inl_p)**n-1/effcncy/100+1)-460
    outlt_fld_dnsty = (144/R)*(out_p/(fluid_temp+460))
    if trigger_id == "out-Q-ACFM":
        SCFM_val = round(calc_scfm(ACFM,outlt_fld_dnsty))
        ACFM_val = ACFM
    else:
        ACFM_val = round(calc_acfm(SCFM,outlt_fld_dnsty))
        SCFM_val = SCFM
    print(round(calc_scfm(ACFM,outlt_fld_dnsty),2))
    return SCFM_val, ACFM_val

@app.callback(Output(component_id='inl-my-vel_graph', component_property='figure'),
               [Input(component_id='ambient-temp', component_property='value'),
                Input(component_id='inlet-pressure', component_property='value'),
                Input(component_id='ambient-pressure', component_property='value'),
                Input(component_id='diameter', component_property='value'),],
               prevent_initial_call=False)
def gen_inl_graph(amb_temp, inl_p, amb_p, dia):
    inl_temp = ((amb_temp + 460)*(inl_p/amb_p)**n)-460
    inl_fld_dnsty = (144/R)*(inl_p/(inl_temp+460))
    fig = generate_plot(dia, inl_fld_dnsty)
    return fig
    
    
    
@app.callback(Output(component_id='out-my-vel_graph', component_property='figure'),
               [Input(component_id='ambient-temp', component_property='value'),
                Input(component_id='inlet-pressure', component_property='value'),
                Input(component_id='outlet-pressure', component_property='value'),
                Input(component_id='efficiency', component_property='value'),
                Input(component_id='ambient-pressure', component_property='value'),
                Input(component_id='diameter', component_property='value'),],
               prevent_initial_call=False)
def gen_out_graph(amb_temp, inl_p, out_p, effcncy, amb_p, dia):
    inl_temp = ((amb_temp + 460)*(inl_p/amb_p)**n)-460
    fluid_temp = (inl_temp+460)*((1/effcncy/100)*(out_p/inl_p)**n-1/effcncy/100+1)-460
    outlt_fld_dnsty = (144/R)*(out_p/(fluid_temp+460))
    fig = generate_plot(dia, outlt_fld_dnsty)
    return fig
    
    
if __name__ == '__main__':
    app.run_server(debug=False)