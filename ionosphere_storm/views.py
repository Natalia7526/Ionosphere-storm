from django.shortcuts import render
import plotly.offline as pltoff
# from plotly.offline import plot
from ionosphere_storm.obliczeniaMagisterka import *
import os
from numpy.linalg import inv
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from django_plotly_dash import DjangoDash
from datetime import datetime


# Create your views here.


def about(request):
    return render(request, 'ionosphere_storm/about.html', {})


def home(request):
    files_in_directory = os.listdir('..\\Praca_magisterska')
    filtered_files = [file for file in files_in_directory if file.endswith(".gz")]
    for file in filtered_files:
        path_to_file = os.path.join('..\\Praca_magisterska', file)
        os.remove(path_to_file)

    return render(request, 'ionosphere_storm/home.html')


downloadFile_Galileo()
downloadFile('d')
downloadFile('n')

"""Wczytywanie i łączenie danych"""
TEC_GPS = getElevation_Azimuth_Angle("GPS")
TEC_Galileo = getElevation_Azimuth_Angle("Galileo")
TEC_data = pd.concat([TEC_GPS, TEC_Galileo], axis=0)

print(TEC_data.columns[13])

def stations(request):
    return render(request, 'ionosphere_storm/stations.html', {})


def storm(request):
    return render(request, 'ionosphere_storm/storm.html', {})


def charts(request):
    """Tymczasowy df =, w których następuje zliczenie danych po kolumnie sv"""
    temp = TEC_data.groupby('sv').apply(lambda x: x.count())
    temp = temp[['sTEC']].reset_index()

    """Wybieranie wierszy d=dla każdego z systemów"""
    GPS = temp[temp['sv'].str.startswith('G')]
    galileo = temp[temp["sv"].str.startswith('E')]

    """Dane dla satelitów GPS i Galileo"""
    gps_kod_list = GPS[GPS['sTEC'] > 0]['sv'].to_list()
    galileo_kod_list = galileo[galileo['sTEC'] > 0]['sv'].to_list()
    kod_list = gps_kod_list + galileo_kod_list
    data_kod = TEC_data[TEC_data['sv'].isin(kod_list)]

    """Wykres pseudoodległości dla satelitów GPS i Galileo"""

    sv_list_GPS = []
    C1_GPS = []
    C2_GPS = []

    sv_list_Galileo = []
    C1_Galileo = []
    C2_Galileo = []

    for sv in kod_list:
        if sv.startswith('G'):
            data = data_kod.loc[data_kod['sv'] == sv].mean()
            sv_list_GPS.append(sv)
            C1_GPS.append(data[TEC_data.columns[13]] / 1000)
            C2_GPS.append(data[TEC_data.columns[14]] / 1000)
        else:
            data = data_kod.loc[data_kod['sv'] == sv].mean()
            sv_list_Galileo.append(sv)
            C1_Galileo.append(data[TEC_data.columns[13]] / 1000)
            C2_Galileo.append(data[TEC_data.columns[18]] / 1000)

    """inicjalizowanie tworzenia wykresu"""
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=sv_list_GPS,
        y=C1_GPS,
        name='GPS ' + str(TEC_data.columns[13]),
        marker_color='rgb(52, 102, 111)',
    ))
    fig2.add_trace(go.Bar(
        x=sv_list_GPS,
        y=C2_GPS,
        name='GPS '+ str(TEC_data.columns[14]),
        marker_color='rgb(100, 162, 173)',
    ))

    fig2.add_trace(go.Bar(
        x=sv_list_Galileo,
        y=C1_Galileo,
        name='Galileo ' + str(TEC_data.columns[13]),
        marker_color='rgb(52, 102, 111)',
        visible=False
    ))
    fig2.add_trace(go.Bar(
        x=sv_list_Galileo,
        y=C2_Galileo,
        name='Galileo ' + str(TEC_data.columns[18]),
        marker_color='rgb(100, 162, 173)',
        visible=False
    ))

    """właściwości wygenerowanego wykresu"""
    fig2.update_layout(
        yaxis_range=[(data_kod[TEC_data.columns[13]] / 1000).min() - 0.05 * (data_kod[TEC_data.columns[13]] / 1000).min(),
                     (data_kod[TEC_data.columns[13]] / 1000).max() + 0.01 * (data_kod[TEC_data.columns[13]] / 1000).max()],
        yaxis=dict(
            title='pseudoodległość [m]'
        ),
        yaxis_tickformat='m',
        updatemenus=[
            dict(
                type="buttons",
                buttons=list([
                    dict(label="GPS",
                         method="update",
                         args=[{"visible": [True, True, False, False]},
                               {"title": "Wartość pseudoodległości dla systemu GPS na wybranych częstotliwościach"}]),
                    dict(label="Galileo",
                         method="update",
                         args=[{"visible": [False, False, True, True]},
                               {
                                   "title": "Wartość pseudoodległości dla systemu Galileo na wybranych częstotliwościach"}])
                ]),
            )
        ])

    plot_C1P2 = pltoff.plot(fig2, output_type='div')
    # plot_C1P2 = plot(fig2, output_type='div')

    """Wykres różnic w pseudoodległościach"""

    P2C1_GPS = []
    C5C1_Galileo = []

    for sv in kod_list:
        if sv.startswith('G'):
            data = data_kod.loc[data_kod['sv'] == sv].mean()
            P2C1_GPS.append(data['roznica'])
        else:
            data = data_kod.loc[data_kod['sv'] == sv].mean()
            C5C1_Galileo.append(data['roznica'])

    """inicjalizowanie tworzenia wykresu"""
    fig3 = go.Figure()

    fig3.add_trace(go.Bar(
        x=sv_list_GPS,
        y=P2C1_GPS,
        name='P2-C1',
        marker_color='rgb(52, 102, 111)',
    ))

    fig3.add_trace(go.Bar(
        x=sv_list_Galileo,
        y=C5C1_Galileo,
        name='C5-C1',
        marker_color='rgb(100, 162, 173)',
        visible=False
    ))

    """właściwości wygenerowanego wykresu"""
    fig3.update_layout(
        yaxis=dict(
            title='różnica pseudoodległości [m]'
        ),
        yaxis_tickformat='m',
        updatemenus=[
            dict(
                type="buttons",
                buttons=list([
                    dict(label="GPS",
                         method="update",
                         args=[{"visible": [True, False]},
                               {"title": "Różnica pseudoodległości dla systemu GPS na wybranych częstotliwościach"}]),
                    dict(label="Galileo",
                         method="update",
                         args=[{"visible": [False, True]},
                               {
                                   "title": "Różnica pseudoodległości dla systemu Galileo na wybranych częstotliwościach"}])
                ]),
            )
        ])

    plot_roznica = pltoff.plot(fig3, output_type='div')

    return render(request, 'ionosphere_storm/charts.html',
                  context={'plot_C1P2': plot_C1P2, 'plot_roznica': plot_roznica})


def sTEC_charts(request):
    temp_df = TEC_data.copy()
    size_of_groups = temp_df.groupby('sv').size().sort_values(ascending=False)
    size_of_groups = size_of_groups.index.tolist()
    TEC_data['sv'] = pd.Categorical(
        TEC_data['sv'],
        categories=size_of_groups,
        ordered=True)

    """Tymczasowy df =, w których następuje zliczenie danych po kolumnie sv"""
    temp = TEC_data.groupby('sv').apply(lambda x: x.count())

    temp = temp[['sTEC']].reset_index()
    GPS = temp[temp['sv'].str.startswith('G')]
    galileo = temp[temp["sv"].str.startswith('E')]
    gps_kod_list = GPS[GPS['sTEC'] > 0]['sv'].to_list()
    galileo_kod_list = galileo[galileo['sTEC'] > 0]['sv'].to_list()

    kod_list = gps_kod_list + galileo_kod_list
    data_kod = TEC_data[TEC_data['sv'].isin(kod_list)]
    kod_GPS = len(GPS[GPS['sTEC'] > 0])
    kod_galileo = len(galileo[galileo['sTEC'] > 0])

    """Inicjalizowanie tworzenia wykresu"""

    fig = go.Figure()
    i = 0
    for sv in kod_list:
        data = data_kod.loc[data_kod['sv'] == sv]
        if sv.startswith('G'):
            fig.add_trace(go.Scatter(x=data['time_only'], y=data['sTEC'], mode='lines', name=str(data['sv'].iloc[0])))
        else:
            fig.add_trace(go.Scatter(x=data['time_only'], y=data['sTEC'], mode='lines', name=str(data['sv'].iloc[0]),
                                     visible=False))

    """ustawienia widoczności poszczególnych kombinacji"""
    visible = [kod_GPS * [True], kod_galileo * [False]]
    visible1 = [kod_GPS * [False], kod_galileo * [True]]

    merged = list(itertools.chain(*visible))
    merged1 = list(itertools.chain(*visible1))

    """właściwości wygenerowanego wykresu"""
    fig.update_layout(
        yaxis=dict(
            title='wartość sTEC [TECu]'
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=list([
                    dict(label="GPS",
                         method="update",
                         args=[{"visible": merged},
                               {"title": "Wartość sTEC dla systemu GPS na podstawie obserwacji kodowych"}]),
                    dict(label="Galileo",
                         method="update",
                         args=[{"visible": merged1},
                               {"title": "Wartość sTEC dla systemu Galileo na podstawie obserwacji kodowych"}])
                ]),
            )
        ])

    fig.update_xaxes(tickangle=45)

    plot_wykres_zbiorczy = pltoff.plot(fig, output_type='div')

    """WYKRES 2a dla wszystkich epok"""
    data2a = TEC_data.copy()
    data2a['time_only'] = [x.strftime('%H:%M:%S') for x in data2a['time']]
    times = data2a['time_only'].tolist()
    unique_times = list(set(times))
    unique_times.sort()

    selected_times = []
    selected_times.append(unique_times[-1])

    GPS_data2a = data2a[data2a['sv'].str.startswith('G')]
    Galileo_data2a = data2a[data2a['sv'].str.startswith('E')]

    system_list = [GPS_data2a, Galileo_data2a]

    GPS_data2a.name = 'GPS'
    Galileo_data2a.name = 'Galileo'

    Ge = []
    Gn = []
    sTEC_wyrownane = []
    kolumna_system = []

    for system in system_list:
        for times in unique_times:
            data = system.loc[system['time_only'] == times]

            B = np.array([np.cos(data['Azimuth']) * np.tan(data['Elevation']),
                          np.sin(data['Azimuth']) * np.tan(data['Elevation'])]).T
            N = len(data['Azimuth'])
            A = np.c_[np.ones(N), B]
            L = np.array(data['sTEC'])

            x = np.dot(np.dot(inv(np.dot(A.T, A)), A.T), L)

            temp = list(x)
            sTEC_wyrownane.append(temp[0])
            Gn.append(temp[1])
            Ge.append(temp[2])

            if system.name == 'GPS':
                kolumna_system.append("GPS")
            elif system.name == "Galileo":
                kolumna_system.append("Galileo")

    data_wykres_2 = pd.DataFrame(list(zip(sTEC_wyrownane, Gn, Ge, kolumna_system)),
                                 columns=['sTEC_wyrownane', 'Gn', 'Ge', 'system'])
    przedzialy = np.linspace(0, 120, num=120)

    """DLA GPS"""
    data_GPS = data_wykres_2[data_wykres_2['system'] == 'GPS']
    data_GPS['unique_times'] = unique_times

    # DODAWANIE WEKTORÓW
    u1 = [0] * 120
    v1 = (data_GPS['Gn']).to_list()
    u2 = data_GPS['Ge'].to_list()
    v2 = [0] * 120

    u3 = [a + b for a, b in zip(u1, u2)]
    v3 = [a + b for a, b in zip(v1, v2)]

    x = przedzialy
    y1 = data_GPS['sTEC_wyrownane'].values.tolist()

    """DLA GALILEO"""
    data_Galileo = data_wykres_2[data_wykres_2['system'] == 'Galileo']
    data_Galileo['unique_times'] = unique_times

    v4 = (data_Galileo['Gn']).to_list()
    u4 = data_Galileo['Ge'].to_list()

    u5 = [a + b for a, b in zip(u1, u4)]
    v5 = [a + b for a, b in zip(v4, v2)]

    y2 = data_Galileo['sTEC_wyrownane'].values.tolist()

    # quiver plots
    fig1 = ff.create_quiver(x, y1, u3, v3)
    fig2 = ff.create_quiver(x, y2, u5, v5)
    subplots = make_subplots(rows=2,
                             cols=1,
                             subplot_titles=("Dla danych GPS", "Dla danych Galileo"))

    subplots.update_layout(coloraxis={'colorscale': px.colors.sequential.Bluered},
                           title_text="Średnia wartość sTEC z uwzględnieniem gradientu północnego i wschodniego")

    wybrane_przedzialy = przedzialy[::4]
    wybrany_czas = unique_times[::4]

    # add all fig1.data as individual traces in fig at row=1, col=1
    for d in fig1.data:
        subplots.add_trace(go.Scatter(x=d['x'],
                                      y=d['y'],
                                      name="gradient GPS",
                                      showlegend=False),
                           row=1, col=1)
    subplots.add_trace(go.Scatter(x=przedzialy,
                                  y=y1,
                                  mode="markers",
                                  marker_size=5,
                                  marker_color=y1,
                                  showlegend=False,
                                  marker=dict(
                                      color=y1,
                                      coloraxis='coloraxis'
                                  ),
                                  name='sTEC GPS'),
                       row=1, col=1)

    # add all fig2.data as individual traces in fig at row=1, col=2
    for d in fig2.data:
        subplots.add_trace(go.Scatter(x=d['x'],
                                      y=d['y'],
                                      name='gradient Galileo',
                                      showlegend=False),
                           row=2, col=1)
    subplots.add_trace(go.Scatter(x=przedzialy,
                                  y=y2,
                                  mode="markers",
                                  marker_size=5,
                                  marker_color=y2,
                                  showlegend=False,
                                  marker=dict(
                                      color=y2,
                                      coloraxis='coloraxis'
                                  ),
                                  text=y2,
                                  name="sTEC Galileo"),
                       row=2, col=1)

    subplots.update_layout(xaxis1=dict(
        tickvals=wybrane_przedzialy,
        ticktext=wybrany_czas,
        tickangle=45))
    subplots.update_layout(xaxis2=dict(
        tickvals=wybrane_przedzialy,
        ticktext=wybrany_czas,
        tickangle=45))

    subplots['layout']['yaxis']['title'] = 'średnia wartość sTEC [TECu]'
    subplots['layout']['yaxis2']['title'] = 'średnia wartość sTEC [TECu]'

    plot_elewacja_all = pltoff.plot(subplots, output_type='div')

    """WYKRES NR 3"""

    fig3 = go.Figure()
    data2 = TEC_data.copy()

    GPS_data = data2[data2['sv'].str.startswith('G')]
    Galileo_data = data2[data2['sv'].str.startswith('E')]

    for sv in kod_list:
        data = data_kod.loc[data_kod['sv'] == sv]
        if sv.startswith('G'):
            fig3.add_trace(go.Scatter(x=data['Azimuth_deg'],
                                      y=data['Elevation_deg'],
                                      mode='markers',
                                      text=data['sTEC'],
                                      marker=dict(
                                          colorscale=px.colors.sequential.Bluered,
                                          color=data['sTEC'],
                                          showscale=True,
                                          cmin=GPS_data['sTEC'].min(),
                                          cmax=GPS_data['sTEC'].max()),
                                      name=str(data['sv'].iloc[0])))
        else:
            fig3.add_trace(go.Scatter(x=data['Azimuth_deg'],
                                      y=data['Elevation_deg'],
                                      mode='markers',
                                      marker_color=data['sTEC'],
                                      marker=dict(
                                          colorscale=px.colors.sequential.Bluered,
                                          color=data['sTEC'],
                                          showscale=True,
                                          cmin=Galileo_data['sTEC'].min(),
                                          cmax=Galileo_data['sTEC'].max()),
                                      text=data['sTEC'],
                                      name=str(data['sv'].iloc[0]),
                                      visible=False))

    """ustawienia widoczności poszczególnych kombinacji"""
    visible = [kod_GPS * [True], kod_galileo * [False]]
    visible1 = [kod_GPS * [False], kod_galileo * [True]]

    merged = list(itertools.chain(*visible))
    merged1 = list(itertools.chain(*visible1))

    """właściwości wygenerowanego wykresu"""
    fig3.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            title='kąt elewacji [°]'
        ),
        xaxis=dict(
            title='kąt azymutalny [°]'
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=list([
                    dict(label="GPS",
                         method="update",
                         args=[{"visible": merged},
                               {"title": "Zależność wartości sTEC od kąta elewacji i azymutu"}]),
                    dict(label="Galileo",
                         method="update",
                         args=[{"visible": merged1},
                               {"title": "Zależność wartości sTEC od kąta elewacji i azymutu"}])
                ]),
            )
        ])

    fig3.update_xaxes(tickangle=45)

    plot_elewacja_sTEC = pltoff.plot(fig3, output_type='div')

    return render(request, 'ionosphere_storm/sTEC_charts.html',
                  context={'plot_wykres_zbiorczy': plot_wykres_zbiorczy, 'plot_elewacja_all': plot_elewacja_all,
                           'plot_elewacja_sTEC': plot_elewacja_sTEC})


def vTEC_charts(request):
    temp_df = TEC_data.copy()
    size_of_groups = temp_df.groupby('sv').size().sort_values(ascending=False)
    size_of_groups = size_of_groups.index.tolist()
    TEC_data['sv'] = pd.Categorical(
        TEC_data['sv'],
        categories=size_of_groups,
        ordered=True
    )

    temp = TEC_data.groupby('sv').apply(lambda x: x.count())
    temp = temp[['vTEC']].reset_index()
    GPS = temp[temp['sv'].str.startswith('G')]
    galileo = temp[temp["sv"].str.startswith('E')]
    gps_kod_list = GPS[GPS['vTEC'] > 0]['sv'].to_list()
    galileo_kod_list = galileo[galileo['vTEC'] > 0]['sv'].to_list()

    kod_list = gps_kod_list + galileo_kod_list
    data_kod = TEC_data[TEC_data['sv'].isin(kod_list)]
    kod_GPS = len(GPS[GPS['vTEC'] > 0])
    kod_galileo = len(galileo[galileo['vTEC'] > 0])

    """Inicjalizowanie tworzenia wykresu"""

    fig = go.Figure()
    i = 0
    x_min = min(TEC_data['time_only'])
    x_max = max(TEC_data['time_only'])
    for sv in kod_list:
        data = data_kod.loc[data_kod['sv'] == sv]
        if sv.startswith('G'):
            fig.add_trace(go.Scatter(x=data['time_only'], y=data['vTEC'], mode='lines', name=str(data['sv'].iloc[0])))
        else:
            fig.add_trace(go.Scatter(x=data['time_only'], y=data['vTEC'], mode='lines', name=str(data['sv'].iloc[0]),
                                     visible=False))
    fig.update_xaxes(range=[x_min, x_max])
    """ustawienia widoczności poszczególnych kombinacji"""
    visible = [kod_GPS * [True], kod_galileo * [False]]
    visible1 = [kod_GPS * [False], kod_galileo * [True]]

    merged = list(itertools.chain(*visible))
    merged1 = list(itertools.chain(*visible1))

    """właściwości wygenerowanego wykresu"""
    fig.update_layout(
        yaxis=dict(
            title='wartość vTEC [TECu]'
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=list([
                    dict(label="GPS",
                         method="update",
                         args=[{"visible": merged},
                               {"title": "Wartość vTEC dla systemu GPS na podstawie obserwacji kodowych"}]),
                    dict(label="Galileo",
                         method="update",
                         args=[{"visible": merged1},
                               {"title": "Wartość vTEC dla systemu Galileo na podstawie obserwacji kodowych"}])
                ]),
            )
        ])

    fig.update_xaxes(tickangle=45)

    plot_wykres_zbiorczy_vTEC = pltoff.plot(fig, output_type='div')

    """WYKRES 2 dla wybranej ostatniej epoki"""

    data2 = TEC_data.copy()
    data2['time_only'] = [x.strftime('%H:%M:%S') for x in data2['time']]
    times = data2['time_only'].tolist()
    unique_times = list(set(times))
    unique_times.sort()

    selected_times = []
    selected_times.append(unique_times[-1])

    GPS_data = data2[data2['sv'].str.startswith('G')]
    Galileo_data = data2[data2['sv'].str.startswith('E')]

    system_list = [GPS_data, Galileo_data]
    GPS_data.name = "GPS"
    Galileo_data.name = "Galileo"

    TEC_srednie = []
    azymut = []
    elewacja = []

    for system in system_list:
        for times in selected_times:
            data = system.loc[system['time_only'] == times]
            B = np.array([np.cos(data['Azimuth']) / np.tan(data['Elevation']),
                          np.sin(data['Azimuth']) / np.tan(data['Elevation'])]).T
            N = len(data['Azimuth'])
            A = np.c_[np.ones(N), B]
            L = np.array(data['sTEC'])

            x = np.dot(np.dot(inv(np.dot(A.T, A)), A.T), L)

            for az in np.arange(360):
                az = np.radians(az)
                for elev in np.arange(5, 91):
                    elev = np.radians(elev)
                    TEC_temp = x[0] + x[1] * (np.cos(az) / np.tan(elev)) + x[2] * (np.sin(az) / np.tan(elev))
                    #             TEC_srednie.append(TEC_temp*np.sin(elev))
                    if TEC_temp * np.sin(elev) < 0:
                        TEC_srednie.append(0)
                    elif TEC_temp * np.sin(elev):
                        TEC_srednie.append(TEC_temp * np.sin(elev))
                    azymut.append(az)
                    elewacja.append(elev)

    data_wykres = pd.DataFrame(list(zip(TEC_srednie, np.degrees(azymut), np.degrees(elewacja))),
                               columns=['TEC_srednie', 'azymut', 'elewacja'])

    wykres_GPS = data_wykres.iloc[0:30960]
    wykres_Galileo = data_wykres.iloc[30960:61920]

    GPS_df = wykres_GPS.pivot(index='elewacja', columns='azymut', values='TEC_srednie')
    GPS_lista = GPS_df.values.tolist()

    Galileo_df = wykres_Galileo.pivot(index='elewacja', columns='azymut', values='TEC_srednie')
    Galileo_lista = Galileo_df.values.tolist()

    fig2 = make_subplots(rows=2, cols=1, subplot_titles=("Dla obserwacji GPS", "Dla obserwacji Galileo"))
    fig2.add_trace(go.Heatmap(z=GPS_lista, coloraxis="coloraxis", name='vTEC GPS', hovertemplate='kąt azymutalny: %{x}<br>kąt elewacji: %{y}<br>vTEC średnie: %{z}<extra></extra>'), 1, 1)
    fig2.add_trace(go.Heatmap(z=Galileo_lista, coloraxis="coloraxis", name='vTEC Galileo', hovertemplate='kąt azymutalny: %{x}<br>kąt elewacji: %{y}<br>vTEC średnie: %{z}<extra></extra>'), 2, 1)
    fig2.update_layout(coloraxis={'colorscale': px.colors.sequential.Electric[::-1]},
                       title_text="Wartość vTEC średniego obliczonego na podstawie prostego modelu opóźnienia")

    fig2.update_xaxes(tickangle=45)

    fig2['layout']['xaxis']['title'] = 'kąt azymutalny [°]'
    fig2['layout']['xaxis2']['title'] = 'kąt azymutalny [°]'
    fig2['layout']['yaxis']['title'] = 'kąt elewacji [°]'
    fig2['layout']['yaxis2']['title'] = 'kąt elewacji [°]'

    plot_elewacja = pltoff.plot(fig2, output_type='div')

    """WYKRES NR 3"""

    fig3 = go.Figure()
    data3 = TEC_data.copy()
    GPS_data = data3[data3['sv'].str.startswith('G')]
    Galileo_data = data3[data3['sv'].str.startswith('E')]

    for sv in kod_list:
        data = data_kod.loc[data_kod['sv'] == sv]
        if sv.startswith('G'):
            fig3.add_trace(go.Scatter(x=data['Azimuth_deg'],
                                      y=data['Elevation_deg'],
                                      mode='markers',
                                      text=data['vTEC'],
                                      marker=dict(
                                          colorscale=px.colors.sequential.Bluered,
                                          color=data['vTEC'],
                                          showscale=True,
                                          cmin=GPS_data['vTEC'].min(),
                                          cmax=GPS_data['vTEC'].max()),
                                      name=str(data['sv'].iloc[0])))
        else:
            fig3.add_trace(go.Scatter(x=data['Azimuth_deg'],
                                      y=data['Elevation_deg'],
                                      mode='markers',
                                      marker_color=data['vTEC'],
                                      marker=dict(
                                          colorscale=px.colors.sequential.Bluered,
                                          color=data['sTEC'],
                                          showscale=True,
                                          cmin=Galileo_data['vTEC'].min(),
                                          cmax=Galileo_data['vTEC'].max()),
                                      text=data['vTEC'],
                                      name=str(data['sv'].iloc[0]),
                                      visible=False))

    """ustawienia widoczności poszczególnych kombinacji"""
    visible = [kod_GPS * [True], kod_galileo * [False]]
    visible1 = [kod_GPS * [False], kod_galileo * [True]]

    merged = list(itertools.chain(*visible))
    merged1 = list(itertools.chain(*visible1))

    """właściwości wygenerowanego wykresu"""
    fig3.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            title='kąt elewacji [°]'
        ),
        xaxis=dict(
            title='kąt azymutalny [°]'
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=list([
                    dict(label="GPS",
                         method="update",
                         args=[{"visible": merged},
                               {"title": "Zależność wartości vTEC od kąta elewacji i azymutu dla systemu GPS"}]),
                    dict(label="Galileo",
                         method="update",
                         args=[{"visible": merged1},
                               {"title": "Zależność wartości vTEC od kąta elewacji i azymutu dla systemu Galileo"}])
                ]),
            )
        ])

    fig3.update_xaxes(tickangle=45)

    plot_elewacja_vTEC = pltoff.plot(fig3, output_type='div')

    return render(request, 'ionosphere_storm/vTEC_charts.html',
                  context={'plot_wykres_zbiorczy_vTEC': plot_wykres_zbiorczy_vTEC,
                           'plot_elewacja': plot_elewacja,
                           'plot_elewacja_vTEC': plot_elewacja_vTEC})


def vTec_map(request):
    data, godzina, minuta, sekundy = date_and_time()

    """wybieranie zakresu """

    min_time = min(TEC_data['time_only'])
    max_time = max(TEC_data['time_only'])

    """zakres wyswietlania ikon na pasku - 120 epok"""
    zakres = pd.date_range(min_time, max_time, freq="30S").time

    zakres_keys = list(np.arange(len(zakres)))
    zakres_str = [str(godzina) for godzina in zakres]

    """Podział zakresu na 5 części"""
    ile_czesci = 10
    podzial = len(zakres) // ile_czesci
    selected_keys = zakres_keys[::podzial]
    # selected_values = zakres[::podzial]
    #

    """tworzenie słownika {0: '17:00:00' ...}"""
    new_mark_values = dict(zip(zakres_keys, zakres_str))

    app = DjangoDash('vTEC_map')
    app.layout = html.Div([
        html.Div([
            html.Pre(
                style={"text-align": "center", "font-size": "100%", "color": "black"})
        ]),

        html.Div([
            dcc.Graph(id='the_graph', figure={'layout': {"height": 800}})
        ]),

        html.Div([
            dcc.RangeSlider(id='the_hour',
                            min=0,
                            max=int(len(zakres)) - 1,
                            value=[0, 119],
                            marks={
                                int(zakres_keys[0]): {'label': zakres_str[int(zakres_keys[0])],
                                                      'style': {'color': '#99d6ff'}},
                                int(selected_keys[1]): {'label': zakres_str[int(zakres_keys[selected_keys[1]])],
                                                        'style': {'color': '#80ccff'}},
                                int(selected_keys[2]): {'label': zakres_str[int(zakres_keys[selected_keys[2]])],
                                                        'style': {'color': '#4db8ff'}},
                                int(selected_keys[3]): {'label': zakres_str[int(zakres_keys[selected_keys[3]])],
                                                        'style': {'color': '#33adff'}},
                                int(selected_keys[4]): {'label': zakres_str[int(zakres_keys[selected_keys[4]])],
                                                        'style': {'color': '#0099ff'}},
                                int(selected_keys[5]): {'label': zakres_str[int(zakres_keys[selected_keys[5]])],
                                                        'style': {'color': '#008ae6'}},
                                int(selected_keys[6]): {'label': zakres_str[int(zakres_keys[selected_keys[6]])],
                                                        'style': {'color': '#007acc'}},
                                int(selected_keys[7]): {'label': zakres_str[int(zakres_keys[selected_keys[7]])],
                                                        'style': {'color': '#005c99'}},
                                int(selected_keys[8]): {'label': zakres_str[int(zakres_keys[selected_keys[8]])],
                                                        'style': {'color': '#004d80'}},
                                int(selected_keys[9]): {'label': zakres_str[int(zakres_keys[selected_keys[9]])],
                                                        'style': {'color': '#003d66'}},
                                int(zakres_keys[-1]): {'label': zakres_str[int(zakres_keys[-1])],
                                                       'style': {'color': '#002e4d'}}})
        ], style={"width": "70%", "position": "absolute",
                  "left": "5%"})

    ])

    @app.callback(
        Output('the_graph', 'figure'),
        [Input('the_hour', 'value')])
    def update_graph(hours_chosen):
        df = TEC_data.copy()
        df['time_only'] = pd.to_datetime(df['time_only'], format='%H:%M:%S').dt.time

        start = datetime.strptime(new_mark_values[int(hours_chosen[0])], '%H:%M:%S').time()
        endd = datetime.strptime(new_mark_values[int(hours_chosen[1])], '%H:%M:%S').time()
        mask = (df['time_only'] >= start) & (df['time_only'] <= endd)
        aaa = df.loc[mask]

        scatterplot = go.Figure()
        scatterplot.add_trace(go.Scattermapbox(
            mode="markers",
            # name=str(TEC_data['sv']),
            lat=aaa['IPP_phi'],
            lon=aaa['IPP_lambda'],
            text=aaa['sv'].astype(str) + '<br>' + aaa['vTEC'].astype(str),
            hoverinfo=('text'),
            # line=go.scattermapbox.Line(),
            marker=go.scattermapbox.Marker(
                size=7,
                color=aaa['vTEC'],
                colorscale=px.colors.diverging.RdYlGn[::-1],
                showscale=True,
                colorbar=dict(
                    tickvals=[0, 10, 20, 30, 40, 50],
                    ticktext=["0", "NISKA [0-10 TECU] ", "ŚREDNIA [10-20 TECU]", "WYSOKA [20-30 TECU]",
                              "BARDZO WYSOKA [30-40 TECU]", "EKSTREMALNA [40-50 TECU]"],
                ),
                cmax=50,
                cmin=0),
        ))

        scatterplot.update_layout(
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                )),
            margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
            mapbox={
                'center': {'lon': 17, 'lat': 51},
                'style': "open-street-map",
                # 'style': "stamen-terrain",
                'center': {'lon': 17, 'lat': 51},
                'zoom': 3},
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01),
        ),
        return scatterplot

    return render(request, 'ionosphere_storm/vTec_map.html',
                  context={'data': data, 'h': godzina, 'min': minuta, 's': sekundy})


def sTec_map(request):
    data, godzina, minuta, sekundy = date_and_time()
    """wybieranie zakresu """

    min_time = min(TEC_data['time_only'])
    max_time = max(TEC_data['time_only'])

    """zakres wyswietlania ikon na pasku - 120 epok"""
    zakres = pd.date_range(min_time, max_time, freq="30S").time

    zakres_keys = list(np.arange(len(zakres)))
    zakres_str = [str(godzina) for godzina in zakres]

    """Podział zakresu na 5 części"""
    ile_czesci = 10
    podzial = len(zakres) // ile_czesci
    # selected_keys = zakres_keys[::podzial]
    selected_keys = zakres_keys[::podzial]
    # selected_values = zakres[::podzial]
    #

    """tworzenie słownika {0: '17:00:00' ...}"""
    new_mark_values = dict(zip(zakres_keys, zakres_str))

    app = DjangoDash('sTEC_map')
    app.layout = html.Div([
        html.Div([
            html.Pre(
                style={"text-align": "center", "font-size": "100%", "color": "black"})
        ]),

        html.Div([
            dcc.Graph(id='the_graph', figure={'layout': {"height": 800}})
        ]),

        html.Div([
            dcc.RangeSlider(id='the_hour',
                            min=0,
                            max=int(len(zakres)) - 1,
                            # tooltip={'always_visible': False, 'placement': 'bottom'},
                            # updatemode="drag",
                            value=[0, 119],
                            marks={
                                int(zakres_keys[0]): {'label': zakres_str[int(zakres_keys[0])],
                                                      'style': {'color': '#99d6ff'}},
                                int(selected_keys[1]): {'label': zakres_str[int(zakres_keys[selected_keys[1]])],
                                                        'style': {'color': '#80ccff'}},
                                int(selected_keys[2]): {'label': zakres_str[int(zakres_keys[selected_keys[2]])],
                                                        'style': {'color': '#4db8ff'}},
                                int(selected_keys[3]): {'label': zakres_str[int(zakres_keys[selected_keys[3]])],
                                                        'style': {'color': '#33adff'}},
                                int(selected_keys[4]): {'label': zakres_str[int(zakres_keys[selected_keys[4]])],
                                                        'style': {'color': '#0099ff'}},
                                int(selected_keys[5]): {'label': zakres_str[int(zakres_keys[selected_keys[5]])],
                                                        'style': {'color': '#008ae6'}},
                                int(selected_keys[6]): {'label': zakres_str[int(zakres_keys[selected_keys[6]])],
                                                        'style': {'color': '#007acc'}},
                                int(selected_keys[7]): {'label': zakres_str[int(zakres_keys[selected_keys[7]])],
                                                        'style': {'color': '#005c99'}},
                                int(selected_keys[8]): {'label': zakres_str[int(zakres_keys[selected_keys[8]])],
                                                        'style': {'color': '#004d80'}},
                                int(selected_keys[9]): {'label': zakres_str[int(zakres_keys[selected_keys[9]])],
                                                        'style': {'color': '#003d66'}},
                                int(zakres_keys[-1]): {'label': zakres_str[int(zakres_keys[-1])],
                                                       'style': {'color': '#002e4d'}}})
        ], style={"width": "70%", "position": "absolute",
                  "left": "5%"})

    ])

    @app.callback(
        Output('the_graph', 'figure'),
        [Input('the_hour', 'value')])
    def update_graph(hours_chosen):
        df = TEC_data.copy()
        df['time_only'] = pd.to_datetime(df['time_only'], format='%H:%M:%S').dt.time

        poczatek = new_mark_values[hours_chosen[0]]
        koniec = new_mark_values[hours_chosen[1]]

        start = datetime.strptime(new_mark_values[int(hours_chosen[0])], '%H:%M:%S').time()
        endd = datetime.strptime(new_mark_values[int(hours_chosen[1])], '%H:%M:%S').time()
        mask = (df['time_only'] >= start) & (df['time_only'] <= endd)
        aaa = df.loc[mask]

        scatterplot = go.Figure()
        scatterplot.add_trace(go.Scattermapbox(
            mode="markers",
            lat=aaa['IPP_phi'],
            lon=aaa['IPP_lambda'],
            text=aaa['sv'].astype(str) + '<br>' + aaa['sTEC'].astype(str),
            hoverinfo=('text'),
            marker=go.scattermapbox.Marker(
                size=7,
                color=aaa['sTEC'],
                colorscale=px.colors.diverging.RdYlGn[::-1],
                showscale=True,
                colorbar=dict(
                    tickvals=[0, 20, 40, 60, 80, 100],
                    ticktext=["0", "NISKA [0-20 TECU] ", "ŚREDNIA [20-40 TECU]", "WYSOKA [40-60 TECU]",
                              "BARDZO WYSOKA [60-80 TECU]", "EKSTREMALNA [80-100 TECU]"],
                ),
                cmax=100,
                cmin=0),
        ))

        scatterplot.update_layout(
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                )),
            margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
            mapbox={
                'center': {'lon': 17, 'lat': 51},
                'style': "open-street-map",
                'center': {'lon': 17, 'lat': 51},
                'zoom': 3},
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01),
        ),
        return scatterplot

    return render(request, 'ionosphere_storm/sTec_map.html',
                  context={'data': data, 'h': godzina, 'min': minuta, 's': sekundy})
