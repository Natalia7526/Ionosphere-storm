from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('plot', views.plot, name='plot'),
    path('vTec_map', views.vTec_map, name='vTec_map'),
    path('sTec_map', views.sTec_map, name='sTec_map'),
    path('stations', views.stations, name='stations'),
    path('about', views.about, name='about'),
    path('storm', views.storm, name='storm'),
    path('charts', views.charts, name='charts'),
    path('sTEC_charts', views.sTEC_charts, name='sTEC_charts'),
    path('vTEC_charts', views.vTEC_charts, name='vTEC_charts'),

]