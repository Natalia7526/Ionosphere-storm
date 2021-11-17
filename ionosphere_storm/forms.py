from django import forms
from .models import *

# Create your models here.
GPSsv_CHOICES = (
    ('1', 'G01'),
    ('2', 'G02'),
    ('3', 'G03'),
    ('4', 'G04'),
    ('5', 'G05'),
)

class SV_Form(forms.Form):
    Identyfikator_SV = forms.ChoiceField(choices=GPSsv_CHOICES)

