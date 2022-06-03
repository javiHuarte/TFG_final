from django import forms
from .models import Parasite

class Parasiteform(forms.ModelForm):
    class Meta:
        model = Parasite
        #fields =['name', 'image']
        fields = ['image'] 
         
        
