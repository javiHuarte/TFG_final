from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('showImage/', views.showImage, name='showImage'),
    
    
]