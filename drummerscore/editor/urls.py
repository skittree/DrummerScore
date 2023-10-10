from django.urls import path

from . import views

urlpatterns = [
    path('upload_mp3/', views.upload_mp3, name='upload_mp3'),
]