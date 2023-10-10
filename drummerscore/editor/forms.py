from django import forms

class UploadMP3Form(forms.Form):
    mp3_file = forms.FileField()