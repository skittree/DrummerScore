from django.contrib import admin
from .models import Song, Note

# Register your models here.
admin.site.register([Song, Note])