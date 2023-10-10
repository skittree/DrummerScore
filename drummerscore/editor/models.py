from django.db import models
from django.contrib.auth.models import User

def user_directory_path(instance, filename):
    return "uploads/user_{0}/{1}".format(instance.owner.id, filename)

# Create your models here.
class Song(models.Model):
    title = models.CharField(max_length=100)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    mp3 = models.FileField(upload_to=user_directory_path)

class Note(models.Model):
    song = models.ForeignKey(Song, on_delete=models.CASCADE)
    timestamp = models.FloatField()
    duration = models.FloatField()
    velocity = models.IntegerField()
    pitch = models.IntegerField()