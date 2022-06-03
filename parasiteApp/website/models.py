from django.db import models

# Create your models here.

class ParasiteManager(models.Manager):
  def create_parasite(self, image):
    parasite = self.create(image = image)
    return parasite

class Parasite(models.Model):
   # name = models.CharField(max_length=100)
    image = models.ImageField(null=True, upload_to="images/")
    objects = ParasiteManager()

   # def __str__(self):
     #   return self.name
     