
from django.db import models

class Image(models.Model):
    name = models.CharField(max_length=100)
    src = models.FileField(blank=True, null=True)

    def __str__(self):
        return self.name