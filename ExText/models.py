from django.db import models

# Create your models here.

from django.db import models
class Extract(models.Model):
    title = models.CharField(max_length=100)
    content = models.FileField(blank=True, null=True)

    def __str__(self):
        return self.title
