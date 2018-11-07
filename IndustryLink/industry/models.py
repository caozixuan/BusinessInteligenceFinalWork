from __future__ import unicode_literals

from django.db import models


class Dictionary(models.Model):
    is_industry = models.BooleanField(default=False)
    name = models.CharField(max_length=100)
    field = models.CharField(max_length=100,blank=True)


class Divided(models.Model):
    is_industry = models.BooleanField(default=False)
    name = models.CharField(max_length=100)
    field = models.CharField(max_length=100,blank=True)


class StopWord(models.Model):
    name = models.CharField(max_length=100)