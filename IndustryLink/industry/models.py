from __future__ import unicode_literals

from django.db import models


class Industry(models.Model):
    name = models.CharField(max_length=100)
    upstream = models.CharField(max_length=100,null=True)
    downstream = models.CharField(max_length=100,null=True)


class Company(models.Model):
    name = models.CharField(max_length=100)
    up_link = models.ManyToManyField(Industry, null=True,related_name='up')
    down_link = models.ManyToManyField(Industry, null=True,related_name='down')
    mid_link = models.ManyToManyField(Industry, null=True,related_name='mid')


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


class Sentence(models.Model):
    content = models.TextField()


class Word(models.Model):
    sentence_id = models.IntegerField(null=False)
    name=models.CharField(max_length=100)
    div_type=models.IntegerField(null=True,default=0)


class WordTest(models.Model):
    sentence_id = models.IntegerField(null=False)
    name=models.CharField(max_length=100)
    div_type=models.IntegerField(null=True,default=0)


class Context(models.Model):
    entity = models.CharField(max_length=100)
    div_type = models.IntegerField(null=True, default=0)
    up1=models.CharField(max_length=100)
    up2 = models.CharField(max_length=100)
    up3 = models.CharField(max_length=100)
    up4 = models.CharField(max_length=100)
    up5 = models.CharField(max_length=100)
    down1=models.CharField(max_length=100)
    down2 = models.CharField(max_length=100)
    down3 = models.CharField(max_length=100)
    down4 = models.CharField(max_length=100)
    down5 = models.CharField(max_length=100)


class ContextTest(models.Model):
    entity = models.CharField(max_length=100)
    div_type = models.IntegerField(null=True, default=0)
    up1=models.CharField(max_length=100)
    up2 = models.CharField(max_length=100)
    up3 = models.CharField(max_length=100)
    up4 = models.CharField(max_length=100)
    up5 = models.CharField(max_length=100)
    down1=models.CharField(max_length=100)
    down2 = models.CharField(max_length=100)
    down3 = models.CharField(max_length=100)
    down4 = models.CharField(max_length=100)
    down5 = models.CharField(max_length=100)


class Article(models.Model):
    entity = models.CharField(max_length=100)
    div_type = models.IntegerField(null=True, default=0)
    position = models.CharField(max_length=50)
    up1=models.CharField(max_length=100)
    up2 = models.CharField(max_length=100)
    up3 = models.CharField(max_length=100)
    up4 = models.CharField(max_length=100)
    up5 = models.CharField(max_length=100)
    down1=models.CharField(max_length=100)
    down2 = models.CharField(max_length=100)
    down3 = models.CharField(max_length=100)
    down4 = models.CharField(max_length=100)
    down5 = models.CharField(max_length=100)