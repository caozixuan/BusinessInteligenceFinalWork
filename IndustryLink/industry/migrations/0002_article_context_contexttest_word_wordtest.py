# -*- coding: utf-8 -*-
# Generated by Django 1.10.3 on 2018-11-22 06:59
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('industry', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Article',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('entity', models.CharField(max_length=100)),
                ('div_type', models.IntegerField(default=0, null=True)),
                ('position', models.CharField(max_length=50)),
                ('up1', models.CharField(max_length=100)),
                ('up2', models.CharField(max_length=100)),
                ('up3', models.CharField(max_length=100)),
                ('up4', models.CharField(max_length=100)),
                ('up5', models.CharField(max_length=100)),
                ('down1', models.CharField(max_length=100)),
                ('down2', models.CharField(max_length=100)),
                ('down3', models.CharField(max_length=100)),
                ('down4', models.CharField(max_length=100)),
                ('down5', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Context',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('entity', models.CharField(max_length=100)),
                ('div_type', models.IntegerField(default=0, null=True)),
                ('up1', models.CharField(max_length=100)),
                ('up2', models.CharField(max_length=100)),
                ('up3', models.CharField(max_length=100)),
                ('up4', models.CharField(max_length=100)),
                ('up5', models.CharField(max_length=100)),
                ('down1', models.CharField(max_length=100)),
                ('down2', models.CharField(max_length=100)),
                ('down3', models.CharField(max_length=100)),
                ('down4', models.CharField(max_length=100)),
                ('down5', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='ContextTest',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('entity', models.CharField(max_length=100)),
                ('div_type', models.IntegerField(default=0, null=True)),
                ('up1', models.CharField(max_length=100)),
                ('up2', models.CharField(max_length=100)),
                ('up3', models.CharField(max_length=100)),
                ('up4', models.CharField(max_length=100)),
                ('up5', models.CharField(max_length=100)),
                ('down1', models.CharField(max_length=100)),
                ('down2', models.CharField(max_length=100)),
                ('down3', models.CharField(max_length=100)),
                ('down4', models.CharField(max_length=100)),
                ('down5', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Word',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sentence_id', models.IntegerField()),
                ('name', models.CharField(max_length=100)),
                ('div_type', models.IntegerField(default=0, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='WordTest',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sentence_id', models.IntegerField()),
                ('name', models.CharField(max_length=100)),
                ('div_type', models.IntegerField(default=0, null=True)),
            ],
        ),
    ]
