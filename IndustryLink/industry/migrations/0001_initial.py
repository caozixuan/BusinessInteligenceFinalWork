# -*- coding: utf-8 -*-
# Generated by Django 1.10.3 on 2018-11-18 01:13
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Company',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Dictionary',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('is_industry', models.BooleanField(default=False)),
                ('name', models.CharField(max_length=100)),
                ('field', models.CharField(blank=True, max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Divided',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('is_industry', models.BooleanField(default=False)),
                ('name', models.CharField(max_length=100)),
                ('field', models.CharField(blank=True, max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Industry',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('upstream', models.CharField(max_length=100, null=True)),
                ('downstream', models.CharField(max_length=100, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Sentence',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='StopWord',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.AddField(
            model_name='company',
            name='down_link',
            field=models.ManyToManyField(null=True, related_name='down', to='industry.Industry'),
        ),
        migrations.AddField(
            model_name='company',
            name='mid_link',
            field=models.ManyToManyField(null=True, related_name='mid', to='industry.Industry'),
        ),
        migrations.AddField(
            model_name='company',
            name='up_link',
            field=models.ManyToManyField(null=True, related_name='up', to='industry.Industry'),
        ),
    ]
