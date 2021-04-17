# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib import admin
from django.urls import path, include  # add this
from app import views 
from django.urls import path, re_path

urlpatterns = [
    path('admin/', admin.site.urls),          # Django admin route
    path("", include("authentication.urls")), # Auth routes - login / register
    path("", include("app.urls")),         # UI Kits Html files
    path('get/ajax/emotion/<str:lyrics>', views.get_emotion, name='get_emotion')

]
