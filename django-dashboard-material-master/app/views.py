# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template
from django.http import JsonResponse

from django.http import StreamingHttpResponse

import pandas as pd
import numpy as np
import json
import os





def get_suggestions(request, state, team):

        err = "no"
        print("EEEE")
        context = { "errors":err,"models":team, "states": state}
        return JsonResponse(context)
    


#@login_required(login_url="/login/")
def index(request):

    context = {}
    context['segment'] = 'index'

    return render(request, 'index.html',context)

#@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template      = request.path.split('/')[-1]
        context['segment'] = load_template
        context = {}
        context['segment'] = 'data'




        html_template = loader.get_template( load_template )
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:

        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))
