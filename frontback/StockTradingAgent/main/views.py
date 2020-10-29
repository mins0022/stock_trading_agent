from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import messages
# Create your views here.

def main(request):
    return render(request, 'main/index.html')

