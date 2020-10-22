from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views import View
# Create your views here.
def board(request):
    return render(request, 'board/board.html')