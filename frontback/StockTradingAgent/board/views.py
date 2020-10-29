from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views import View
# Create your views here.
def board(request):
    return render(request, 'board/board.html')

def viewChart(request):
    return render(request,'board/StockChart.html', )