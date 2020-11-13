from django.shortcuts import render

# Create your views here.
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Extract
from .serializer import ExtractImage
from .main_module import Main

class ImageView(APIView):
    def get(self, request):
        image = Extract.objects.all()
        serializer = ExtractImage(data=image, many=True)
        if serializer.is_valid():
            print('serializer.data')
        print(serializer.data)
        return Response({'message': 'ok', 'data': serializer.data}, status=status.HTTP_200_OK)

    def post(self, request):
        path = request.FILES.get('content')
        res = Main(path)
        return Response(res, status=status.HTTP_200_OK)