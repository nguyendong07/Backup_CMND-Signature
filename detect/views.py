from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
# Create your views here.
from .models import Image
from .serializer import ImageSerializer
from .module import main

class ExtractAreaView(APIView):
    def get(self, request):
        image = Image.objects.all()
        serializer = ImageSerializer(data=image, many=True)
        if serializer.is_valid():
            print('serializer.data')
        print(serializer.data)
        return Response({'message': 'ok', 'data': serializer.data}, status=status.HTTP_200_OK)

    def post(self, request):
        files = request.FILES.get('src')
        res = main(files)
        return Response(res, status=status.HTTP_200_OK)