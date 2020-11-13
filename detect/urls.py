from django.urls import path
from .views import ExtractAreaView
urlpatterns = [
  path('ExtractSignArea', ExtractAreaView.as_view())
]