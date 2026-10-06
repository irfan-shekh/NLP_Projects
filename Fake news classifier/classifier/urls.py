from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_news, name='predict_news'),
    path('about/', views.about, name='about'),
]
