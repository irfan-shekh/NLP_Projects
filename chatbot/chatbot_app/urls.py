from django.urls import path
from . import views

app_name = 'chatbot_app'

urlpatterns = [
    path('', views.home, name='home'),
    path('chat/', views.chat, name='chat'),
    path('train/', views.train_model, name='train_model'),
    path('status/', views.model_status, name='model_status'),
]
