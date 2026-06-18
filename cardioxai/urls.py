from django.contrib import admin
from django.urls import path
from core import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.landing, name="landing"),
    path("about/", views.about, name="about"),
    path("assess/", views.assess, name="assess"),
    path("results/", views.results, name="results"),
    path("demo/", views.demo_results, name="demo"),
    path("api/predict/", views.api_predict, name="api_predict"),
]
