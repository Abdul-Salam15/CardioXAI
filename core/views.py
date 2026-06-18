import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from ml_pipeline.inference import predict, get_metrics
from ml_pipeline.knowledge_graph import get_graph_data


def landing(request):
    return render(request, 'core/landing.html')


def about(request):
    metrics = get_metrics()
    return render(request, 'core/about.html', {'metrics': json.dumps(metrics)})


def assess(request):
    return render(request, 'core/assess.html')


def results(request):
    if request.method == 'POST':
        form_data = {}
        for key in ['age', 'sex', 'bmi', 'genHealth']:
            form_data[key] = request.POST.get(key, '')
        if form_data['bmi']:
            form_data['bmi'] = float(form_data['bmi'])

        bool_fields = [
            'highBP', 'highChol', 'diabetes', 'stroke', 'diffWalk',
            'smoker', 'heavyAlcohol', 'physAct', 'fruits', 'veggies', 'chestPain'
        ]
        for field in bool_fields:
            val = request.POST.get(field, '')
            form_data[field] = val == 'true' or val == 'Yes' or val == '1'

        for field in ['mentalHealth', 'physHealth', 'sleep']:
            val = request.POST.get(field, '0')
            form_data[field] = int(val) if val else 0

        prediction = predict(form_data)
        prediction['form_data'] = form_data
        request.session['prediction'] = prediction
        return redirect('results')

    prediction = request.session.get('prediction')
    if not prediction:
        return redirect('assess')

    graph_data = get_graph_data(prediction.get('shap_values'))

    return render(request, 'core/results.html', {
        'prediction': prediction,
        'prediction_json': prediction,
        'graph_data_json': graph_data,
    })


def demo_results(request):
    form_data = {
        'age': '55-59', 'sex': 'Male', 'bmi': 28.4, 'genHealth': 'Fair',
        'highBP': True, 'highChol': True, 'diabetes': False, 'stroke': False,
        'diffWalk': False, 'smoker': False, 'heavyAlcohol': False,
        'physAct': False, 'fruits': False, 'veggies': True, 'chestPain': True,
        'mentalHealth': 4, 'physHealth': 6, 'sleep': 6
    }
    prediction = predict(form_data)
    prediction['form_data'] = form_data
    request.session['prediction'] = prediction
    return redirect('results')


@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    try:
        body = json.loads(request.body)
        form_data = body.get('form_data', body)
        prediction = predict(form_data)
        prediction['form_data'] = form_data
        return JsonResponse(prediction)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
