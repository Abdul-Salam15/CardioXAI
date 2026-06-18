"""
Inference engine: loads trained models, runs ensemble prediction,
generates SHAP + LIME explanations, computes confidence tiers.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_store')

_cache = {}

FEATURE_DISPLAY = {
    'HighBP': 'High Blood Pressure',
    'HighChol': 'High Cholesterol',
    'BMI': 'BMI',
    'Smoker': 'Smoker',
    'Stroke': 'History of Stroke',
    'Diabetes': 'Diabetes',
    'PhysActivity': 'Physical Activity',
    'Fruits': 'Eats Fruit Daily',
    'Veggies': 'Eats Vegetables Daily',
    'HvyAlcoholConsump': 'Heavy Alcohol Use',
    'GenHlth': 'General Health',
    'MentHlth': 'Mental Health Days',
    'PhysHlth': 'Physical Health Days',
    'DiffWalk': 'Difficulty Walking',
    'Sex': 'Sex',
    'Age': 'Age',
}


def _load():
    if _cache:
        return
    _cache['feature_names'] = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
    _cache['ensemble_info'] = joblib.load(os.path.join(MODELS_DIR, 'ensemble_info.joblib'))

    for name in ['lr', 'rf', 'xgb']:
        path = os.path.join(MODELS_DIR, f'{name}_model.joblib')
        if os.path.exists(path):
            _cache[name] = joblib.load(path)

    with open(os.path.join(MODELS_DIR, 'metrics.json')) as f:
        _cache['metrics'] = json.load(f)

    bg_path = os.path.join(MODELS_DIR, 'bg_sample.joblib')
    if os.path.exists(bg_path):
        bg_data = joblib.load(bg_path)
    else:
        splits = joblib.load(os.path.join(MODELS_DIR, 'splits.joblib'))
        bg_data = splits['X_train'].values[:5000]

    _cache['lr_explainer'] = shap.LinearExplainer(_cache['lr'], bg_data[:2000])
    _cache['xgb_explainer'] = shap.TreeExplainer(_cache['xgb'])

    _cache['lime_explainer'] = LimeTabularExplainer(
        bg_data,
        feature_names=_cache['feature_names'],
        class_names=['No CHD', 'CHD'],
        mode='classification',
        random_state=42
    )


def predict(form_data):
    _load()
    feature_names = _cache['feature_names']
    ensemble_info = _cache['ensemble_info']

    x_dict = form_to_features(form_data)
    X = pd.DataFrame([x_dict], columns=feature_names)

    probas = {}
    preds = {}
    for name in ['lr', 'rf', 'xgb']:
        model = _cache[name]
        p = model.predict_proba(X)[0, 1]
        probas[name] = float(round(p, 4))
        preds[name] = int(p >= 0.5)

    top_two = ensemble_info['models']
    weights = ensemble_info['weights']
    ensemble_proba = sum(weights[m] * probas[m] for m in top_two)
    ensemble_pred = int(ensemble_proba >= 0.5)

    risk_score = round(ensemble_proba * 100, 1)
    if risk_score < 30:
        risk_band = 'LOW'
    elif risk_score <= 60:
        risk_band = 'MODERATE'
    else:
        risk_band = 'HIGH'

    disagreement = abs(probas[top_two[0]] - probas[top_two[1]])
    entropy = -ensemble_proba * np.log2(max(ensemble_proba, 1e-10)) - (1 - ensemble_proba) * np.log2(max(1 - ensemble_proba, 1e-10))

    if disagreement > 0.25 or entropy > 0.95:
        confidence_tier = 'low'
    elif disagreement > 0.15 or entropy > 0.85:
        confidence_tier = 'medium'
    else:
        confidence_tier = 'high'

    shap_values = _compute_shap(X)
    lime_explanation = _compute_lime(X)

    recommendations = _generate_recommendations(shap_values, form_data)

    return {
        'risk_score': risk_score,
        'risk_band': risk_band,
        'confidence_tier': confidence_tier,
        'disagreement': round(disagreement, 4),
        'entropy': round(entropy, 4),
        'model_probas': probas,
        'model_preds': preds,
        'ensemble_proba': round(ensemble_proba, 4),
        'ensemble_pred': ensemble_pred,
        'shap_values': shap_values,
        'lime_explanation': lime_explanation,
        'recommendations': recommendations,
    }


def form_to_features(form_data):
    feature_names = _cache['feature_names']

    age_map = {
        '18-24': 1, '25-29': 2, '30-34': 3, '35-39': 4, '40-44': 5,
        '45-49': 6, '50-54': 7, '55-59': 8, '60-64': 9, '65-69': 10,
        '70-74': 11, '75-79': 12, '80+': 13
    }
    health_map = {'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}
    yn = lambda v: 1.0 if v else 0.0

    x = {
        'HighBP': yn(form_data.get('highBP')),
        'HighChol': yn(form_data.get('highChol')),
        'BMI': float(form_data.get('bmi', 25)),
        'Smoker': yn(form_data.get('smoker')),
        'Stroke': yn(form_data.get('stroke')),
        'Diabetes': yn(form_data.get('diabetes')),
        'PhysActivity': yn(form_data.get('physAct')),
        'Fruits': yn(form_data.get('fruits')),
        'Veggies': yn(form_data.get('veggies')),
        'HvyAlcoholConsump': yn(form_data.get('heavyAlcohol')),
        'GenHlth': float(health_map.get(form_data.get('genHealth', 'Good'), 3)),
        'MentHlth': float(form_data.get('mentalHealth', 0)),
        'PhysHlth': float(form_data.get('physHealth', 0)),
        'DiffWalk': yn(form_data.get('diffWalk')),
        'Sex': 1.0 if form_data.get('sex') == 'Male' else 0.0,
        'Age': float(age_map.get(form_data.get('age', '45-49'), 6)),
    }
    return {k: x.get(k, 0.0) for k in feature_names}


def _compute_shap(X):
    _load()
    feature_names = _cache['feature_names']

    xgb_shap = _cache['xgb_explainer'].shap_values(X)
    if isinstance(xgb_shap, list):
        xgb_shap = xgb_shap[1]
    xgb_vals = xgb_shap[0] if len(xgb_shap.shape) > 1 else xgb_shap

    lr_shap = _cache['lr_explainer'].shap_values(X)
    if isinstance(lr_shap, list):
        lr_shap = lr_shap[1] if len(lr_shap) > 1 else lr_shap[0]
    lr_vals = lr_shap[0] if len(lr_shap.shape) > 1 else lr_shap

    weights = _cache['ensemble_info']['weights']
    top_two = _cache['ensemble_info']['models']
    combined = np.zeros(len(feature_names))
    for name, vals in [('xgb', xgb_vals), ('lr', lr_vals)]:
        if name in top_two:
            combined += weights[name] * np.array(vals)

    result = []
    for i, fname in enumerate(feature_names):
        val = float(combined[i])
        display = FEATURE_DISPLAY.get(fname, fname)
        result.append({
            'feature': fname,
            'display_name': display,
            'shap_value': round(val, 4),
            'direction': 'risk' if val > 0 else 'protective',
            'abs_value': round(abs(val), 4),
        })

    result.sort(key=lambda x: x['abs_value'], reverse=True)
    return result[:8]


def _compute_lime(X):
    _load()
    ensemble_info = _cache['ensemble_info']
    top_two = ensemble_info['models']
    weights = ensemble_info['weights']

    def ensemble_predict_proba(X_input):
        p1 = _cache[top_two[0]].predict_proba(X_input)
        p2 = _cache[top_two[1]].predict_proba(X_input)
        return weights[top_two[0]] * p1 + weights[top_two[1]] * p2

    exp = _cache['lime_explainer'].explain_instance(
        X.values[0],
        ensemble_predict_proba,
        num_features=8,
        num_samples=500
    )

    feature_names = _cache['feature_names']
    result = []
    for feat_idx, weight in exp.as_list():
        result.append({
            'feature_desc': feat_idx,
            'weight': round(weight, 4),
            'direction': 'risk' if weight > 0 else 'protective',
        })

    return result


CLINICAL_NOTES = {
    'HighBP': 'Hypertension is one of the leading modifiable risk factors for coronary heart disease.',
    'HighChol': 'Elevated cholesterol promotes arterial plaque formation, narrowing blood flow.',
    'BMI': 'Higher BMI increases cardiac workload and metabolic strain on the cardiovascular system.',
    'Smoker': 'Smoking damages blood vessels and significantly accelerates atherosclerosis.',
    'Stroke': 'History of stroke indicates existing vascular damage and elevated cardiovascular risk.',
    'Diabetes': 'Diabetes damages blood vessels over time and doubles the risk of heart disease.',
    'PhysActivity': 'Regular physical activity strengthens the heart and lowers blood pressure.',
    'Fruits': 'Daily fruit consumption provides antioxidants that protect cardiovascular health.',
    'Veggies': 'Daily vegetable intake is a significant protective dietary factor against CHD.',
    'HvyAlcoholConsump': 'Heavy alcohol use raises blood pressure and can weaken the heart muscle.',
    'GenHlth': 'Self-reported general health correlates with objective cardiovascular risk markers.',
    'MentHlth': 'Poor mental health is associated with increased inflammatory markers and CHD risk.',
    'PhysHlth': 'Frequent physical health problems may indicate underlying cardiovascular issues.',
    'DiffWalk': 'Mobility difficulties often reflect deconditioning and increased cardiac risk.',
    'Sex': 'Biological sex influences cardiovascular risk through hormonal and physiological differences.',
    'Age': 'Age is a non-modifiable factor that compounds all other cardiovascular risk factors.',
}


def _generate_recommendations(shap_values, form_data):
    recs = []

    risk_features = [s for s in shap_values if s['direction'] == 'risk']

    rec_map = {
        'HighBP': {
            'title': 'Monitor Your Blood Pressure',
            'text': 'Check your blood pressure regularly and discuss readings with your doctor. Controlling hypertension is one of the most effective ways to reduce CHD risk. Aim for below 130/80 mmHg.',
            'icon': 'heart',
        },
        'HighChol': {
            'title': 'Manage Your Cholesterol',
            'text': 'Ask your doctor about a lipid panel test. Reducing saturated fat intake, increasing soluble fibre, and regular exercise can help lower LDL cholesterol naturally.',
            'icon': 'chart',
        },
        'BMI': {
            'title': 'Work Toward a Healthier Weight',
            'text': 'Even a 5-10% reduction in body weight can meaningfully lower cardiovascular risk. Focus on sustainable dietary changes and regular movement rather than rapid weight loss.',
            'icon': 'scale',
        },
        'Smoker': {
            'title': 'Quit Smoking',
            'text': 'Smoking cessation is the single most impactful lifestyle change for heart health. Your cardiovascular risk begins to drop within weeks of quitting. Ask your GP about cessation support.',
            'icon': 'no-smoking',
        },
        'PhysActivity': {
            'title': 'Increase Physical Activity',
            'text': 'Aim for at least 150 minutes of moderate aerobic activity per week. Start with 10-minute walks and gradually increase. Regular exercise strengthens the heart and lowers blood pressure.',
            'icon': 'activity',
        },
        'Diabetes': {
            'title': 'Manage Your Blood Sugar',
            'text': 'Work with your healthcare provider to keep blood sugar levels well-controlled. A combination of diet, exercise, and medication adherence can significantly reduce cardiovascular complications.',
            'icon': 'shield',
        },
        'HvyAlcoholConsump': {
            'title': 'Reduce Alcohol Intake',
            'text': 'Limit alcohol to moderate levels — no more than 1 drink per day for women, 2 for men. Excessive drinking raises blood pressure and can weaken the heart muscle over time.',
            'icon': 'alert',
        },
        'GenHlth': {
            'title': 'Improve Your Overall Health',
            'text': 'Schedule regular check-ups with your doctor. Small improvements in diet, sleep, and stress management compound over time to significantly improve cardiovascular health.',
            'icon': 'health',
        },
        'MentHlth': {
            'title': 'Support Your Mental Health',
            'text': 'Chronic stress and poor mental health increase inflammation and CHD risk. Consider mindfulness, therapy, or counselling. Even moderate stress reduction has measurable cardiovascular benefits.',
            'icon': 'mind',
        },
        'PhysHlth': {
            'title': 'Address Physical Health Concerns',
            'text': 'Frequent days of poor physical health may indicate underlying conditions. Discuss persistent symptoms with your doctor to identify and treat any contributing factors.',
            'icon': 'health',
        },
        'Fruits': {
            'title': 'Add More Fruit to Your Diet',
            'text': 'Aim for at least 2 servings of fruit daily. Berries, citrus, and apples are particularly beneficial for heart health due to their antioxidant and fibre content.',
            'icon': 'fruit',
        },
        'Veggies': {
            'title': 'Eat More Vegetables',
            'text': 'Aim for at least 3 servings of vegetables daily. Leafy greens, cruciferous vegetables, and legumes provide nutrients that support cardiovascular health.',
            'icon': 'veggie',
        },
        'DiffWalk': {
            'title': 'Improve Your Mobility',
            'text': 'Difficulty walking may limit your ability to exercise. Discuss with your doctor about physical therapy or adapted exercise programs that can safely improve your cardiovascular fitness.',
            'icon': 'walk',
        },
    }

    for sf in risk_features:
        feat = sf['feature']
        if feat in rec_map:
            rec = rec_map[feat].copy()
            rec['clinical_note'] = CLINICAL_NOTES.get(feat, '')
            rec['shap_value'] = sf['shap_value']
            recs.append(rec)

    if not form_data.get('physAct') and 'PhysActivity' not in [r.get('feature') for r in recs]:
        rec = rec_map['PhysActivity'].copy()
        rec['clinical_note'] = CLINICAL_NOTES.get('PhysActivity', '')
        rec['shap_value'] = 0
        recs.append(rec)

    return recs[:5]


def get_metrics():
    _load()
    return _cache['metrics']
