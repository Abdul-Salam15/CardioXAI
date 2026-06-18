"""
Build a CHD Risk Factor Knowledge Graph using NetworkX and visualize with Pyvis.
Generates a static HTML visualization showing interconnected risk factors.
"""
import os
import json
import networkx as nx
from pyvis.network import Network

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_store')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'core', 'static', 'core')

RISK_FACTORS = {
    'HighBP': {
        'label': 'High Blood Pressure',
        'category': 'cardiovascular',
        'modifiable': True,
        'description': 'Hypertension is the leading modifiable risk factor for CHD.',
    },
    'HighChol': {
        'label': 'High Cholesterol',
        'category': 'cardiovascular',
        'modifiable': True,
        'description': 'Elevated LDL promotes arterial plaque formation.',
    },
    'BMI': {
        'label': 'BMI (Overweight)',
        'category': 'metabolic',
        'modifiable': True,
        'description': 'Higher BMI increases cardiac workload.',
    },
    'Smoker': {
        'label': 'Smoking',
        'category': 'lifestyle',
        'modifiable': True,
        'description': 'Smoking damages blood vessels and accelerates atherosclerosis.',
    },
    'Diabetes': {
        'label': 'Diabetes',
        'category': 'metabolic',
        'modifiable': True,
        'description': 'Doubles the risk of heart disease through vascular damage.',
    },
    'PhysActivity': {
        'label': 'Physical Activity',
        'category': 'lifestyle',
        'modifiable': True,
        'description': 'Regular exercise strengthens the heart (protective).',
    },
    'Stroke': {
        'label': 'History of Stroke',
        'category': 'cardiovascular',
        'modifiable': False,
        'description': 'Indicates existing vascular damage.',
    },
    'Age': {
        'label': 'Age',
        'category': 'demographic',
        'modifiable': False,
        'description': 'Non-modifiable factor that compounds other risks.',
    },
    'Sex': {
        'label': 'Biological Sex',
        'category': 'demographic',
        'modifiable': False,
        'description': 'Influences risk through hormonal and physiological differences.',
    },
    'GenHlth': {
        'label': 'General Health',
        'category': 'health_status',
        'modifiable': True,
        'description': 'Self-reported health correlates with cardiovascular risk.',
    },
    'HvyAlcoholConsump': {
        'label': 'Heavy Alcohol Use',
        'category': 'lifestyle',
        'modifiable': True,
        'description': 'Raises blood pressure and can weaken heart muscle.',
    },
    'DiffWalk': {
        'label': 'Difficulty Walking',
        'category': 'health_status',
        'modifiable': True,
        'description': 'Mobility difficulty reflects deconditioning and cardiac risk.',
    },
    'Fruits': {
        'label': 'Fruit Consumption',
        'category': 'lifestyle',
        'modifiable': True,
        'description': 'Provides antioxidants that protect cardiovascular health.',
    },
    'Veggies': {
        'label': 'Vegetable Consumption',
        'category': 'lifestyle',
        'modifiable': True,
        'description': 'Significant protective dietary factor against CHD.',
    },
    'MentHlth': {
        'label': 'Mental Health',
        'category': 'health_status',
        'modifiable': True,
        'description': 'Poor mental health increases inflammation and CHD risk.',
    },
    'PhysHlth': {
        'label': 'Physical Health',
        'category': 'health_status',
        'modifiable': True,
        'description': 'Frequent health problems may indicate cardiovascular issues.',
    },
}

KNOWN_EDGES = [
    ('HighBP', 'HighChol', 'Both contribute to atherosclerosis', 0.8),
    ('HighBP', 'Stroke', 'Hypertension is the leading cause of stroke', 0.9),
    ('HighBP', 'BMI', 'Obesity raises blood pressure', 0.7),
    ('HighBP', 'DiffWalk', 'Hypertension with mobility limitation compounds risk', 0.5),
    ('HighChol', 'BMI', 'Obesity affects lipid metabolism', 0.7),
    ('HighChol', 'Diabetes', 'Diabetes impairs cholesterol regulation', 0.6),
    ('BMI', 'Diabetes', 'Obesity is a major risk factor for Type 2 diabetes', 0.8),
    ('BMI', 'PhysActivity', 'Exercise helps control weight', 0.7),
    ('BMI', 'DiffWalk', 'Excess weight impairs mobility', 0.6),
    ('Smoker', 'HighBP', 'Smoking raises blood pressure', 0.6),
    ('Smoker', 'Stroke', 'Smoking doubles stroke risk', 0.7),
    ('Smoker', 'HvyAlcoholConsump', 'Substance use often co-occurs', 0.4),
    ('Diabetes', 'Stroke', 'Diabetes increases stroke risk', 0.7),
    ('Diabetes', 'DiffWalk', 'Diabetic neuropathy impairs mobility', 0.5),
    ('PhysActivity', 'BMI', 'Exercise controls weight', 0.7),
    ('PhysActivity', 'MentHlth', 'Exercise improves mental health', 0.5),
    ('PhysActivity', 'GenHlth', 'Active people report better health', 0.6),
    ('Age', 'HighBP', 'Blood pressure rises with age', 0.7),
    ('Age', 'HighChol', 'Cholesterol management changes with age', 0.5),
    ('Age', 'Diabetes', 'Type 2 diabetes risk increases with age', 0.6),
    ('Age', 'DiffWalk', 'Mobility declines with age', 0.7),
    ('Age', 'GenHlth', 'General health tends to decline with age', 0.6),
    ('Fruits', 'Veggies', 'Both are dietary protective factors', 0.8),
    ('Fruits', 'BMI', 'Healthy diet helps control weight', 0.4),
    ('Veggies', 'HighChol', 'Vegetable intake helps lower cholesterol', 0.4),
    ('GenHlth', 'PhysHlth', 'Physical and general health correlate', 0.8),
    ('GenHlth', 'MentHlth', 'Mental and general health correlate', 0.6),
    ('MentHlth', 'PhysHlth', 'Mental and physical health interact', 0.7),
    ('Sex', 'HighBP', 'Hypertension prevalence differs by sex', 0.4),
]

CATEGORY_COLORS = {
    'cardiovascular': '#B71C1C',
    'metabolic': '#E65100',
    'lifestyle': '#2E7D32',
    'demographic': '#6b7280',
    'health_status': '#1A73E8',
}


def build_graph():
    G = nx.Graph()

    for feat, info in RISK_FACTORS.items():
        G.add_node(feat, **info)

    for src, tgt, desc, weight in KNOWN_EDGES:
        if src in RISK_FACTORS and tgt in RISK_FACTORS:
            G.add_edge(src, tgt, description=desc, weight=weight)

    return G


def generate_pyvis_html(G, output_path=None):
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, 'knowledge_graph.html')

    net = Network(
        height='600px', width='100%',
        bgcolor='#f8fafc', font_color='#1f2937',
        directed=False, notebook=False,
    )

    net.set_options(json.dumps({
        "nodes": {
            "font": {"size": 13, "face": "Public Sans, system-ui, sans-serif", "strokeWidth": 3, "strokeColor": "#ffffff"},
            "borderWidth": 2,
            "borderWidthSelected": 3,
            "shadow": {"enabled": True, "size": 8, "color": "rgba(0,0,0,0.1)"},
        },
        "edges": {
            "color": {"color": "#d1d5db", "highlight": "#1A73E8"},
            "smooth": {"type": "continuous"},
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -160,
                "centralGravity": 0.008,
                "springLength": 280,
                "springConstant": 0.04,
                "damping": 0.6,
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 200},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
        },
    }))

    degree_cent = nx.degree_centrality(G)

    for node_id, data in G.nodes(data=True):
        color = CATEGORY_COLORS.get(data.get('category', ''), '#6b7280')
        size = 18 + degree_cent.get(node_id, 0) * 35
        shape = 'dot' if data.get('modifiable', True) else 'diamond'
        title = f"<b>{data['label']}</b><br>{data['description']}<br><i>{'Modifiable' if data.get('modifiable') else 'Non-modifiable'}</i>"
        net.add_node(
            node_id,
            label=data['label'],
            title=title,
            color=color,
            size=size,
            shape=shape,
        )

    min_edge_weight = 0.6
    for src, tgt, data in G.edges(data=True):
        w = data.get('weight', 0.5)
        if w < min_edge_weight:
            continue
        net.add_edge(
            src, tgt,
            title=data.get('description', ''),
            width=w * 2.5,
            value=w,
        )

    net.save_graph(output_path)
    print(f"Knowledge graph saved to {output_path}")
    return output_path


def get_graph_data(shap_values=None):
    G = build_graph()

    if shap_values:
        shap_map = {s['feature']: s for s in shap_values}
        for node_id in G.nodes():
            if node_id in shap_map:
                G.nodes[node_id]['shap_value'] = shap_map[node_id]['shap_value']
                G.nodes[node_id]['shap_direction'] = shap_map[node_id]['direction']

    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            'id': node_id,
            'label': data['label'],
            'category': data.get('category', ''),
            'modifiable': data.get('modifiable', True),
            'description': data.get('description', ''),
            'color': CATEGORY_COLORS.get(data.get('category', ''), '#6b7280'),
            'shap_value': data.get('shap_value'),
            'shap_direction': data.get('shap_direction'),
        })

    edges = []
    for src, tgt, data in G.edges(data=True):
        edges.append({
            'source': src,
            'target': tgt,
            'description': data.get('description', ''),
            'weight': data.get('weight', 0.5),
        })

    return {'nodes': nodes, 'edges': edges}


if __name__ == '__main__':
    G = build_graph()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    generate_pyvis_html(G)
