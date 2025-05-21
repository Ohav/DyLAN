import sys
sys.path.insert(0, '/home/morg/students/ohavbarbi/rogue-agents-tmp/')
import glob, tqdm, json, pickle
import os
import pandas as pd
import numpy as np
import plotly.express as px
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from monitors.metric_utils import *
from monitors import train_classifier
# from intervention.intervention_models import PolynomialModel, FullyConnected
from interventions.intervention_models import PolynomialModel
from variants.asymmetric.intel_agent import IntelAgent
from variants.symmetric.symm_agent import SymmAgent


def train_all_columns(name, games, output_path, validation_ratio=3, verbose=True):
    all_games = list(games.exp_name.unique())
    np.random.seed(777)
    np.random.shuffle(all_games)
    if validation_ratio > 0:
        valid_len = (len(all_games) // 10) * validation_ratio
        validation_games = games[games.exp_name.apply(lambda x: x in all_games[:valid_len])]
        all_valid, _ = train_classifier.load_training_data(validation_games, METRIC_COLUMNS)
    else:
        valid_len = 0
        all_valid = None
    train_games = games[games.exp_name.apply(lambda x: x in all_games[valid_len:])]


    all_train, norm_params = train_classifier.load_training_data(train_games, METRIC_COLUMNS)
    flat_train = train_classifier.flatten_data(all_train)

    col_names = np.array(['None', 'ent', 'var', 'kur'])
    os.makedirs(output_path + name, exist_ok=True)
    
    trained_models = dict()
    vals = []
    for i in range(1, len(METRIC_COLUMNS) + 1):
        for cols in itertools.combinations(range(1, len(METRIC_COLUMNS) + 1), i):
            cols = list(cols)
            feature_indices = [0] + cols
            for degree in [1, 2, 3, 4, 5]:
                polynomial_model = PolynomialModel(feature_indices=feature_indices, normalization_params=norm_params, degree=degree)
                # polynomial_model = FullyConnected(feature_indices=feature_indices, normalization_params=norm_params, degree=degree)
                
                # Features are selected on init, so the model gets all_train
                polynomial_model.fit(flat_train)
                model_name = f'{name}/poly_' + '_'.join(col_names[cols]) + f'_deg_{degree}'
                with open(f'{output_path}{model_name}.pkl', 'wb') as f:
                    pickle.dump(polynomial_model, f)

                if all_valid is not None:
                    performance = train_classifier.test_model_theory(polynomial_model, all_valid)
                    performance['name'] = model_name
                else:
                    performance = pd.DataFrame.from_dict([{'name': model_name, 'intervention_gain':0, 'th':0}])
                vals.append(performance)
        
    random_model = lambda x: np.random.random(size=len(x))
    model_name = f'{name}/random_model'
    if all_valid is not None:
        performance = train_classifier.test_model_theory(random_model, all_valid)
        performance['name'] = model_name
    else:
        performance = pd.DataFrame.from_dict([{'name': model_name, 'intervention_gain':0, 'th':0}])
    vals.append(performance)
    
    
    vals = pd.concat(vals)
    hyper_parameter_search_lines = []
    
    if all_valid is not None:
        title = name.split('_')[0] + ' ' + name.split('_')[1].replace('acc', 'Accuser').replace('intel', 'Intel') + ' Classifiers'
        fig = px.line(vals, x='precision', y='recall', color='name', title=title) 
        fig.write_image(f'{output_path}{name}/{name}_models.png')
    
        for sub_name in vals['name'].unique():
            cur = vals[vals['name'] == sub_name]
            best = cur[cur.intervention_gain == cur.intervention_gain.max()].iloc[0]
            hyper_parameter_search_lines.append(f"Best for {sub_name}: gain={best.intervention_gain:.3f}, f1={best.f1_score:.3f} (r={best.recall:.3f}, p={best.precision:.3f} th={best.threshold:.2f})")
    
        # real_best = vals[vals.intervention_gain == vals.intervention_gain.max()].iloc[0]
        real_best = vals[vals.intervention_gain == vals.intervention_gain.min()].iloc[0]
        hyper_parameter_search_lines.append(f"Absolute best {name}: {real_best['name']} gain={real_best.intervention_gain:.3f}, f1={real_best.f1_score:.3f} (r={real_best.recall:.3f}, p={real_best.precision:.3f} th={real_best.threshold:.2f})")
        with open(f'{output_path}{name}/res.txt', 'w') as f:
            for l in hyper_parameter_search_lines:
                f.write(l + '\n')
        vals.to_csv(f'{output_path}{name}/model_res.csv')
    if verbose:
        print('\n'.join(hyper_parameter_search_lines))
    return vals

    

    

    