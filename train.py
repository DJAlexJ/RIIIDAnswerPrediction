import pandas as pd
import numpy as np
import gc
import sys
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm.notebook import tqdm
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import json

import random
import os

import datatable as dt
import trueskill
from trueskill import Rating, quality_1vs1, rate_1vs1
import math

from preprocess import preprocess

SEED = 123

# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
seed_everything(SEED)

TARGET = 'answered_correctly'

train_pickle = '../input/riiid-cross-validation-files/cv3_train.pickle'
valid_pickle = '../input/riiid-cross-validation-files/cv3_valid.pickle'
question_file = '../input/riiid-test-answer-prediction/questions.csv'
questions_df = pd.read_csv(question_file)

# Read data
feld_needed = ['timestamp', 'user_id', 'answered_correctly', 'content_id', 'task_container_id', 'content_type_id', 'prior_question_elapsed_time', 'prior_question_had_explanation']
train = pd.read_pickle(train_pickle)[feld_needed]
valid = pd.read_pickle(valid_pickle)[feld_needed]

_ = preprocess(train, valid, save_dicts=True)

FEATURES = ['prior_question_elapsed_time', 'prior_question_had_explanation', 'part_bundle_id', 'answered_correctly_u_avg', 'answered_correctly_u_std', 'elapsed_time_u_avg', 'elapsed_time_u_avg_correct', 'explanation_u_avg', 'answered_correctly_q_avg', 'elapsed_time_q_avg', 'explanation_q_avg', 'timestamp_u_recency_1', 'timestamp_u_recency_2', 'timestamp_u_recency_3', 'timestamp_u_recency_4', 'timestamp_u_incorrect_recency', 'hmean_user_content', 'content_id', 'task_container_id', 'engagement', 'content_correctness_std', 'bundle_correctness', 'content_explation_false_mean','timestamp_u_correct_recency', 'content_explation_true_mean', 'timelag_mean', 'timelag_std', 'hmean_correct_incorrect_tms', 'theta', 'theta_b', 'beta', 'gamma', 'epsilon', 'prob_of_good_answer1', 'prob_of_good_answer2', 'correct_incorrect', 'prob_of_good_answer3', 'prob_of_good_answer4', 'student_nb_answers', 'item_nb_answers', 'container_nb_answers', 'u_mu', 'u_sigma', 'q_mu', 'q_sigma', 'win_prob', 'quality', 'answered_correctly_uq_count'
           ]

params = [
          {'objective': 'binary',
          'seed': SEED,
          'metric': 'auc',
          'feature_fraction': 0.7,
          'num_leaves': 300,
          'max_depth': 13,
          'bagging_freq': 6,
          'bagging_fraction': 0.8,
          'learning_rate': 0.1,
          'lambda_l1': 0.02,
          'lambda_l2': 0.4,
          'min_child_samples': 33,
          'min_child_weight': 16
          }
]

drop_cols = list(set(train.columns) - set(FEATURES))

print(f'Traning with {train.shape[0]} rows and {len(FEATURES)} features')

y_train = train[TARGET]
y_val = valid[TARGET]
# Drop unnecessary columns
train.drop(drop_cols, axis = 1, inplace = True)
valid.drop(drop_cols, axis = 1, inplace = True)
gc.collect()

lgb_train = lgb.Dataset(train[FEATURES], y_train)
lgb_valid = lgb.Dataset(valid[FEATURES], y_val)
    
del train, y_train
gc.collect()

model = lgb.train(
    params = params[0],
    train_set = lgb_train,
    num_boost_round = 1000,
    valid_sets = [lgb_train, lgb_valid],
    early_stopping_rounds = 25,
    verbose_eval = 50
)


print(f'Roc Auc score for the validation data of the model is:', roc_auc_score(y_val, model.predict(valid[FEATURES])))

feature_importance = model.feature_importance()
feature_importance = pd.DataFrame({'Features': FEATURES, 'Importance': feature_importance}).sort_values('Importance', ascending = False)

fig = plt.figure(figsize = (10, 10))
fig.suptitle('Feature Importance', fontsize = 20)
plt.tick_params(axis = 'x', labelsize = 12)
plt.tick_params(axis = 'y', labelsize = 12)
plt.xlabel('Importance', fontsize = 15)
plt.ylabel('Features', fontsize = 15)
sns.barplot(x = feature_importance['Importance'], y = feature_importance['Features'], orient = 'h')
    plt.show()

#Saving model
joblib.dump(model, 'model.pkl')
