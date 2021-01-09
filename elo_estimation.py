import pandas as pd
import numpy as np
import gc
import sys
from collections import defaultdict
from tqdm.notebook import tqdm
import pickle

import random
import os

from elo_features import estimate_parameters

train_pickle = '../input/riiid-cross-validation-files/cv3_train.pickle'
valid_pickle = '../input/riiid-cross-validation-files/cv3_valid.pickle'
question_file = '../input/riiid-test-answer-prediction/questions.csv'

# Read data
feld_needed = ['timestamp', 'user_id', 'answered_correctly', 'content_id', 'task_container_id', 'content_type_id', 'prior_question_elapsed_time', 'prior_question_had_explanation']
train = pd.read_pickle(train_pickle)[feld_needed]
valid = pd.read_pickle(valid_pickle)[feld_needed]

questions_df = pd.read_csv(question_file)
questions_df['part'] = questions_df['part'].astype(np.int32)
questions_df['bundle_id'] = questions_df['bundle_id'].astype(np.int32)
questions_df['part_bundle_id']=questions_df['part']*100000+questions_df['bundle_id']
questions_df.part_bundle_id=questions_df.part_bundle_id.astype('int32')

coefs = train[["content_id", "task_container_id", "user_id", "answered_correctly"]].copy()
coefs.rename(columns={'user_id': 'student_id'}, inplace=True)
coefs = coefs[coefs.answered_correctly != -1]
coefs = pd.merge(coefs, questions_df[['question_id', 'part_bundle_id']], left_on = 'content_id', right_on = 'question_id', how = 'left')
coefs['left_asymptote'] = 1/4

print(f"Dataset of shape {coefs.shape}")
print(f"Columns are {list(coefs.columns)}")

student_parameters, student_parameters_b, item_parameters, bundle_parameters, container_parameters = estimate_parameters(coefs)

student_params = pd.DataFrame(student_parameters).transpose().reset_index()
student_params.columns = ['user_id', 'theta', 'student_nb_answers']
student_params_b = pd.DataFrame(student_parameters_b).transpose().reset_index()
student_params_b.columns = ['user_id', 'theta_b', 'student_nb_answers_b']
item_params = pd.DataFrame(item_parameters).transpose().reset_index()
item_params.columns = ['content_id', 'beta', 'item_nb_answers']
bundle_params = pd.DataFrame(bundle_parameters).transpose().reset_index()
bundle_params.columns = ['part_bundle_id', 'epsilon', 'bundle_nb_answers']
container_params = pd.DataFrame(container_parameters).transpose().reset_index()
container_params.columns = ['task_container_id', 'gamma', 'container_nb_answers']

student_params.to_pickle('student_params_df.pkl')
student_params_b.to_pickle('student_params_b_df.pkl')
item_params.to_pickle('item_params_df.pkl')
bundle_params.to_pickle('bundle_params_df.pkl')
container_params.to_pickle('container_params_df.pkl')
