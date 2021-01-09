import pandas as pd
import numpy as np
import gc
import sys
from collections import defaultdict
from tqdm.notebook import tqdm
import joblib
import argparse

import random
import os

from trueskill_features import add_trueskill

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg("--samples", type=int, default=10000000, help="train samples to estimate parameters")
args = parser.parse_args()

train_pickle = '../input/riiid-cross-validation-files/cv3_train.pickle'
valid_pickle = '../input/riiid-cross-validation-files/cv3_valid.pickle'
question_file = '../input/riiid-test-answer-prediction/questions.csv'

# Read data
feld_needed = ['timestamp', 'user_id', 'answered_correctly', 'content_id', 'task_container_id', 'content_type_id', 'prior_question_elapsed_time', 'prior_question_had_explanation']
train = pd.read_pickle(train_pickle)[feld_needed]
valid = pd.read_pickle(valid_pickle)[feld_needed]

train = train.iloc[-args.samples:]

train = train.loc[train.content_type_id == False].reset_index(drop = True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop = True)

 users_rating = defaultdict(trueskill.setup(mu=0.3, sigma=0.164486, beta=0.05, tau=0.00164,
 draw_probability=0).Rating)
 questions_rating = defaultdict(trueskill.setup(mu=0.3, sigma=0.164486, beta=0.05, tau=0.00164,
 draw_probability=0).Rating)

print('User feature trueskill calculation started...')
print('\n')
train = add_trueskill(train, users_rating, questions_rating)
valid = add_trueskill(valid, users_rating, questions_rating)
gc.collect()
print('User feature trueskill calculation completed...')
print('\n')

ur = defaultdict(str)
qr = defaultdict(str)
for u in users_rating.keys():
    ur[int(u)] = str(users_rating[u].mu) + (' ') + str(users_rating[u].sigma)
for q in questions_rating.keys():
    qr[int(q)] = str(questions_rating[q].mu) + (' ') + str(questions_rating[q].sigma)
    
joblib.dump(ur, 'users_rating.pkl')
joblib.dump(qr, 'questions_rating.pkl')
