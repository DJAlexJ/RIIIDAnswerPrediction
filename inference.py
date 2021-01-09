import pandas as pd
import numpy as np
import gc
import sys
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm.notebook import tqdm
import lightgbm as lgb
import joblib
import pickle
import argparse
import json

import random
import os

import trueskill
from trueskill import Rating, quality_1vs1, rate_1vs1

from elo_features import update_parameters, mish, sigmoid
from loop_features import update_features
from trueskill_features import update_trueskill

TARGET = 'answered_correctly'

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg("--validation", type=int, default=False, help="Whether to use own test_df for validation")
args = parser.parse_args()

valid_flag = args.validation

#Loading dicts with features:
item_params = pd.read_pickle('../input/data-for-riiid/item_params_df.pkl')
bundle_params = pd.read_pickle('../input/data-for-riiid/bundle_params_df.pkl')
container_params = pd.read_pickle('../input/data-for-riiid/container_params_df.pkl')
student_params = pd.read_pickle('../input/data-for-riiid/student_params_df.pkl')
student_params_b = pd.read_pickle('../input/data-for-riiid/student_params_b_df.pkl')

answered_correctly_u_count = joblib.load('../input/data-for-riiid/answered_correctly_u_count.pkl')
answered_correctly_u_sum = joblib.load('../input/data-for-riiid/answered_correctly_u_sum.pkl')
elapsed_time_u_sum = joblib.load('../input/data-for-riiid/elapsed_time_u_sum.pkl')
elapsed_time_u_sum_correct = joblib.load('../input/data-for-riiid/elapsed_time_u_sum_correct.pkl')
explanation_u_sum = joblib.load('../input/data-for-riiid/explanation_u_sum.pkl')
timestamp_u = joblib.load('../input/data-for-riiid/timestamp_u.pkl')
timestamp_u_sum = joblib.load('../input/data-for-riiid/timestamp_u_sum.pkl')
timestamp_u_count = joblib.load('../input/data-for-riiid/timestamp_u_count.pkl')
timestamp_u_incorrect = joblib.load('../input/data-for-riiid/timestamp_u_incorrect.pkl')
timestamp_u_correct = joblib.load('../input/data-for-riiid/timestamp_u_correct.pkl')
    
answered_correctly_q_count = joblib.load('../input/data-for-riiid/answered_correctly_q_count.pkl')
answered_correctly_q_sum = joblib.load('../input/data-for-riiid/answered_correctly_q_sum.pkl')
elapsed_time_q_sum = joblib.load('../input/data-for-riiid/elapsed_time_q_sum.pkl')
explanation_q_sum = joblib.load('../input/data-for-riiid/explanation_q_sum.pkl')
#-----------------------------------------------------------------------------------

model = joblib.load('../input/data-for-riiid/model.pkl')
    
with open("../input/data-for-riiid/answered_correctly_uq.pkl", "r") as read_file:
    answered_correctly_uq = json.load(read_file)
answered_correctly_uq = defaultdict(lambda: defaultdict(int), answered_correctly_uq)

ur = joblib.load('../input/data-for-riiid/users_rating.pkl')
qr = joblib.load('../input/data-for-riiid/questions_rating.pkl')

users_rating = defaultdict(trueskill.setup(mu=0.3, sigma=0.164486, beta=0.05, tau=0.00164,
draw_probability=0).Rating)
questions_rating = defaultdict(trueskill.setup(mu=0.3, sigma=0.164486, beta=0.05, tau=0.00164,
draw_probability=0).Rating)

for u in ur.keys():
    users_rating[u] = Rating(mu=float(ur[u].split(' ')[0]), sigma=float(ur[u].split(' ')[1]))
for q in qr.keys():
    questions_rating[q] = Rating(mu=float(qr[q].split(' ')[0]), sigma=float(qr[q].split(' ')[1]))
    
del ur, qr
gc.collect()


class Iter_Valid(object):
    def __init__(self, df, max_user=1000):
        df = df.reset_index(drop=True)
        self.df = df
        self.user_answer = df['user_answer'].astype(str).values
        self.answered_correctly = df['answered_correctly'].astype(str).values
        df['prior_group_responses'] = "[]"
        df['prior_group_answers_correct'] = "[]"
        self.sample_df = df[df['content_type_id'] == 0][['row_id']]
        self.sample_df['answered_correctly'] = 0
        self.len = len(df)
        self.user_id = df.user_id.values
        self.task_container_id = df.task_container_id.values
        self.content_type_id = df.content_type_id.values
        self.max_user = max_user
        self.current = 0
        self.pre_user_answer_list = []
        self.pre_answered_correctly_list = []

    def __iter__(self):
        return self
    
    def fix_df(self, user_answer_list, answered_correctly_list, pre_start):
        df= self.df[pre_start:self.current].copy()
        sample_df = self.sample_df[pre_start:self.current].copy()
        df.loc[pre_start,'prior_group_responses'] = '[' + ",".join(self.pre_user_answer_list) + ']'
        df.loc[pre_start,'prior_group_answers_correct'] = '[' + ",".join(self.pre_answered_correctly_list) + ']'
        self.pre_user_answer_list = user_answer_list
        self.pre_answered_correctly_list = answered_correctly_list
        return df, sample_df

    def __next__(self):
        added_user = set()
        pre_start = self.current
        pre_added_user = -1
        pre_task_container_id = -1
        pre_content_type_id = -1
        user_answer_list = []
        answered_correctly_list = []
        while self.current < self.len:
            crr_user_id = self.user_id[self.current]
            crr_task_container_id = self.task_container_id[self.current]
            crr_content_type_id = self.content_type_id[self.current]
            if crr_user_id in added_user and (crr_user_id != pre_added_user or (crr_task_container_id != pre_task_container_id and crr_content_type_id == 0 and pre_content_type_id == 0)):
                 # known user(not prev user or (differnt task container and both question))
                return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            if len(added_user) == self.max_user:
                if  crr_user_id == pre_added_user and (crr_task_container_id == pre_task_container_id or crr_content_type_id == 1):
                    user_answer_list.append(self.user_answer[self.current])
                    answered_correctly_list.append(self.answered_correctly[self.current])
                    self.current += 1
                    continue
                else:
                    return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            added_user.add(crr_user_id)
            pre_added_user = crr_user_id
            pre_task_container_id = crr_task_container_id
            pre_content_type_id = crr_content_type_id
            user_answer_list.append(self.user_answer[self.current])
            answered_correctly_list.append(self.answered_correctly[self.current])
            self.current += 1
        if pre_start < self.current:
            return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
        else:
            raise StopIteration()
            
if valid_flag:
    target_df = pd.read_pickle(valid_pickle)
    iter_test = Iter_Valid(target_df,max_user=1000)
    predicted = []
    def set_predict(df):
        predicted.append(df)
else:
    import riiideducation
    env = riiideducation.make_env()
    iter_test = env.iter_test()
    set_predict = env.predict
     
     
item_params = item_params.set_index('content_id').transpose().to_dict()
bundle_params = bundle_params.set_index('part_bundle_id').transpose().to_dict()
container_params = container_params.set_index('task_container_id').transpose().to_dict()
student_params = student_params.set_index('user_id').transpose().to_dict()
student_params_b = student_params_b.set_index('user_id').transpose().to_dict()

previous_test_df = None
for (test_df, sample_prediction_df) in iter_test:
    if previous_test_df is not None:
        previous_test_df[TARGET] = eval(test_df["prior_group_answers_correct"].iloc[0])
        previous_test_df['prior_question_had_explanation'] = previous_test_df.prior_question_had_explanation.fillna(False).astype('int8')
        previous_test_df = pd.concat([previous_test_df.reset_index(drop=True), questions_df[['question_id', 'part_bundle_id', 'bundle_id', 'bundle_correctness', 'content_correctness_std']].reindex(previous_test_df['content_id'].values).reset_index(drop=True)], axis=1)
        update_features(previous_test_df, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, elapsed_time_u_sum_correct, explanation_u_sum, timestamp_u, timestamp_u_sum, timestamp_u_count, timestamp_u_incorrect, timestamp_u_correct, answered_correctly_q_count, answered_correctly_q_sum, elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq)
        update_trueskill(previous_test_df, users_rating, questions_rating)
        
        previous_test_df = previous_test_df.rename(columns={'user_id': 'student_id'})
        previous_test_df['left_asymptote'] = 1/4
        
        student_params, student_params_b, item_params, bundle_params, container_params = update_parameters(previous_test_df, student_params, student_params_b, item_params, bundle_params, container_params)
    
    previous_test_df = test_df.copy()
    test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop = True)
    test_df[TARGET] = 0
    test_df = add_features(test_df, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, elapsed_time_u_sum_correct, explanation_u_sum, timestamp_u, timestamp_u_sum, timestamp_u_count, timestamp_u_incorrect, timestamp_u_correct, answered_correctly_q_count, answered_correctly_q_sum, elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq, update = False)
    test_df = add_trueskill(test_df, users_rating, questions_rating, update = False)
    test_df = pd.concat([test_df, questions_df[['question_id', 'part_bundle_id', 'bundle_id', 'bundle_correctness', 'content_correctness_std']].reindex(test_df['content_id'].values).reset_index(drop=True)], axis=1)
    test_df = pd.concat([test_df, content_explation_agg.reindex(test_df['content_id'].values).reset_index(drop=True)[['content_explation_false_mean', 'content_explation_true_mean']]], axis=1)
    test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
    test_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
    test_df['hmean_user_content'] = 2 * (test_df['answered_correctly_u_avg'] * test_df['answered_correctly_q_avg']) / (test_df['answered_correctly_u_avg'] + test_df['answered_correctly_q_avg'])
    test_df['hmean_correct_incorrect_tms'] = 2 * (test_df['timestamp_u_recency_1'] * test_df['timestamp_u_incorrect_recency']) / (test_df['timestamp_u_recency_1'] + test_df['timestamp_u_incorrect_recency'])
    test_df['correct_incorrect'] = 2 * (test_df['timestamp_u_correct_recency'] * test_df['timestamp_u_incorrect_recency']) / (test_df['timestamp_u_correct_recency'] + test_df['timestamp_u_incorrect_recency'])
    test_df['engagement'] = (test_df['timestamp_u_recency_1'] + test_df['timestamp_u_recency_2'] + test_df['timestamp_u_recency_3'] + test_df['timestamp_u_recency_4']) / test_df['timelag_mean']
    
    for num, row in enumerate(tqdm(test_df[['content_id', 'user_id', 'part_bundle_id', 'task_container_id']].values)):
        if row[0] in item_params.keys():
            test_df.loc[num, 'beta'] = item_params[row[0]]['beta']
            test_df.loc[num, 'item_nb_answers'] = item_params[row[0]]['item_nb_answers']
        else:
            test_df.loc[num, 'beta'] = 0
            test_df.loc[num, 'item_nb_answers'] = 0

        if row[1] in student_params.keys():
            test_df.loc[num, 'theta'] = student_params[row[1]]['theta']
            test_df.loc[num, 'theta_b'] = student_params_b[row[1]]['theta_b']
            test_df.loc[num, 'student_nb_answers'] = student_params[row[1]]['student_nb_answers']
        else:
            test_df.loc[num, 'theta'] = 0
            test_df.loc[num, 'theta_b'] = 0
            test_df.loc[num, 'student_nb_answers'] = 0
        
        if row[2] in bundle_params.keys():
            test_df.loc[num, 'epsilon'] = bundle_params[row[2]]['epsilon']
        else:
            test_df.loc[num, 'epsilon'] = 0
            
        if row[3] in container_params.keys():
            test_df.loc[num, 'gamma'] = container_params[row[3]]['gamma']
            test_df.loc[num, 'container_nb_answers'] = container_params[row[3]]['container_nb_answers']
        else:
            test_df.loc[num, 'gamma'] = 0
            test_df.loc[num, 'container_nb_answers'] = 0
    
    test_df['prob_of_good_answer1'] = 1/4 + 3/4 * sigmoid(test_df.theta - test_df.beta)
    test_df['prob_of_good_answer2'] = 1/4 + 3/4 * mish(test_df.theta_b - test_df.beta)
    test_df['prob_of_good_answer3'] = 1/4 + 3/4 * sigmoid(test_df.epsilon)
    test_df['prob_of_good_answer4'] = 1/4 + 3/4 * sigmoid(2*(test_df.theta * test_df.beta)/(test_df.theta + test_df.beta))
    test_df[TARGET] =  model.predict(test_df[FEATURES])
    set_predict(test_df[['row_id', TARGET]])
        
print('Job Done')
