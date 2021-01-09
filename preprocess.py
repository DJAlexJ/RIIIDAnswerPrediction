import pandas as pd
import numpy as np
import gc
import sys
from collections import defaultdict
from tqdm.notebook import tqdm
import joblib
import pickle
import json
import random
import os
import trueskill
from trueskill import Rating, quality_1vs1, rate_1vs1
import math
from loop_features import add_features
from elo_features import update_parameters, mish, sigmoid

def preprocess(train, valid, questions_df, save_dicts=False, n_samples=10000000):

    # Reading saved dicts and initializing new ones
    item_params = pd.read_pickle('../input/student-item-params/item_params_df.pkl')
    bundle_params = pd.read_pickle('../input/student-item-params/bundle_params_df.pkl')
    container_params = pd.read_pickle('../input/student-item-params/container_params_df.pkl')
    student_params = pd.read_pickle('../input/student-item-params/student_params_df.pkl')
    student_params_b = pd.read_pickle('../input/student-item-params/student_params_b_df.pkl')
    
     # Client dictionaries
    answered_correctly_u_count = defaultdict(int)
    answered_correctly_u_sum = defaultdict(int)
    elapsed_time_u_sum = defaultdict(int)
    elapsed_time_u_sum_correct = defaultdict(float)
    explanation_u_sum = defaultdict(int)
    timestamp_u = defaultdict(list)
    timestamp_u_sum = defaultdict(int)
    timestamp_u_count = defaultdict(int)
    timestamp_u_incorrect = defaultdict(list)
    timestamp_u_correct = defaultdict(list)
        
    # Question dictionaries
    answered_correctly_q_count = defaultdict(int)
    answered_correctly_q_sum = defaultdict(int)
    elapsed_time_q_sum = defaultdict(int)
    explanation_q_sum = defaultdict(int)
        
    # Client Question dictionary
    answered_correctly_uq = defaultdict(lambda: defaultdict(int))

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
    #----------------------------------------------------------------
    
    content_explation_agg=train[["content_id","prior_question_had_explanation",TARGET]].groupby(["content_id","prior_question_had_explanation"])[TARGET].agg(['mean'])
    content_explation_agg=content_explation_agg.unstack()
    content_explation_agg=content_explation_agg.reset_index()
    content_explation_agg.columns = ['content_id', 'content_explation_false_mean', 'content_explation_true_mean']
    content_explation_agg.content_id=content_explation_agg.content_id.astype('int16')
    content_explation_agg.content_explation_false_mean=content_explation_agg.content_explation_false_mean.astype('float16')
    content_explation_agg.content_explation_true_mean=content_explation_agg.content_explation_true_mean.astype('float16')

    prior_question_elapsed_time_mean = train['prior_question_elapsed_time'].dropna().mean()

    content_agg = train.groupby('content_id')[TARGET].agg(['sum','count','var'])
    content_agg=content_agg.astype('float32')

    train = train.iloc[-n_samples:]

    train = train.loc[train.content_type_id == False].reset_index(drop = True)
    valid = valid.loc[valid.content_type_id == False].reset_index(drop = True)

    train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
    valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')

    train = pd.merge(train, content_explation_agg, on='content_id', how='left')
    valid = pd.merge(valid, content_explation_agg, on='content_id', how='left')
    
    print('User feature calculation started...')
    print('\n')
    train = add_features(train, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, elapsed_time_u_sum_correct, explanation_u_sum, timestamp_u, timestamp_u_sum, timestamp_u_count, timestamp_u_incorrect, timestamp_u_correct, answered_correctly_q_count, answered_correctly_q_sum, elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq)
    valid = add_features(valid, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, elapsed_time_u_sum_correct, explanation_u_sum, timestamp_u, timestamp_u_sum, timestamp_u_count, timestamp_u_incorrect, timestamp_u_correct, answered_correctly_q_count, answered_correctly_q_sum, elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq)
    gc.collect()
    print('User feature calculation completed...')
    print('\n')

    train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
    valid['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)

    train['hmean_user_content'] = 2 * (train['answered_correctly_u_avg'] * train['answered_correctly_q_avg']) / (train['answered_correctly_u_avg'] + train['answered_correctly_q_avg'])
    valid['hmean_user_content'] = 2 * (valid['answered_correctly_u_avg'] * valid['answered_correctly_q_avg']) / (valid['answered_correctly_u_avg'] + valid['answered_correctly_q_avg'])

    train['hmean_correct_incorrect_tms'] = 2 * (train['timestamp_u_recency_1'] * train['timestamp_u_incorrect_recency']) / (train['timestamp_u_recency_1'] + train['timestamp_u_incorrect_recency'])
    valid['hmean_correct_incorrect_tms'] = 2 * (valid['timestamp_u_recency_1'] * valid[r'timestamp_u_incorrect_recency']) / (valid['timestamp_u_recency_1'] + valid['timestamp_u_incorrect_recency'])

    train['correct_incorrect'] = 2 * (train['timestamp_u_correct_recency'] * train['timestamp_u_incorrect_recency']) / (train['timestamp_u_correct_recency'] + train['timestamp_u_incorrect_recency'])
    valid['correct_incorrect'] = 2 * (valid['timestamp_u_correct_recency'] * valid['timestamp_u_incorrect_recency']) / (valid['timestamp_u_correct_recency'] + valid['timestamp_u_incorrect_recency'])

    train['engagement'] = (train['timestamp_u_recency_1'] + train['timestamp_u_recency_2'] + train['timestamp_u_recency_3'] + train['timestamp_u_recency_4']) / train['timelag_mean']
    valid['engagement'] = (valid['timestamp_u_recency_1'] + valid['timestamp_u_recency_2'] + valid['timestamp_u_recency_3'] + valid['timestamp_u_recency_4']) / valid['timelag_mean']

    questions_df['part'] = questions_df['part'].astype(np.int32)
    questions_df['bundle_id'] = questions_df['bundle_id'].astype(np.int32)
    questions_df['part_bundle_id']=questions_df['part']*100000+questions_df['bundle_id']
    questions_df.part_bundle_id=questions_df.part_bundle_id.astype('int32')
    questions_df['content_correctness_std'] = questions_df['question_id'].map(content_agg['var'])
    questions_df.content_correctness_std=questions_df.content_correctness_std.astype('float16')
    questions_df['content_correctness'] = questions_df['question_id'].map(content_agg['sum'] / content_agg['count'])
    questions_df.content_correctness=questions_df.content_correctness.astype('float16')
    bundle_agg = questions_df.groupby('bundle_id')['content_correctness'].agg(['mean'])
    questions_df['bundle_correctness'] = questions_df['bundle_id'].map(bundle_agg['mean'])
    questions_df.bundle_correctness=questions_df.bundle_correctness.astype('float16')

    train = pd.merge(train, questions_df[['question_id', 'part_bundle_id', 'bundle_id', 'bundle_correctness', 'content_correctness_std']], left_on = 'content_id', right_on = 'question_id', how = 'left')
    valid = pd.merge(valid, questions_df[['question_id', 'part_bundle_id', 'bundle_id', 'bundle_correctness', 'content_correctness_std']], left_on = 'content_id', right_on = 'question_id', how = 'left')


     
    train = pd.merge(train, item_params, on='content_id', how='left')
    train = pd.merge(train, bundle_params, on='part_bundle_id', how='left')
    train = pd.merge(train, container_params, on='task_container_id', how='left')
    train = pd.merge(train, student_params, on='user_id', how='left')
    train = pd.merge(train, student_params_b, on='user_id', how='left')
    train['prob_of_good_answer1'] = 1/4 + 3/4 * sigmoid(train.theta - train.beta)
    train['prob_of_good_answer2'] = 1/4 + 3/4 * mish(train.theta_b - train.beta)
    train['prob_of_good_answer3'] = 1/4 + 3/4 * sigmoid(train.epsilon)
    train['prob_of_good_answer4'] = 1/4 + 3/4 * sigmoid(2*(train.theta * train.beta)/(train.theta + train.beta))
    valid = valid.rename(columns={'user_id': 'student_id'})
    valid['left_asymptote'] = 1/4
    item_params = item_params.set_index('content_id').transpose().to_dict()
    bundle_params = bundle_params.set_index('part_bundle_id').transpose().to_dict()
    container_params = container_params.set_index('task_container_id').transpose().to_dict()
    student_params = student_params.set_index('user_id').transpose().to_dict()
    student_params_b = student_params_b.set_index('user_id').transpose().to_dict()

    student_params, student_params_b, item_params, bundle_params, container_params = update_parameters(valid, student_params, student_params_b, item_params, bundle_params, container_params)

    student_params = pd.DataFrame(student_params).transpose().reset_index()
    student_params.columns = ['user_id', 'theta', 'student_nb_answers']
    student_params_b = pd.DataFrame(student_params_b).transpose().reset_index()
    student_params_b.columns = ['user_id', 'theta_b', 'student_nb_answers_b']
    item_params = pd.DataFrame(item_params).transpose().reset_index()
    item_params.columns = ['content_id', 'beta', 'item_nb_answers']
    bundle_params = pd.DataFrame(bundle_params).transpose().reset_index()
    bundle_params.columns = ['part_bundle_id', 'epsilon', 'bundle_nb_answers']
    container_params = pd.DataFrame(container_params).transpose().reset_index()
    container_params.columns = ['task_container_id', 'gamma', 'container_nb_answers']

    valid = valid.rename(columns={'student_id': 'user_id'})
    valid = pd.merge(valid, item_params, on='content_id', how='left')
    valid = pd.merge(valid, bundle_params, on='part_bundle_id', how='left')
    valid = pd.merge(valid, container_params, on='task_container_id', how='left')
    valid = pd.merge(valid, student_params, on='user_id', how='left')
    valid = pd.merge(valid, student_params_b, on='user_id', how='left')
    valid['prob_of_good_answer1'] = 1/4 + 3/4 * sigmoid(valid.theta - valid.beta)
    valid['prob_of_good_answer2'] = 1/4 + 3/4 * mish(valid.theta_b - valid.beta)
    valid['prob_of_good_answer3'] = 1/4 + 3/4 * sigmoid(valid.epsilon)
    valid['prob_of_good_answer4'] = 1/4 + 3/4 * sigmoid(2*(valid.theta * valid.beta)/(valid.theta + valid.beta))
     
    if save_dicts:
         joblib.dump(answered_correctly_u_count, 'answered_correctly_u_count.pkl')
         joblib.dump(answered_correctly_u_sum, 'answered_correctly_u_sum.pkl')
         joblib.dump(elapsed_time_u_sum, 'elapsed_time_u_sum.pkl')
         joblib.dump(elapsed_time_u_sum_correct, 'elapsed_time_u_sum_correct.pkl')
         joblib.dump(explanation_u_sum, 'explanation_u_sum.pkl')
         joblib.dump(timestamp_u, 'timestamp_u.pkl')
         joblib.dump(timestamp_u_sum, 'timestamp_u_sum.pkl')
         joblib.dump(timestamp_u_count, 'timestamp_u_count.pkl')
         joblib.dump(timestamp_u_incorrect, 'timestamp_u_incorrect.pkl')
         joblib.dump(timestamp_u_correct, 'timestamp_u_correct.pkl')
         joblib.dump(answered_correctly_q_count, 'answered_correctly_q_count.pkl')
         joblib.dump(answered_correctly_q_sum, 'answered_correctly_q_sum.pkl')
         joblib.dump(elapsed_time_q_sum, 'elapsed_time_q_sum.pkl')
         joblib.dump(explanation_q_sum, 'explanation_q_sum.pkl')
         json.dump(answered_correctly_uq, open('answered_correctly_uq.pkl', 'w'))

         student_params.to_pickle('student_params_df.pkl')
         student_params_b.to_pickle('student_params_b_df.pkl')
         item_params.to_pickle('item_params_df.pkl')
         bundle_params.to_pickle('bundle_params_df.pkl')
         container_params.to_pickle('container_params_df.pkl')
         
    return {
        'answered_correctly_u_count': answered_correctly_u_count,
        'answered_correctly_u_sum' : answered_correctly_u_sum,
        'elapsed_time_u_sum': elapsed_time_u_sum,
        'elapsed_time_u_sum_correct': elapsed_time_u_sum_correct,
        'explanation_u_sum': explanation_u_sum,
        'timestamp_u': timestamp_u,
        'timestamp_u_sum': timestamp_u_sum,
        'timestamp_u_count': timestamp_u_count,
        'timestamp_u_incorrect': timestamp_u_incorrect,
        'timestamp_u_correct': timestamp_u_correct,
        'answered_correctly_q_count': answered_correctly_q_count,
        'answered_correctly_q_sum': answered_correctly_q_sum,
        'elapsed_time_q_sum': elapsed_time_q_sum,
        'explanation_q_sum': explanation_q_sum,
        'answered_correctly_uq': answered_correctly_uq,
        'users_rating': users_rating,
        'questions_rating': questions_rating,
        'student_params_df': student_params_df,
        'student_params_b_df': student_params_b_df,
        'item_params_df': item_params_df,
        'bundle_params_df': bundle_params_df,
        'container_params_df': container_params_df
}
