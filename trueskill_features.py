import numpy as np
import pandas as pd
import trueskill
from trueskill import Rating, quality_1vs1, rate_1vs1
from tqdm import tqdm
import math

def win_probability(team1, team2):
    delta_mu = team1.mu - team2.mu
    sum_sigma = sum([team1.sigma ** 2, team2.sigma ** 2])
    size = 2
    denom = math.sqrt(size * (0.05 * 0.05) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)

def add_trueskill(df, users_rating, questions_rating, update = True):
    # -----------------------------------------------------------------------
    users_mean = np.zeros(len(df), dtype = np.float32)
    users_var = np.zeros(len(df), dtype = np.float32)
    questions_mean = np.zeros(len(df), dtype = np.float32)
    questions_var = np.zeros(len(df), dtype = np.float32)
    win_prob = np.zeros(len(df), dtype = np.float32)
    quality = np.zeros(len(df), dtype = np.float32)
    
    for num, row in enumerate(tqdm(df[['user_id', 'content_id', 'answered_correctly']].values)):
        
        users_mean[num] = users_rating[row[0]].mu
        users_var[num] = users_rating[row[0]].sigma
            
        questions_mean[num] = questions_rating[row[1]].mu
        questions_var[num] = questions_rating[row[1]].sigma
            
        win_prob[num] = win_probability(users_rating[row[0]], questions_rating[row[1]])
        quality[num] = quality_1vs1(users_rating[row[0]], questions_rating[row[1]])
        
        if update is True:
            
            old_user_rating = users_rating[row[0]]
            old_question_rating = questions_rating[row[1]]
            if row[2] == 1:
                new_user_rating, new_question_rating = rate_1vs1(old_user_rating, old_question_rating)
            if row[2] == 0:
                new_question_rating, new_user_rating = rate_1vs1(old_question_rating, old_user_rating)
                
            users_rating[row[0]] = new_user_rating
            questions_rating[row[1]] = new_question_rating
               
             
            
    user_df = pd.DataFrame({'u_mu': users_mean, 'u_sigma': users_var, 'q_mu': questions_mean,
                            'q_sigma': questions_var, 'win_prob': win_prob, 'quality': quality
                           })
    
    df = pd.concat([df, user_df], axis = 1)
    return df

        
def update_trueskill(df, users_rating, questions_rating):
    for row in df[['user_id', 'content_id', 'answered_correctly', 'content_type_id']].values:
        if row[3] == 0:
            old_user_rating = users_rating[row[0]]
            old_question_rating = questions_rating[row[1]]
            if row[2] == 1:
                new_user_rating, new_question_rating = rate_1vs1(old_user_rating, old_question_rating)
            if row[2] == 0:
                new_question_rating, new_user_rating = rate_1vs1(old_question_rating, old_user_rating)
                
            users_rating[row[0]] = new_user_rating
            questions_rating[row[1]] = new_question_rating
            
