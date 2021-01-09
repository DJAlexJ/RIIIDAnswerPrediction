import numpy as np
import pandas as pd
from tqdm import tqdm

# Funcion for user stats with loops
def add_features(df, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, elapsed_time_u_sum_correct, explanation_u_sum, timestamp_u, timestamp_u_sum, timestamp_u_count, timestamp_u_incorrect, timestamp_u_correct, answered_correctly_q_count, answered_correctly_q_sum, elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq, update = True):
    # -----------------------------------------------------------------------
    # Client features
    answered_correctly_u_avg = np.zeros(len(df), dtype = np.float32)
    answered_correctly_u_std = np.zeros(len(df), dtype = np.float32)
    elapsed_time_u_avg = np.zeros(len(df), dtype = np.float32)
    elapsed_time_u_avg_correct = np.zeros(len(df), dtype = np.float32)
    explanation_u_avg = np.zeros(len(df), dtype = np.float32)
    timestamp_u_avg = np.zeros(len(df), dtype = np.float32)
    timestamp_u_std = np.zeros(len(df), dtype = np.float32)
    timestamp_u_recency_1 = np.zeros(len(df), dtype = np.float32)
    timestamp_u_recency_2 = np.zeros(len(df), dtype = np.float32)
    timestamp_u_recency_3 = np.zeros(len(df), dtype = np.float32)
    timestamp_u_recency_4 = np.zeros(len(df), dtype = np.float32)
    timestamp_u_incorrect_recency = np.zeros(len(df), dtype = np.float32)
    timestamp_u_correct_recency = np.zeros(len(df), dtype = np.float32)
    # -----------------------------------------------------------------------
    # Question features
    answered_correctly_q_avg = np.zeros(len(df), dtype = np.float32)
    elapsed_time_q_avg = np.zeros(len(df), dtype = np.float32)
    explanation_q_avg = np.zeros(len(df), dtype = np.float32)
    # -----------------------------------------------------------------------
    # User Question
    answered_correctly_uq_count = np.zeros(len(df), dtype = np.int32)
    # -----------------------------------------------------------------------
    
    for num, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp']].values)):
        
        # Client features assignation
        # ------------------------------------------------------------------
        if answered_correctly_u_count[row[0]] != 0:
            answered_correctly_u_avg[num] = answered_correctly_u_sum[row[0]] / answered_correctly_u_count[row[0]]
            elapsed_time_u_avg[num] = elapsed_time_u_sum[row[0]] / answered_correctly_u_count[row[0]]
            explanation_u_avg[num] = explanation_u_sum[row[0]] / answered_correctly_u_count[row[0]]
        else:
            answered_correctly_u_avg[num] = np.nan
            elapsed_time_u_avg[num] = np.nan
            explanation_u_avg[num] = np.nan
        
        if answered_correctly_u_sum[row[0]] != 0:
            elapsed_time_u_avg_correct[num] = elapsed_time_u_sum_correct[row[0]] / answered_correctly_u_sum[row[0]]
        else:
            elapsed_time_u_avg_correct[num] = np.nan
            
        if answered_correctly_u_count[row[0]] >= 2:
            answered_correctly_u_std[num] = np.sqrt((answered_correctly_u_sum[row[0]] - answered_correctly_u_avg[num])**2 / (answered_correctly_u_count[row[0]] - 1))
        else:
            answered_correctly_u_std[num] = np.nan
            
        if timestamp_u_count[row[0]] != 0:
            timestamp_u_avg[num] = timestamp_u_sum[row[0]] / timestamp_u_count[row[0]]
        else:
            timestamp_u_avg[num] = np.nan
            
        if timestamp_u_count[row[0]] >= 2:
            timestamp_u_std[num] = np.sqrt((timestamp_u_sum[row[0]] - timestamp_u_avg[num])**2 / (timestamp_u_count[row[0]] - 1))
        else:
            timestamp_u_std[num] = np.nan
            
        if len(timestamp_u[row[0]]) == 0:
            timestamp_u_recency_1[num] = np.nan
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan
            timestamp_u_recency_4[num] = np.nan
        elif len(timestamp_u[row[0]]) == 1:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan
            timestamp_u_recency_4[num] = np.nan
        elif len(timestamp_u[row[0]]) == 2:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_3[num] = np.nan
            timestamp_u_recency_4[num] = np.nan
        elif len(timestamp_u[row[0]]) == 3:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][2]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_3[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_4[num] = np.nan
        elif len(timestamp_u[row[0]]) == 4:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][3]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][2]
            timestamp_u_recency_3[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_4[num] = row[5] - timestamp_u[row[0]][0]
            
        if len(timestamp_u_incorrect[row[0]]) == 0:
            timestamp_u_incorrect_recency[num] = np.nan
        else:
            timestamp_u_incorrect_recency[num] = row[5] - timestamp_u_incorrect[row[0]][0]
            
        if len(timestamp_u_correct[row[0]]) == 0:
            timestamp_u_correct_recency[num] = np.nan
        else:
            timestamp_u_correct_recency[num] = row[5] - timestamp_u_correct[row[0]][0]
            
            
        if answered_correctly_q_count[row[2]] != 0:
            answered_correctly_q_avg[num] = answered_correctly_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            elapsed_time_q_avg[num] = elapsed_time_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            explanation_q_avg[num] = explanation_q_sum[row[2]] / answered_correctly_q_count[row[2]]
        else:
            answered_correctly_q_avg[num] = np.nan
            elapsed_time_q_avg[num] = np.nan
            explanation_q_avg[num] = np.nan

        answered_correctly_uq_count[num] = answered_correctly_uq[row[0]][row[2]]
        # ------------------------------------------------------------------
        if update is True:
            # Client features updates
            answered_correctly_u_count[row[0]] += 1
            elapsed_time_u_sum[row[0]] += row[3]
            explanation_u_sum[row[0]] += int(row[4])
            if len(timestamp_u[row[0]]) == 4:
                timestamp_u[row[0]].pop(0)
                timestamp_u[row[0]].append(row[5])
            else:
                timestamp_u[row[0]].append(row[5])
                
            if len(timestamp_u[row[0]]) >= 2:
                timestamp_u_sum[row[0]] += timestamp_u[row[0]][-1] - timestamp_u[row[0]][-2]
            timestamp_u_count[row[0]] += 1

            answered_correctly_q_count[row[2]] += 1
            elapsed_time_q_sum[row[2]] += row[3]
            explanation_q_sum[row[2]] += int(row[4])

            answered_correctly_uq[row[0]][row[2]] += 1

            answered_correctly_u_sum[row[0]] += row[1]
            if row[1] == 1:
                elapsed_time_u_sum_correct[row[0]] += row[3]
                
                if len(timestamp_u_correct[row[0]]) == 1:
                    timestamp_u_correct[row[0]].pop(0)
                    timestamp_u_correct[row[0]].append(row[5])
                else:
                    timestamp_u_correct[row[0]].append(row[5])
            
                
            if row[1] == 0:
                if len(timestamp_u_incorrect[row[0]]) == 1:
                    timestamp_u_incorrect[row[0]].pop(0)
                    timestamp_u_incorrect[row[0]].append(row[5])
                else:
                    timestamp_u_incorrect[row[0]].append(row[5])
            

            answered_correctly_q_sum[row[2]] += row[1]
            # ------------------------------------------------------------------
             
            
    user_df = pd.DataFrame({'answered_correctly_u_avg': answered_correctly_u_avg, 'answered_correctly_u_std': answered_correctly_u_std, 'elapsed_time_u_avg': elapsed_time_u_avg, 'elapsed_time_u_avg_correct': elapsed_time_u_avg_correct,
                            'explanation_u_avg': explanation_u_avg, 'answered_correctly_q_avg': answered_correctly_q_avg, 'elapsed_time_q_avg': elapsed_time_q_avg, 'explanation_q_avg': explanation_q_avg,
                            'answered_correctly_uq_count': answered_correctly_uq_count, 'timestamp_u_recency_1': timestamp_u_recency_1, 'timestamp_u_recency_2': timestamp_u_recency_2,
                            'timestamp_u_recency_3': timestamp_u_recency_3, 'timestamp_u_recency_4': timestamp_u_recency_4, 'timestamp_u_incorrect_recency': timestamp_u_incorrect_recency, 'timestamp_u_correct_recency': timestamp_u_correct_recency,
                            'timelag_mean': timestamp_u_avg, 'timelag_std': timestamp_u_std})
    
    df = pd.concat([df, user_df], axis = 1)
    return df

        
def update_features(df, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, elapsed_time_u_sum_correct, explanation_u_sum, timestamp_u, timestamp_u_sum, timestamp_u_count, timestamp_u_incorrect, timestamp_u_correct, answered_correctly_q_count, answered_correctly_q_sum, elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq):
    for row in df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp', 'content_type_id']].values:
        if row[6] == 0:
            
            # Client features updates
            answered_correctly_u_count[row[0]] += 1
            elapsed_time_u_sum[row[0]] += row[3]
            explanation_u_sum[row[0]] += int(row[4])
            if len(timestamp_u[row[0]]) == 4:
                timestamp_u[row[0]].pop(0)
                timestamp_u[row[0]].append(row[5])
            else:
                timestamp_u[row[0]].append(row[5])
                
            if len(timestamp_u[row[0]]) >= 2:
                timestamp_u_sum[row[0]] += timestamp_u[row[0]][-1] - timestamp_u[row[0]][-2]
            timestamp_u_count[row[0]] += 1
            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_count[row[2]] += 1
            elapsed_time_q_sum[row[2]] += row[3]
            explanation_q_sum[row[2]] += int(row[4])
            # ------------------------------------------------------------------
            # Client Question updates
            answered_correctly_uq[row[0]][row[2]] += 1
            # ------------------------------------------------------------------
            # Flag for training and inference
            # ------------------------------------------------------------------
            # Client features updates
            answered_correctly_u_sum[row[0]] += row[1]
            if row[1] == 1:
                elapsed_time_u_sum_correct[row[0]] += row[3]
                
                if len(timestamp_u_correct[row[0]]) == 1:
                    timestamp_u_correct[row[0]].pop(0)
                    timestamp_u_correct[row[0]].append(row[5])
                else:
                    timestamp_u_correct[row[0]].append(row[5])
            
                
            if row[1] == 0:
                if len(timestamp_u_incorrect[row[0]]) == 1:
                    timestamp_u_incorrect[row[0]].pop(0)
                    timestamp_u_incorrect[row[0]].append(row[5])
                else:
                    timestamp_u_incorrect[row[0]].append(row[5])
            
            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_sum[row[2]] += row[1]
            # ------------------------------------------------------------------
