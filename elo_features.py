import numpy as np
import pandas as pd
from tqdm import tqdm
import math

def estimate_parameters(answers_df, granularity_feature_name1='content_id', granularity_feature_name2='part_bundle_id',
                       granularity_feature_name3='task_container_id'):
    item_parameters = {
        granularity_feature_value1: {"beta": 0, "item_nb_answers": 0}
        for granularity_feature_value1 in np.unique(answers_df[granularity_feature_name1])
    }
    bundle_parameters = {
        granularity_feature_value2: {"epsilon": 0, "bundle_nb_answers": 0}
        for granularity_feature_value2 in np.unique(answers_df[granularity_feature_name2])
    }
    container_parameters = {
        granularity_feature_value3: {"gamma": 0, "container_nb_answers": 0}
        for granularity_feature_value3 in np.unique(answers_df[granularity_feature_name3])
    }
    student_parameters = {
        student_id: {"theta": 0, "student_nb_answers": 0}
        for student_id in np.unique(answers_df.student_id)
    }
    student_parameters_b = {
        student_id_b: {"theta_b": 0, "student_nb_answers_b": 0}
        for student_id_b in np.unique(answers_df.student_id)
    }
    

    print("Parameter estimation is starting...")

    for student_id, item_id, bundle_id, container_id, left_asymptote, answered_correctly in tqdm(
        zip(answers_df.student_id.values, answers_df[granularity_feature_name1].values, answers_df[granularity_feature_name2].values,
            answers_df[granularity_feature_name3].values, answers_df.left_asymptote.values, answers_df.answered_correctly.values)
    ):
        theta = student_parameters[student_id]["theta"]
        theta_b = student_parameters_b[student_id]["theta_b"]
        beta = item_parameters[item_id]["beta"]
        epsilon = bundle_parameters[bundle_id]["epsilon"]
        gamma = container_parameters[container_id]["gamma"]

        item_parameters[item_id]["beta"] = get_new_beta(
            answered_correctly, beta, left_asymptote, theta, item_parameters[item_id]["item_nb_answers"],
        )
        bundle_parameters[bundle_id]["epsilon"] = get_new_epsilon(
            answered_correctly, epsilon, left_asymptote, theta_b, bundle_parameters[bundle_id]["bundle_nb_answers"],
        )
        container_parameters[container_id]["gamma"] = get_new_gamma(
            answered_correctly, gamma, left_asymptote, theta, container_parameters[container_id]["container_nb_answers"],
        )
        student_parameters[student_id]["theta"] = get_new_theta(
            answered_correctly, beta, left_asymptote, theta, student_parameters[student_id]["student_nb_answers"],
        )
        student_parameters_b[student_id]["theta_b"] = get_new_theta(
            answered_correctly, epsilon, left_asymptote, theta_b, student_parameters_b[student_id]["student_nb_answers_b"],
        )
        
        item_parameters[item_id]["item_nb_answers"] += 1
        bundle_parameters[bundle_id]["bundle_nb_answers"] += 1
        container_parameters[container_id]["container_nb_answers"] += 1
        student_parameters[student_id]["student_nb_answers"] += 1
        student_parameters_b[student_id]["student_nb_answers_b"] += 1

    print(f"Theta & beta estimations on {granularity_feature_name1} are completed.")
    print(f"Theta_b & epsilon estimations on {granularity_feature_name2} are completed.")
    print(f"Gamma estimation on {granularity_feature_name3} is completed.")
    return student_parameters, student_parameters_b, item_parameters, bundle_parameters, container_parameters

def update_parameters(answers_df, student_parameters, student_parameters_b, item_parameters, bundle_parameters,
                      container_parameters, granularity_feature_name1='content_id',
                      granularity_feature_name2='part_bundle_id', granularity_feature_name3='task_container_id'):
    
    for student_id, item_id, bundle_id, container_id, left_asymptote, answered_correctly in tqdm(zip(
        answers_df.student_id.values,
        answers_df[granularity_feature_name1].values,
        answers_df[granularity_feature_name2].values,
        answers_df[granularity_feature_name3].values,
        answers_df.left_asymptote.values,
        answers_df.answered_correctly.values)
    ):
        if answered_correctly != -1:
            if student_id not in student_parameters:
                student_parameters[student_id] = {'theta': 0, 'student_nb_answers': 0}
            if student_id not in student_parameters_b:
                student_parameters_b[student_id] = {'theta_b': 0, 'student_nb_answers_b': 0}
            if item_id not in item_parameters:
                item_parameters[item_id] = {'beta': 0, 'item_nb_answers': 0}
            if bundle_id not in bundle_parameters:
                bundle_parameters[bundle_id] = {'epsilon': 0, 'bundle_nb_answers': 0}
            if container_id not in container_parameters:
                container_parameters[container] = {'gamma': 0, 'container_nb_answers': 0}


            theta = student_parameters[student_id]["theta"]
            theta_b = student_parameters_b[student_id]["theta_b"]
            beta = item_parameters[item_id]["beta"]
            epsilon = bundle_parameters[bundle_id]["epsilon"]
            gamma = container_parameters[container_id]["gamma"]

            item_parameters[item_id]["beta"] = get_new_beta(
                answered_correctly, beta, left_asymptote, theta, item_parameters[item_id]["item_nb_answers"],
            )
            bundle_parameters[bundle_id]["epsilon"] = get_new_epsilon(
                answered_correctly, epsilon, left_asymptote, theta_b, bundle_parameters[bundle_id]["bundle_nb_answers"],
            )
            container_parameters[container_id]["gamma"] = get_new_gamma(
                answered_correctly, gamma, left_asymptote, theta, container_parameters[container_id]["container_nb_answers"],
            )
            student_parameters[student_id]["theta"] = get_new_theta(
                answered_correctly, beta, left_asymptote, theta, student_parameters[student_id]["student_nb_answers"],
            )
            student_parameters_b[student_id]["theta_b"] = get_new_theta(
                answered_correctly, epsilon, left_asymptote, theta_b, student_parameters_b[student_id]["student_nb_answers_b"],
            )

            item_parameters[item_id]["item_nb_answers"] += 1
            bundle_parameters[bundle_id]["bundle_nb_answers"] += 1
            container_parameters[container_id]["container_nb_answers"] += 1
            student_parameters[student_id]["student_nb_answers"] += 1
            student_parameters_b[student_id]["student_nb_answers_b"] += 1
        
    return student_parameters, student_parameters_b, item_parameters, bundle_parameters, container_parameters

def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
    return theta + learning_rate_theta(nb_previous_answers) * (
        is_good_answer - probability_of_good_answer(theta, beta, left_asymptote))

def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
    return beta - learning_rate_beta(nb_previous_answers) * (
        is_good_answer - probability_of_good_answer(theta, beta, left_asymptote))

def get_new_gamma(is_good_answer, gamma, left_asymptote, theta, nb_previous_answers):
    return gamma - learning_rate_gamma(nb_previous_answers) * (
        is_good_answer - probability_of_good_answer(theta, gamma, left_asymptote))

def get_new_epsilon(is_good_answer, epsilon, left_asymptote, theta, nb_previous_answers):
    return epsilon - learning_rate_epsilon(nb_previous_answers) * (
        is_good_answer - probability_of_good_answer(theta, epsilon, left_asymptote))

def learning_rate_theta(nb_answers):
    return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

def learning_rate_theta_b(nb_answers):
    return max(0.3 / (1 + 0.005 * nb_answers), 0.02)

def learning_rate_beta(nb_answers):
    return 1 / (1 + 0.05 * nb_answers)

def learning_rate_gamma(nb_answers):
    return 1 / (1 + 0.05 * nb_answers)

def learning_rate_epsilon(nb_answers):
    return 1 / (1 + 0.01 * nb_answers)

def probability_of_good_answer(theta, beta, left_asymptote):
    return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))
