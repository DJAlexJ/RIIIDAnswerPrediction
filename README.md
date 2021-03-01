# RIIIDAnswerPrediction

In RIIID Answer Correctness Prediction competition, the main challenge is to create algorithms for "Knowledge Tracing", the modeling of student knowledge over time. The goal is to accurately predict how students will perform on future interactions. My solution is based on a single lightGBM model utilizing features that I have collected using different approaches:

### Elo Rating
Elo rating system is a method for calculating the relative skill levels where a student and a question are considered to be rivals. Each student can be described with this rating. Initially, it is initialized to zero and then updated each time student answers the question, acting like gradient descent.

### Loop features
These features are recieved based on a common desire to get some statistics such as percentage of correct student's answers, variance of answers, percentage of correct answers to a particular question, mean answering time per student, etc... without looking into the future. The number of such features is limited only by your imagination and memory and time restrictions.

### Trueskill features
Microsoft trueskill rating system looks something like elo rating in the sense that each student is described by a rating representing his ability to win. In short, this rating is a normal distribution with mean mu (general ability to win) and variance sigma (consistency of skills)

The description of trueskill and elo features is in `trueskill_features.py` and `elo_features.py` respectively. Since estimating these ratings requires a lot of time, scripts `trueskill_estimation.py` and `elo_estimation.py` should be run before training the final model. The results are saved in dictionaries and can be used in the future.

After running `train.py`, the model is saved and can be used in inference on test data.
