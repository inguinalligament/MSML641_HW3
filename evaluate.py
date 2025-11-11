################################################################################################
####        TITLE: MSML HW3                                                                 ####
####        DESCRIPTION: SENTIMENT ANALYSIS - EVALUATE.PY                                   ####
####        AUTHOR: BRADLEY SCOTT                                                           ####
####        UMD ID: 119 775 028                                                             ####
####        DATE: 26OCT2025                                                                 ####
####        REFERENCES USED (see paper for full details):                                   ####
####            ChatGPT 5                                                                   ####
################################################################################################

'''
[BS10262025] ev3_641_000001
[BS10262025] import all necessary modules
'''
from sklearn.metrics import accuracy_score, f1_score

'''
[BS10262025] ev3_641_000005
[BS10262025] Get the model accuracy and F1 score
'''
def evaluate_model(model, X_test, y_test):
    y_prob = model.predict(X_test)
    y_pred = (y_prob.ravel() >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, f1