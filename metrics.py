from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import math 
from math import sqrt
import numpy as np


def get_mse(actual, predicted):
    loss = ((actual - predicted) ** 2).mean(axis=0)
    return loss
    

   
def get_accuracy(actual, predicted, threshold):
    correct = 0
    predicted_classes = []
    for prediction in predicted :
      if prediction >= threshold :
        predicted_classes.append(1)
      else :
        predicted_classes.append(0)
    for i in range(len(actual)):
      if actual[i] == predicted_classes[i]:
        correct += 1
    return correct / float(len(actual)) * 100.0



def pred_to_classes(actual, predicted, threshold):
    predicted_classes = []
    for prediction in predicted :
      if prediction >= threshold :
        predicted_classes.append(1)
      else :
        predicted_classes.append(0)
    return predicted_classes
    
#precision
def get_tp(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    tp = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 1 and actual[i] == 1:
       tp += 1
    return tp
    
     
    
def get_fp(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    fp = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 1 and actual[i] == 0:
       fp += 1
    return fp


def get_tn(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    tn = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 0 and actual[i] == 0:
       tn += 1
    return tn


def get_fn(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    fn = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 0 and actual[i] == 1:
       fn += 1
    return fn
    
    
#precision = TP/ (TP + FP)    
def precision(actual, predicted, threshold):
    tp = get_tp(actual, predicted, threshold)
    fp = get_fp(actual, predicted, threshold)
    denominator = tp + fp
    if denominator == 0:  # Handle division by zero
        return 0.0  # Return 0 if there are no predicted positives
    prec = tp / denominator
    return prec
    
    
    
#recall = TP / (TP + FN)   
# sensitivity = recall 
def sensitivity(actual, predicted, threshold):
    tp = get_tp(actual, predicted, threshold) 
    fn = get_fn(actual, predicted, threshold)
    denominator = tp + fn
    if denominator == 0:  # Handle division by zero
        return 0.0  # Return 0 if there are no actual positives
    sens = tp / denominator
    return sens
    

    
#Specificity = TN/(TN+FP)    
def specificity(actual, predicted, threshold):
    tn = get_tn(actual, predicted, threshold)
    fp = get_fp(actual, predicted, threshold)
    denominator = tn + fp
    if denominator == 0:  # Handle division by zero
        return 0.0  # Return 0 if there are no actual negatives
    spec = tn / denominator
    return spec


#f1 score  = 2 / ((1/ precision) + (1/recall))   
def f_score(actual, predicted, threshold):
    prec = precision(actual, predicted, threshold)
    rec = sensitivity(actual, predicted, threshold)
    
    # Handle division by zero cases
    if prec == 0 and rec == 0:
        return 0.0  # If both precision and recall are 0, f1 score is 0
    elif prec == 0 or rec == 0:
        return 0.0  # If either precision or recall is 0, f1 score is 0
    
    f_sc = 2 * prec * rec / (prec + rec)
    return f_sc

   
#mcc = (TP * TN - FP * FN) / sqrt((TN+FN) * (FP+TP) *(TN+FP) * (FN+TP)) 
def mcc(act, pred, thre):
   tp = get_tp(act, pred, thre) 
   tn = get_tn(act, pred, thre)
   fp = get_fp(act, pred, thre)
   fn = get_fn(act, pred, thre)
   
   # Check for division by zero
   denominator = sqrt((tn+fn)*(fp+tp)*(tn+fp)*(fn+tp))
   if denominator == 0:
       return 0.0  # Return 0 if denominator is 0
   
   mcc_met = (tp*tn - fp*fn) / denominator
   return mcc_met
   
   

def auroc(act, pred):
   # Check if all predictions are the same (all 0s or all 1s)
   if len(np.unique(pred)) == 1:
       # Can't calculate AUC with only one prediction value
       return 0.5  # Return 0.5 as default (random classifier)
   
   # Check if all actual labels are the same
   if len(np.unique(act)) == 1:
       # Can't calculate AUC with only one class
       return 0.5  # Return 0.5 as default (random classifier)
   
   return roc_auc_score(act, pred)
  

   
def auprc(act, pred):
   # Check if all predictions are the same
   if len(np.unique(pred)) == 1:
       # Can't calculate AUPRC with only one prediction value
       return np.mean(act)  # Return prevalence as default
   
   # Check if all actual labels are the same
   if len(np.unique(act)) == 1:
       # Can't calculate AUPRC with only one class
       if np.mean(act) == 1:  # If all are positive
           return 1.0
       else:  # If all are negative
           return 0.0
   
   return average_precision_score(act, pred)

