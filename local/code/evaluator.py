import json
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score

class Evaluator():
    def __init__(self,
                 y,
                 y_hat,
                 identity,
                 top_k_list = [10,20,50],
                 threshold_list = [0.5,0.7,0.9]):
        
        self.top_k_list = top_k_list
        self.threshold_list = threshold_list
        self.y = y
        self.y_hat = y_hat
        self.identity = identity
        self.eval_result = dict()
        
    def run(self):
        for k,threshold in tqdm(product(self.top_k_list,
                                        self.threshold_list)):
            precision, recall, accuracy, f1, roc_auc, raw_data = self._classifier_metrics_at_k(k=k, 
                                                                                               threshold=threshold)

            self.eval_result['test_top_'+str(k)+'_threshold_'+str(threshold)+'_precision'] = precision
            self.eval_result['test_top_'+str(k)+'_threshold_'+str(threshold)+'_recall'] = recall
            self.eval_result['test_top_'+str(k)+'_threshold_'+str(threshold)+'_accuracy'] = accuracy
            self.eval_result['test_top_'+str(k)+'_threshold_'+str(threshold)+'_f1_score'] = f1
            self.eval_result['test_top_'+str(k)+'_threshold_'+str(threshold)+'_roc_auc'] = roc_auc
            self.eval_result['test_top_'+str(k)+'_threshold_'+str(threshold)+'_raw'] = raw_data.to_json()
        return self.eval_result
            
    def _classifier_metrics_at_k(self, 
                                 k, 
                                 threshold):
        """Return precision and recall at top k metrics for each user"""

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, true_r, est in zip(self.identity, self.y, self.y_hat):
            user_est_true[uid].append((est, true_r))
            
        precisions = dict()
        recalls = dict()
        accuracies = dict()
        f1_scores = dict()
        roc_auc = dict()
        role_len = dict()
        sus_role_len = dict()
        sus_role_ratio = dict()

        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)
    

            #y_true = [int(true_r >= threshold) for (_, true_r) in user_ratings]
            y_true_k = [int(true_r >= threshold) for (_, true_r) in user_ratings[:k]]
            y_pred_k = [int(est >= threshold) for (est, _) in user_ratings[:k]]
            y_score_k = [est for (est, _) in user_ratings[:k]]
            
            accuracies[uid] = accuracy_score(y_true_k, y_pred_k)
            precisions[uid] = precision_score(y_true_k, y_pred_k, zero_division=0)
            recalls[uid] = recall_score(y_true_k, y_pred_k, zero_division=0)
            f1_scores[uid] = f1_score(y_true_k, y_pred_k, zero_division=0)
            try:
                roc_auc[uid]  = roc_auc_score(y_true_k, y_score_k)
            except ValueError:
                roc_auc[uid]  = 0.5
            role_len[uid] = len(y_pred_k)
            sus_role_len[uid] = sum(y_pred_k)
            sus_role_ratio[uid] = sus_role_len[uid]/role_len[uid]

        precision_mean_std = (np.array(list(precisions.values())).mean(),
                             np.array(list(precisions.values())).std())
        recall_mean_std = (np.array(list(recalls.values())).mean(),
                          np.array(list(recalls.values())).std())
        accuracy_mean_std = (np.array(list(accuracies.values())).mean(),
                            np.array(list(accuracies.values())).std())
        f1_mean_std = (np.array(list(f1_scores.values())).mean(),
                      np.array(list(f1_scores.values())).std())
        roc_auc_mean_std = (np.array([x for x in list(roc_auc.values()) if ~np.isnan(x)]).mean(),
                           np.array([x for x in list(roc_auc.values()) if ~np.isnan(x)]).std())

        raw_data = pd.DataFrame({'uid':precisions.keys(),
                                 'accuracy':accuracies.values(),
                                 'precision':precisions.values(),
                                 'recall':recalls.values(),
                                 'f1_score':f1_scores.values(),
                                 'roc_auc':roc_auc.values(),
                                 'role_len':role_len.values(),
                                 'sus_role_len':sus_role_len.values(),
                                 'sus_role_ratio':sus_role_ratio.values()})
        
        raw_data.set_index('uid',inplace=True)
        
        return (precision_mean_std, 
                recall_mean_std, 
                accuracy_mean_std,
                f1_mean_std,
                roc_auc_mean_std,
                raw_data)