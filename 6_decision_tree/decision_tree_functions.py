#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import glob
from scipy.stats import entropy, ks_2samp
import random
from imblearn.combine import SMOTEENN
import os
import scikitplot as skplt
import seaborn as sns
import dtreeviz 
from collections import Counter 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def kl_divergence(p, q):
    """Compute Kullback-Leibler divergence"""
    return entropy(p, q)

def jensen_shannon_divergence(p, q):
    """Compute Jensen-Shannon divergence"""
    # Normalize distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute KL divergence for both directions
    kl_pq = kl_divergence(p, q)
    kl_qp = kl_divergence(q, p)

    # Compute average of KL divergences
    avg_kl = (kl_pq + kl_qp) / 2

    # Compute square root of average KL divergence
    return np.sqrt(avg_kl)

def random_combination(iterable, r, random_state):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    random.seed(random_state)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def custom_stratified_train_test_split_with_group(X, y, group, predetermined_test_groups = None, test_size=0.25, random_state=None, max_imbalance = 1e-04, max_iter = 10000, **kwargs):
    """Optimize test groups based on class imbalance in y"""
    if predetermined_test_groups is None:
        # Initialize with a random selection of groups
        test_groups = np.random.RandomState(seed=random_state).choice(np.unique(group),
                                                                        size=int(test_size * len(np.unique(group))),
                                                                        replace=False)
        test_indices = []
        for grp in test_groups:
            test_indices.extend(np.where(group == grp)[0])
        
        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # Calculate the initial imbalance in the classes of y based on the original dataset
        p_full = np.bincount(y) / len(y)
        p_train = np.bincount(y_train) / len(y_train)
        p_test = np.bincount(y_test) / len(y_test)
        imbalance = jensen_shannon_divergence(p_train, p_full) + jensen_shannon_divergence(p_test, p_full)
        
        print('Initial test groups: ' + str(test_groups) + ', train-full divergence: ' + str(jensen_shannon_divergence(p_train, p_full)) + ' test-full divergence: ' + str(jensen_shannon_divergence(p_test, p_full)))
        
        # Initialize the best combination of groups
        best_test_groups = np.copy(test_groups)
        best_imbalance = imbalance
        
        # Find the best combination of test groups by trying random combinations and checking if they're better
        
        for seed in range(0,max_iter):

            #generate a random combination
            combo = random_combination(np.unique(group), len(test_groups), seed)

            # Skip the current combination if it's the same as the initial test_groups
            if np.array_equal(combo, test_groups):
                print('same array')
                continue
            
            combo_test_indices = []
            for grp in combo:
                combo_test_indices.extend(np.where(group == grp)[0])
            
            combo_train_indices = np.setdiff1d(np.arange(len(y)), combo_test_indices)
            combo_y_train = y[combo_train_indices]
            combo_y_test = y[combo_test_indices]
            
            # Calculate the imbalance for the current combination
            p_combo_train = np.bincount(combo_y_train) / len(combo_y_train)
            p_combo_test = np.bincount(combo_y_test) / len(combo_y_test)
            combo_imbalance = jensen_shannon_divergence(p_combo_train, p_full) + jensen_shannon_divergence(p_combo_test, p_full)
            
            # If the imbalance is below the threshold, stop the optimization
            if combo_imbalance < max_imbalance:
                break

            # If the imbalance is improved, update the best combination of groups
            if combo_imbalance < best_imbalance:
                best_test_groups = combo
                best_imbalance = combo_imbalance
                print('Better combo found at : ' + str(seed) + ' with groups: ' + str(best_test_groups) + ', train-full divergence: ' + str(jensen_shannon_divergence(p_combo_train, p_full)) + ' test-full divergence: ' + str(jensen_shannon_divergence(p_combo_test, p_full)))
                    
        # Use the best combination of groups
        best_test_indices = []
        for grp in best_test_groups:
            best_test_indices.extend(np.where(group == grp)[0])
        
        best_train_indices = np.setdiff1d(np.arange(len(y)), best_test_indices)
        X_train, X_test = X.loc[best_train_indices], X.loc[best_test_indices]
        y_train, y_test = y[best_train_indices], y[best_test_indices]
        
        print("Final imbalance:" +  str(best_imbalance) + ' found at iteration ' + str(seed))
        print("Groups in test set:", best_test_groups)
        print("Groups in train set:", np.unique(group[best_train_indices]))
    else:

        best_test_indices = []
        for grp in predetermined_test_groups:
            best_test_indices.extend(np.where(group == grp)[0])

        best_train_indices = np.setdiff1d(np.arange(len(y)), best_test_indices)
        X_train, X_test = X.loc[best_train_indices], X.loc[best_test_indices]
        y_train, y_test = y[best_train_indices], y[best_test_indices]
        print("Groups in test set:", predetermined_test_groups)
        print("Groups in train set:", np.unique(group[best_train_indices]))

    return X_train, X_test, y_train, y_test
