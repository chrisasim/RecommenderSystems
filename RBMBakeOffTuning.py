# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV
from RBM import RBM

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

#So, at this stage I am going to find out the best optimal parameters for the RBM topology in order to provide the better recommendations not only from accyracy perpective, but also hit rate and top-N recommendation list.

print("Searching for best parameters...")
param_grid = {'hiddenDim': [10,20], 'learningRate': [0.01, 0.001], 'epochs': [20,30], 'batchSize': [20, 30]}
gs = GridSearchCV(RBMAlgorithm, param_grid, measures=['rmse', 'mae'], cv=10)
gs.fit(evaluationData)
print("Best RMSE score attained: ", gs.best_score['rmse'])
print(gs.best_params['rmse'])
param_grid = {'n-epochs': [20,30]}
print(RBM())

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#RBM Bake Off Tuning Chris
RBM = RBMAlgorithm(epochs=20)
# More infor about RBM can be found on: https://www.analyticsvidhya.com/blog/2022/03/a-brief-overview-of-restricted-boltzmann-machine-rbm/
evaluator.AddAlgorithm(RBM, "RBM")

params = gs.best_params['rmse']
#RBMBakeOffTuned = RBMAlgorithm(epochs = [20, 30] """params['epochs']""", hiddenDim = [10,20] """ params['hiddenDim'] """, learningRate = [0.01, 0.001] """ params['learningRate'] """, batchSize = [20,30] """ params['batchSize'] """)
RBMBakeOffTuned = RBMAlgorithm(epochs=[20], hiddenDim=[20], learningRate=[0.001], batchSize=[20])
print(RBMBakeOffTuned.fit(evaluationData))
print(RBMBakeOffTuned)
evaluator.AddAlgorithm(RBMBakeOffTuned, "RBM - Tuned powered by Christodoulos Asiminidis")

# So, we are pitting our RBM against random recommendations which is the following algorithm
# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
