import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from skmisc.loess import loess
from sklearn.linear_model import LogisticRegression

import calibration as cal
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from sklearn.model_selection import GridSearchCV,StratifiedKFold

from utils import bootstrap_ci


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

random_seed = 0
np.random.seed(random_seed)

class Dataset:
    
    def __init__(self, df, k=2, cal=True, cal_size=500):
        
        self.k = k
        self.probs = ["prob_" + str(i) for i in range(0,self.k)]
        df = self.process(df)
        if cal:
            self.df_cal = df.sample(n=cal_size, replace=False, random_state=0).copy()
            self.df = df.drop(self.df_cal.index, axis=0).copy()
        else:
            self.df = df.copy()

    def process(self, df):

        df['incorrect'] = df['actual'] != df['pred']
        df['pred_error'] = df.apply(lambda x: 1-np.max(x[self.prob_names]), axis=1)
        df['err_diff'] = df['incorrect'].astype(int) - df['pred_error']
    
        return df