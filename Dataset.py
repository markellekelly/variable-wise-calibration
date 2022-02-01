import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from skmisc.loess import loess
from sklearn.linear_model import LogisticRegression

import calibration as cal
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
import warnings
from sklearn.model_selection import GridSearchCV,StratifiedKFold

from utils import bootstrap_ci

random_seed = 0
np.random.seed(random_seed)

class Dataset:
    
    def __init__(self, df, k=2, bins=10, cal=True, cal_size=500):
        
        self.k = k
        self.bins = bins
        self.probs = ["prob_" + str(i) for i in range(0,self.k)]
        df = self.process(df)
        if cal:
            self.split_cal_set(df, cal_size)
            methods = ['kumar','hb','log','beta'] if k==2 else ['kumar','hb','dirichlet']
            for m in methods:
                self.calibrate("_"+m, how=m)
        else:
            self.df = df.copy()

    def process(self, df):

        df['incorrect'] = df['actual'] != df['pred']
        df['pred_error'] = df.apply(lambda x: 1-np.max(x[self.probs]), axis=1)
        df['err_diff'] = df['incorrect'].astype(int) - df['pred_error']
    
        return df

    def split_cal_set(self, df, cal_size):
            self.df_cal = df.sample(n=cal_size, replace=False, random_state=0).copy()
            self.df = df.drop(self.df_cal.index, axis=0).copy()
            self.df_cal.reset_index(drop=True, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

    def get_calibrated_probs(self, train, test, how):

        if how=="kumar" or how=="hb":
            cc = cal.PlattBinnerMarginalCalibrator if how=="kumar" else cal.HistogramMarginalCalibrator
            calibrator = cc(len(train), num_bins=self.bins)
            calibrator.train_calibration(train[self.probs], train['actual'])
            return calibrator.calibrate(test[self.probs])
        if how=="log":
            X = np.array(train['prob_1']).reshape(-1, 1)
            lr = LogisticRegression().fit(X, train['actual'])
            a, b = lr.coef_[0][0], lr.intercept_[0]
            new_probs = np.array(1/(1+1/(np.exp(a*test['prob_1'] + b)))).reshape(-1,1)
            return np.hstack([new_probs, 1-new_probs])
        if how=="beta":
            s1 = np.log(train['prob_1']); s2 = -1*np.log(1-train['prob_1'])
            X = np.column_stack((s1, s2))
            lr = LogisticRegression().fit(X, train['actual'])
            a, b, c = lr.coef_[0][0], lr.coef_[0][1], lr.intercept_[0]
            new_probs = np.array(1/(1+1/(np.exp(c) * np.power(test['prob_1'], a) / np.power((1-test['prob_1']),b)))).reshape(-1,1)
            return np.hstack([new_probs, 1-new_probs])
        if how=="dirichlet":
            calibrator = FullDirichletCalibrator(reg_lambda=[1e-3], reg_mu=None)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            gscv = GridSearchCV(calibrator, param_grid={'reg_lambda':  [1e-3], 'reg_mu': [None]},
                cv=skf, scoring='neg_log_loss')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gscv.fit(np.array(train[self.probs]), np.array(train['actual']))
            return gscv.predict_proba(np.array(test[self.probs]))

       
    def calibrate(self, label, how):
        
        calibrated_probs = self.get_calibrated_probs(self.df_cal, self.df, how)

        col_names = [n+label for n in self.probs]
        tmp = pd.DataFrame(calibrated_probs, columns=col_names)

        self.df = pd.merge(self.df, tmp, left_index=True, right_index=True)
        self.df['pred_error'+label] = self.df.apply(lambda x: 1-np.max(x[col_names]), axis=1)
        self.df['err_diff'+label] = self.df['incorrect'].astype(int) - self.df['pred_error'+label]
