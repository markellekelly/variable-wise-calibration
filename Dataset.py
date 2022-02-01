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
        '''
        for plotting and calibration, add useful variables to the dataframe
        incorrect: whether model's prediction was incorrect
        pred_error: model's predicted error, equivalent to 1 - confidence
        err_diff: difference between model's actual and predicted error
        '''

        df['incorrect'] = df['actual'] != df['pred']
        df['pred_error'] = df.apply(lambda x: 1-np.max(x[self.probs]), axis=1)
        df['err_diff'] = df['incorrect'].astype(int) - df['pred_error']
    
        return df


    def split_cal_set(self, df, cal_size):
        '''
        given a calibration set size, split the dataframe into calibration (df_cal)
        and test (df) sets
        '''
        self.df_cal = df.sample(n=cal_size, replace=False, random_state=0).copy()
        self.df = df.drop(self.df_cal.index, axis=0).copy()
        self.df_cal.reset_index(drop=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)


    def get_calibrated_probs(self, train, test, how):
        '''
        given a train and test set and a calibration method, compute calibrated
        probabilities. methods:
        hb: standard histogram binning -- Zadrozny et al. (2001)
        kumar: scaling-binning -- Kumar et al. (2019)
        log: logistic calibration, aka Platt scaling -- Platt (1999)
        beta: beta calibration -- Kull et al. (2017)
        dirichlet: dirichlet calibration -- Kull et al. (2019)

        '''

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
        '''
        for a given method (how), calibrate the entire test set using a calibrator
        trained on the calibration set. then add appropriate columns to the dataframe.
        '''
        
        calibrated_probs = self.get_calibrated_probs(self.df_cal, self.df, how)

        col_names = [n+label for n in self.probs]
        tmp = pd.DataFrame(calibrated_probs, columns=col_names)

        self.df = pd.merge(self.df, tmp, left_index=True, right_index=True)
        self.df['pred_error'+label] = self.df.apply(lambda x: 1-np.max(x[col_names]), axis=1)
        self.df['err_diff'+label] = self.df['incorrect'].astype(int) - self.df['pred_error'+label]
        

    def var_wise_bins(self, metric):
        '''
        for a given metric (e.g. error rate), for each variable-wise bin, 
        compute bootstrapped confidence intervals for plotting
        '''
        
        grouped = self.df.groupby('bin').aggregate({metric:['mean','std','count']})
        grouped.columns = grouped.columns.get_level_values(1)
        grouped.reset_index(inplace=True)
        edges, heights = stepwise_vals(grouped)

        heights_ll = []; heights_ul = []
        for row in grouped.iterrows():
            data = self.df[self.df['bin']==row[1]['bin']][metric]
            l, u = bootstrap_ci(data)
            heights_ll.append(l); heights_ul.append(u)
        heights_ll.append(l); heights_ul.append(u)
        
        return edges, heights, heights_ll, heights_ul


    def gen_plots_binned(self, var, bins=10, label="", title=None, return_coord=False, ax1c=None, ax2c=None):
        '''
        generate a suite of binned plots over a variable of interest var. useful for debugging
        and investigation. includes error and predicted error vs. var, their difference vs. var,
        and a histogram of the var
        '''

        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,11))

        # compute bins, means, and CIs for vars of interest
        self.df['bin'] = pd.qcut(self.df[var], bins, duplicates='drop')
        edges_p, heights_p, heights_ll_p, heights_ul_p = self.var_wise_bins('pred_error'+label)
        edges_a, heights_a, heights_ll_a, heights_ul_a = self.var_wise_bins('incorrect')
        edges, heights, heights_ll, heights_ul = self.var_wise_bins('err_diff'+label)

        # create the corresponding stair plots
        def stair_plot(ax, x, e, ll, ul, label, col):
            ax.stairs(x, edges=e, baseline=None, color=col, label=label)
            ax.fill_between(e, ll, ul, step="post", alpha=0.3, color=col)
        
        m = min(min(heights_ll_p), min(heights_ll_a))
        stair_plot(ax1, heights_p, edges_p, heights_ll_p, heights_ul_p, 'Predicted Error Rate', 'red')
        stair_plot(ax1, heights_a, edges_a, heights_ll_a, heights_ul_a, 'Actual Error Rate', 'blue')
        stair_plot(ax2, heights, edges, heights_ll, heights_ul, None, 'black')

        if ax1c is None:
            ax1.set_ylim(m-0.02, ax1.get_ylim()[1])
        else:
            ax1.set_ylim(ax1c); ax2.set_ylim(ax2c)
        
        ax2.axhspan(ymin=0, ymax=ax2.get_ylim()[1],color='green',alpha=0.05)
        ax2.axhspan(ymin=ax2.get_ylim()[0], ymax=0,color='red',alpha=0.05)
        ax2.axhline(y=np.mean(heights),color='orange',label='mean EE')
        
        # add axis titles and legends
        ax1.set_xlabel(var, fontsize=16); ax1.set_ylabel("% Error", fontsize=16)
        ax2.set_xlabel(var, fontsize=16); ax2.set_ylabel("% Error", fontsize=16)
        ax1.legend(loc='lower right'); #ax2.legend()
        if title:
            ax1.set_title(title, fontsize=22)
        plt.show()
        
        # add kernel density estimate of var
        plt.figure(figsize=(9,2))
        self.df[var].plot.kde(bw_method=0.3, color='black')
        plt.xlim(ax2.get_xlim())
        plt.show()

        if return_coord:
            return ax1.get_ylim(), ax2.get_ylim()

    def lowess_smooth(self, x, y, x_min, x_max, s=0.75):
        ''' 
        perform lowess smoothing given x and y
        '''
        
        l = loess(self.df[x],self.df[y]*100, family='symmetric',span=s,iterations=0)
        l.fit()
        
        x_vals = np.linspace(x_min,x_max,500)
        pred = l.predict(x_vals,stderror=True)
        conf = pred.confidence()
        y_new = pred.values
        ll = conf.lower
        ul = conf.upper

        return x_vals, y_new, ll, ul

    def gen_plot_lowess(self, var, s=0.75, label="", use_lim=False, ylim=None):
        ''' 
        plot smoothed actual and predicted model error for a given variable var
        can choose a smoothing factor s or a calibration method label
        option to set y limits from previous graphs to standardize axes
        '''

        x_min, x_max = np.quantile(self.df[var], [0.05, 0.95])
        xerr, yerr, ll, ul = self.lowess_smooth(var, 'incorrect',x_min, x_max,s)
        xperr, yperr, pll, pul = self.lowess_smooth(var, 'pred_error'+label,x_min, x_max,s)
        
        f, ax1 = plt.subplots(1, 1, figsize=(9,7))
        ax1.plot(xerr,yerr, color='blue', label='Actual Error')
        ax1.fill_between(xerr,ll,ul,color='blue',alpha=0.3)
        ax1.plot(xperr,yperr, color='red', label='Predicted Error')
        ax1.fill_between(xperr,pll,pul,color='red',alpha=0.3)

        ax1.legend(prop={'size': 14})
        ax1.set_xlabel(var, fontsize=16); ax1.set_ylabel('% Error', fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.tick_params(axis='both', which='minor', labelsize=12)

        if use_lim:
            ax1.set_ylim(ylim)
        else:
            ax1.set_ylim(max(-1, ax1.get_ylim()[0]), ax1.get_ylim()[1])
        
        plt.show()

        if not use_lim:
            return ax1.get_ylim()