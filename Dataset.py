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
            return np.hstack([1-new_probs, new_probs])
        if how=="beta":
            train_probs = train['prob_1'].apply(lambda x: x+0.0001 if x==0 else (x-0.0001 if x==1 else x))
            s1 = np.log(train_probs); s2 = -1*np.log(1-train_probs)
            X = np.column_stack((s1, s2))
            lr = LogisticRegression().fit(X, train['actual'])
            a, b, c = lr.coef_[0][0], lr.coef_[0][1], lr.intercept_[0]
            test_probs = test['prob_1'].apply(lambda x: x+0.0001 if x==0 else (x-0.0001 if x==1 else x))
            new_probs = np.array(1/(1+1/(np.exp(c) * np.power(test_probs, a) / np.power((1-test_probs),b)))).reshape(-1,1)
            return np.hstack([1-new_probs, new_probs])
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


    def split_calibrate(self, var, splits, how='kumar'):
        # split up dataset by given breakpoints
        df_cals = [self.df_cal[self.df_cal[var]<splits[0]].copy()]
        dfs = [self.df[self.df[var]<splits[0]].copy()]
        for i in range(1, len(splits)):
            df_cals.append(self.df_cal[np.logical_and(self.df_cal[var]>=splits[i-1], self.df_cal[var]<splits[i])].copy())
            dfs.append(self.df[np.logical_and(self.df[var]>=splits[i-1], self.df[var]<splits[i])].copy())
        df_cals.append(self.df_cal[self.df_cal[var]>=splits[-1]].copy())
        dfs.append(self.df[self.df[var]>=splits[-1]].copy())
        updated_dfs=[]

        col_names = [n+"_split" for n in self.probs]
        
        # perform calibration separately for each subset
        for i in range(len(dfs)):
            new_probs = self.get_calibrated_probs(df_cals[i].copy(), dfs[i].copy(), how=how)
            df_tmp = pd.DataFrame(new_probs, columns=col_names, index=dfs[i].index)
            updated_dfs.append(pd.merge(dfs[i], df_tmp, left_index=True, right_index=True))

        # recombine
        df = pd.concat(updated_dfs)
        df['pred_error_split'] = df.apply(lambda x: 1-np.max(x[col_names]), axis=1)
        df['err_diff_split'] = df['incorrect'].astype(int) - df['pred_error_split'] 
        self.df = df.copy()


    def compute_VECE(self, var, label='', num_bins=10):
        '''
        compute the expected variable-wise calibration error for a given variable var
        '''
        df = self.df.copy()
        n = len(df)
        df['bin_var'] = pd.qcut(df[var], num_bins, duplicates='drop')
        grouped = df.groupby('bin_var').aggregate({'pred_error'+label:'mean','incorrect':'mean', 'prob_0':'count'})
        grouped['cont'] = (grouped['prob_0']/n)*np.absolute(grouped['pred_error'+label]-grouped['incorrect'])
        vece = sum(grouped['cont'])

        return vece

    
    def compute_ECE(self, label='', num_bins=10):
        '''
        compute the standard expected calibration error
        '''
        df = self.df.copy()
        n = len(df)
        df['bin_score'] = pd.qcut(df['pred_error'], num_bins, duplicates='drop')
        grouped = df.groupby('bin_score').aggregate({'pred_error'+label:'mean','incorrect':'mean', 'prob_0':'count'})
        grouped['cont'] = (grouped['prob_0']/n)*np.absolute(grouped['pred_error'+label]-grouped['incorrect'])
        ece = sum(grouped['cont'])

        return ece

    def max_diff(self, var, s=0.75, label=""):
        '''
        for a given variable, determine the maximum variable-wise difference between 
        error and predicted error
        '''
        x_min, x_max = np.quantile(self.df[var], [0.05, 0.95])
        xerr, yerr, _, _ = self.lowess_smooth(var, 'incorrect',x_min, x_max,s)
        xperr, yperr, _, _ = self.lowess_smooth(var, 'pred_error'+label,x_min, x_max,s)
        max_diff = 0

        for i in range(len(xerr)):
            diff = abs(yerr[i]-yperr[i])
            if diff > max_diff:
                max_diff = diff

        return max_diff


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


    def gen_plot_lowess(self, var, s=0.75, label="", use_lim=False, ylim=None, filename=None, hist=None):
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
        ax1.plot(xperr,yperr, color='red', label="Model's Predicted Error")
        ax1.fill_between(xperr,pll,pul,color='red',alpha=0.3)

        if use_lim:
            ax1.set_ylim(ylim)
        else:
            ax1.set_ylim(max(-1, ax1.get_ylim()[0]), ax1.get_ylim()[1])
            
        if hist:
            ax2 = ax1.twinx()
            (counts, bins) = np.histogram(self.df[var], bins=15)
            ax2.hist(bins[:-1], bins, weights=counts, color='black', label="P("+var+")")
            ax2.set_ylim(-500, ax2.get_ylim()[1]*7)
            ax2.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                right=False,      # ticks along the bottom edge are off
                labelright=False)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, prop={'size': 18}, loc="upper left")
        
        else:
            ax1.legend(prop={'size': 18})
            
        ax1.set_xlabel(var, fontsize=20); ax1.set_ylabel('% Error', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.tick_params(axis='both', which='minor', labelsize=14)
            
        if filename:
            plt.savefig(filename,bbox_inches="tight")
        
        plt.show()

        if not use_lim:
            return ax1.get_ylim()
        
    def reliability_diagram(self, label="", hist_weight=0.0001, filename=None):
        '''
        for a given method label, generate a standard reliability diagram.
        '''

        df = self.df.copy()
        df['confidence'+label] = 1-df['pred_error'+label]
        df['correct'] = df['incorrect'].apply(lambda x: 0 if x==1 else 1)
        b = [0.1*i for i in range(5,11)]
        df['bin'] = pd.cut(df['confidence'+label], bins=b, duplicates='raise')
        grouped = df.groupby('bin').aggregate({
            'confidence'+label:'mean',
            'correct':'mean',
            'prob_0':'count'
            })
        y = grouped['correct']
        s = grouped["confidence"+label]
        widths=[b[i+1] -b[i] for i in range(len(b)-1)]
        x = [(b[i] +b[i+1])/2 for i in range(len(b)-1)]

        f, ax1 = plt.subplots(1, 1, figsize=(9,7))
        ax1.plot([0.5, 1], [0.5, 1], ":", label="Perfectly calibrated", color='gray',linewidth=2.5)
        ax1.plot(s, y, color='red',linewidth=2, label="Model")
        ax1.scatter(s, y, color='red',marker="D",s=15)
        ax1.bar(x, y, width=0.1, color='blue', alpha=0.25)

        data = df['confidence'+label]
        (counts, bins) = np.histogram(data, bins=30)
        ax1.hist(bins[:-1], bins, weights=hist_weight*counts, color='black', label="Density")

        ax1.set_ylabel("Accuracy", fontsize=20)
        ax1.set_xlabel("Confidence", fontsize=20)
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="upper left", prop={'size': 18})

        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.tick_params(axis='both', which='minor', labelsize=14)
        
        if filename:
            plt.savefig(filename,bbox_inches="tight")

        plt.show()


    def reliability_diagram_new(self, label="", filename=None):
        '''
        for a given method label, generate a standard reliability diagram.
        '''

        df = self.df.copy()
        df['confidence'+label] = 1-df['pred_error'+label]
        df['correct'] = df['incorrect'].apply(lambda x: 0 if x==1 else 1)
        b = [0.1*i for i in range(5,11)]
        df['bin'] = pd.cut(df['confidence'+label], bins=b, duplicates='raise')
        grouped = df.groupby('bin').aggregate({
            'confidence'+label:'mean',
            'correct':'mean',
            'prob_0':'count'
            })
        y = grouped['correct']
        s = grouped["confidence"+label]
        widths=[b[i+1] -b[i] for i in range(len(b)-1)]
        x = [((b[i] +b[i+1])/2)*100 for i in range(len(b)-1)]

        f, ax1 = plt.subplots(1, 1, figsize=(9,7))
        ax1.plot([50, 100], [0, 50], ":", label="Perfectly calibrated", color='gray',linewidth=2.5)
        ax1.plot(s*100, y*100-50, color='red',linewidth=2, label="Model")
        ax1.scatter(s*100, y*100-50, color='red',marker="D",s=15)
        ax1.bar(x, y*100-50, width=10, color='blue', alpha=0.25)

        data = df['confidence'+label]*100
        ax2 = ax1.twinx()
        (counts, bins) = np.histogram(data, bins=15)
        ax2.hist(bins[:-1], bins, weights=counts, color='black', label="P(Score)")
        ax2.set_ylim(-500, ax2.get_ylim()[1]*7)
        ax2.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                right=False,      # ticks along the bottom edge are off
                labelright=False)


        ax1.set_ylabel("% Accuracy", fontsize=20)
        ax1.set_xlabel("% Confidence", fontsize=20)
        ax1.set_ylim([-5, 55])
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left", prop={'size': 18})
        
        labs = ax1.get_yticks()
        ax1.set_yticklabels([str(int(j)+50) for j in labs])
        
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.tick_params(axis='both', which='minor', labelsize=14)
        
        if filename:
            plt.savefig(filename,bbox_inches="tight")

        plt.show()
        
    def smoothed_reliability_diagram(self, sm=0.75, label="", filename=None):
        '''
        for a given method label, generate a smoothed reliability diagram.
        '''

        self.df['pred_err_100'+label] = self.df['pred_error'+label]*100
        #self.df['correct'] = self.df['incorrect'].apply(lambda x: 0 if x==1 else 1)
        df = self.df.copy()
        b = [10*i for i in range(0,6)]
        df['bin'] = pd.cut(df['pred_err_100'+label], bins=b, duplicates='raise')
        grouped = df.groupby('bin').aggregate({
            'pred_err_100'+label:'mean',
            'incorrect':'mean',
            'prob_0':'count'
            })
        y = grouped['incorrect']
        s = grouped["pred_err_100"+label]
        widths=[b[i+1] -b[i] for i in range(len(b)-1)]
        x = [((b[i] +b[i+1])/2) for i in range(len(b)-1)]
        
        x_min, x_max = np.quantile(df['pred_err_100'+label], [0.05, 0.95])
        xerr, yerr, ll, ul = self.lowess_smooth('pred_err_100'+label, 'incorrect',x_min, x_max,sm)
        f, ax1 = plt.subplots(1, 1, figsize=(9,7))
        ax1.plot(xerr,yerr, color='blue', label='Actual Error')
        ax1.fill_between(xerr,ll,ul,color='blue',alpha=0.3)

        #f, ax1 = plt.subplots(1, 1, figsize=(9,7))
        ax1.plot([0, 50], [0, 50], ":", label="Perfectly calibrated", color='gray',linewidth=2.5)
        #ax1.scatter(s*100, y*100-50, color='red',marker="D",s=15)
        #ax1.bar(x, y*100, width=10, color='blue', alpha=0.25)

        data = df['pred_err_100'+label]
        ax2 = ax1.twinx()
        (counts, bins) = np.histogram(data, bins=15)
        ax2.hist(bins[:-1], bins, weights=counts, color='black', label="P(Score)")
        ax2.set_ylim(-1200, ax2.get_ylim()[1]*11)
        ax2.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                right=False,      # ticks along the bottom edge are off
                labelright=False)


        ax1.set_ylabel("% Error", fontsize=20)
        ax1.set_xlabel("% Predicted Error", fontsize=20)
        ax1.set_ylim([-5, 55])
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left", prop={'size': 18})
        
        labs = ax1.get_yticks()
        #ax1.set_yticklabels([str(int(j)+50) for j in labs])
        
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.tick_params(axis='both', which='minor', labelsize=14)
        
        if filename:
            plt.savefig(filename,bbox_inches="tight")

        plt.show()