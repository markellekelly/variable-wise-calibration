import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from skmisc.loess import loess
from sklearn.linear_model import LogisticRegression

import calibration as cal

random_seed = 0
np.random.seed(random_seed)

def add_vars(df):
    
    df['pred_error'] = 1-np.maximum(df['prob_0'], df['prob_1'])
    df['incorrect'] = df['actual'] != df['pred']
    df['err_diff'] = df['incorrect'].astype(int) - df['pred_error']
    
    return df

def bootstrap_ci(data, n=1000, func=np.mean, p=0.95):

    sample_size = len(data)
    simulations = [func(np.random.choice(data, size=sample_size, replace=True)) for i in range(n)]
    simulations.sort()
    u_pval = (1+p)/2.
    l_pval = (1-u_pval)
    l_indx = int(np.floor(n*l_pval))
    u_indx = int(np.floor(n*u_pval))
    
    return(simulations[l_indx],simulations[u_indx])

def calib(df, bins=15, how='kumar'):
    '''
    train calibrator 
    '''
    
    probs = df[['prob_0','prob_1']]
    if how=="kumar":
        calibrator = cal.PlattBinnerMarginalCalibrator(len(df), num_bins=bins)
    elif how=="hb":
        calibrator = cal.HistogramMarginalCalibrator(len(df), num_bins=bins)
    calibrator.train_calibration(probs, df['actual'])
    
    return calibrator

def stepwise_vals(grouped):
    edges=[grouped.iloc[0]['bin'].left]
    for row in grouped.iterrows():
        b = row[1]['bin']
        edges.append(b.right)
    heights = np.array(grouped['mean'])
    return edges, heights


class Dataset:
    
    def __init__(self, df, cal_size=500, cal_bins=10, cal=True):
        df = add_vars(df)
        self.df_cal = df.sample(n=cal_size, replace=False, random_state=0)
        self.df = df.drop(self.df_cal.index, axis=0).copy()
        if cal:
            hb_cal = calib(self.df_cal, how='hb')
            self.get_calibrated_probs(hb_cal, "_hb")
            kumar_cal = calib(self.df_cal, how='kumar')
            self.get_calibrated_probs(hb_cal, "_kumar")
            self.add_calibrated_probs(self.logistic_calibrate(self.df_cal, self.df), "_log")
            self.add_calibrated_probs(self.beta_calibrate(self.df_cal, self.df), "_beta")

    def reliability_diagram(self, label="", hist_weight=0.0001):

        df = self.df.copy()
        df['confidence'+label] = 1-df['pred_error'+label]
        df['correct'] = df['incorrect'].apply(lambda x: 0 if x==1 else 1)
        b = [0.1*i for i in range(5,11)]
        df['bin'] = pd.cut(df['confidence'+label], bins=b, duplicates='raise')
        grouped = df.groupby('bin').aggregate({'confidence'+label:'mean','correct':'mean', 'prob_0':'count'})
        y = grouped['correct']
        s = grouped["confidence"+label]
        widths=[b[i+1] -b[i] for i in range(len(b)-1)]
        x = [(b[i] +b[i+1])/2 for i in range(len(b)-1)]
        data = df['confidence'+label]

        f, ax1 = plt.subplots(1, 1, figsize=(9,7))
        ax1.plot([0.5, 1], [0.5, 1], ":", label="Perfectly calibrated", color='gray',linewidth=2.5)
        ax1.plot(s, y,color='red',linewidth=2, label="Model")
        ax1.scatter(s, y,color='red',marker="D",s=15)
        ax1.bar(x, y, width=0.1, color='blue', alpha=0.25)

        (counts, bins) = np.histogram(data, bins=30)

        factor = 2
        ax1.hist(bins[:-1], bins, weights=hist_weight*counts, color='black', label="Density")

        ax1.set_ylabel("Accuracy", fontsize=16)
        ax1.set_xlabel("Confidence", fontsize=16)
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="upper left", prop={'size': 14})

        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.tick_params(axis='both', which='minor', labelsize=12)

        plt.show()

        
    def get_calibrated_probs(self, calibrator, label):
        '''
        calibrate probabilities and add appropriate columns to the df
        '''
        
        probs = self.df[['prob_0','prob_1']]
        calibrated_probs = calibrator.calibrate(probs)

        cols = [i+label for i in ['prob_1','pred_error','err_diff']]
        
        self.df[cols[0]] = calibrated_probs.transpose()[1]
        self.df[cols[1]] = 1-np.maximum(self.df[cols[0]], 1-self.df[cols[0]])
        self.df[cols[2]] = self.df['incorrect'].astype(int) - self.df[cols[1]]

    def compute_VECE(self, var, label='', num_bins=10):
        df = self.df.copy()
        n = len(df)
        df['bin_var'] = pd.qcut(df[var], num_bins, duplicates='drop')
        grouped = df.groupby('bin_var').aggregate({'pred_error'+label:'mean','incorrect':'mean', 'prob_0':'count'})
        grouped['cont'] = (grouped['prob_0']/n)*np.absolute(grouped['pred_error'+label]-grouped['incorrect'])
        vece = sum(grouped['cont'])
        return vece
    
    def compute_ECE(self, label='', num_bins=10):
        df = self.df.copy()
        n = len(df)
        df['bin_score'] = pd.qcut(df['pred_error'], num_bins, duplicates='drop')
        grouped = df.groupby('bin_score').aggregate({'pred_error'+label:'mean','incorrect':'mean', 'prob_0':'count'})
        grouped['cont'] = (grouped['prob_0']/n)*np.absolute(grouped['pred_error'+label]-grouped['incorrect'])
        ece = sum(grouped['cont'])
        return ece

    def add_calibrated_probs(self, probs, label):
        '''
        given calibrated probabilities, add them and appropriate columns to the df
        '''
        cols = [i+label for i in ['prob_1','pred_error','err_diff']]
        
        self.df[cols[0]] = list(probs)
        self.df[cols[1]] = 1-np.maximum(self.df[cols[0]], 1-self.df[cols[0]])
        self.df[cols[2]] = self.df['incorrect'].astype(int) - self.df[cols[1]]

    def logistic_calibrate(self, cal, test):
        y_train, s_train = cal['actual'], cal['prob_1']
        s_test = test['prob_1']
        X = np.array(s_train).reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(X, y_train)
        a = lr.coef_[0][0]
        c = lr.intercept_[0]
        p_test = 1/(1+1/(np.exp(a*s_test + c)))
        return p_test
    
    def beta_calibrate(self, cal, test):
        y_train, s_train = cal['actual'], cal['prob_1'].apply(lambda x: x+0.0001 if x==0 else (x-0.0001 if x==1 else x))
        s_test = test['prob_1'].apply(lambda x: x+0.0001 if x==0 else (x-0.0001 if x==1 else x))
        s1 = np.log(s_train); s2 = -1*np.log(1-s_train)
        X = np.column_stack((s1, s2))
        lr = LogisticRegression()
        lr.fit(X, y_train)
        a, b = lr.coef_[0][0], lr.coef_[0][1]
        d = lr.intercept_[0]
        p_test = 1/(1+1/(np.exp(d) * np.power(s_test, a) / np.power((1-s_test),b)))
        return p_test

    def split_calibrate(self, var, splits, how='kumar'):
        df_cals = [self.df_cal[self.df_cal[var]<splits[0]].copy()]
        dfs = [self.df[self.df[var]<splits[0]].copy()]
        for i in range(1, len(splits)):
            df_cals.append(self.df_cal[np.logical_and(self.df_cal[var]>=splits[i-1], self.df_cal[var]<splits[i])].copy())
            dfs.append(self.df[np.logical_and(self.df[var]>=splits[i-1], self.df[var]<splits[i])].copy())
        df_cals.append(self.df_cal[self.df_cal[var]>=splits[-1]].copy())
        dfs.append(self.df[self.df[var]>=splits[-1]].copy())
        
        for i in range(len(dfs)):
            if how=='kumar':
                kumar1 = cal.PlattBinnerMarginalCalibrator(len(df_cals[i]), num_bins=10)
                kumar1.train_calibration(df_cals[i][['prob_0','prob_1']], df_cals[i]['actual'])
                dfs[i]['prob_1_split'] = kumar1.calibrate(dfs[i][['prob_0','prob_1']]).transpose()[1] 
            elif how=='log':
                dfs[i]['prob_1_split'] = self.logistic_calibrate(df_cals[i], dfs[i])
            elif how=='beta':
                dfs[i]['prob_1_split'] = self.beta_calibrate(df_cals[i], dfs[i])

        df = pd.concat(dfs)
        df['pred_error_split']=1-np.maximum(df['prob_1_split'], 1-df['prob_1_split'])
        df['err_diff_split'] = df['incorrect'].astype(int) - df['pred_error_split'] 
        self.df = df.copy()

    def group(self, var):
        grouped = self.df.groupby('bin').aggregate({var:['mean','std','count']})
        grouped.columns = grouped.columns.get_level_values(1)
        grouped.reset_index(inplace=True)
        return grouped
        
    def bins_for_var(self, var):
        
        grouped = self.group(var)
        edges, heights = stepwise_vals(grouped)

        heights_ll = []; heights_ul = []
        for row in grouped.iterrows():
            data = self.df[self.df['bin']==row[1]['bin']][var]
            l, u = bootstrap_ci(data)
            heights_ll.append(l); heights_ul.append(u)
        heights_ll.append(l); heights_ul.append(u)
        
        return edges, heights, heights_ll, heights_ul

    def gen_binned_plots(self, var, bins=10, label="", title=None, return_coord=False, ax1c=None, ax2c=None):

        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,11))

        # compute bins, means, and CIs for vars of interest
        self.df['bin'] = pd.qcut(self.df[var], bins, duplicates='drop')
        edges_p, heights_p, heights_ll_p, heights_ul_p = self.bins_for_var('pred_error'+label)
        edges_a, heights_a, heights_ll_a, heights_ul_a = self.bins_for_var('incorrect')
        edges, heights, heights_ll, heights_ul = self.bins_for_var('err_diff'+label)

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
            ax1.set_ylim(ax1c)
            ax2.set_ylim(ax2c)
        
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

    def plot_comparison(self, var, labels, titles):
        lim1, lim2 = self.gen_binned_plots(var, label=labels[0], title=titles[0], return_coord=True)
        for l, t in zip(labels[1:], titles[1:]):
            self.gen_binned_plots(var, label=l, title=t, ax1c=lim1, ax2c=lim2)

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

    def gen_plots_lowess(self, var, s=0.75, label=""):
        ''' 
        plot smoothed actual and predicted model error, and their difference.
        again, only a tool for detecting regions of interest.
        '''
        xerr, yerr, ll, ul = self.lowess_smooth(var, 'incorrect',s)
        xperr, yperr, pll, pul = self.lowess_smooth(var, 'pred_error'+label,s)
        xdiff, diff, diffll, difful = self.lowess_smooth(var, 'err_diff'+label,s)
        
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,11))
        ax1.plot(xerr,yerr, color='blue', label='Actual Error')
        ax1.fill_between(xerr,ll,ul,color='blue',alpha=0.3)
        ax1.plot(xperr,yperr, color='red', label='Predicted Error')
        ax1.fill_between(xperr,pll,pul,color='red',alpha=0.3)
        ax1.legend()
        ax1.set_xlabel(var); ax1.set_ylabel('error')
        
        ax2.plot(xerr, diff, color='black')
        ax2.fill_between(xerr,diffll,difful,color='black',alpha=0.3)
        
        ax2.axhspan(ymin=0, ymax=ax2.get_ylim()[1],color='green',alpha=0.05)
        ax2.axhspan(ymin=ax2.get_ylim()[0], ymax=0,color='red',alpha=0.05)
        
        ax2.set_xlabel(var); ax2.set_ylabel('error')
        
        ax1.set_xlabel(var, fontsize=16); ax1.set_ylabel('% Error', fontsize=16)
        ax2.set_xlabel(var, fontsize=16); ax2.set_ylabel('% Error', fontsize=16)
        
        plt.show()

    def max_diff(self, var, s=0.75, label=""):
        x_min, x_max = np.quantile(self.df[var], [0.05, 0.95])
        xerr, yerr, _, _ = self.lowess_smooth(var, 'incorrect',x_min, x_max,s)
        xperr, yperr, _, _ = self.lowess_smooth(var, 'pred_error'+label,x_min, x_max,s)
        max_diff = 0
        for i in range(len(xerr)):
            diff = abs(yerr[i]-yperr[i])
            if diff > max_diff:
                max_diff = diff

        return max_diff

    def gen_plot_lowess(self, var, s=0.75, label="", use_lim=False, ylim=None):
        ''' 
        plot smoothed actual and predicted model error on a single, nicely-formatted plot.
        option to set y limits from previous graphs to standardize
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
        ax1.set_xlabel(var); ax1.set_ylabel('error')
        
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