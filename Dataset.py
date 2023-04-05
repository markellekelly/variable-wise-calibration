import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from skmisc.loess import loess
from sklearn.linear_model import LogisticRegression
from sklearn import tree

import calibration as cal
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
import warnings
from sklearn.model_selection import GridSearchCV,StratifiedKFold


random_seed = 0
np.random.seed(random_seed)
plt.rcParams["font.family"] = "Times New Roman"

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
    
    def compute_accuracy(self, label=None):
        '''
        compute the accuracy of the model. optionally, include a calibration label
        to compute accuracy after post-hoc recalibration
        '''
        
        if label==None:
            return len(self.df[self.df['actual'] == self.df['pred']])/len(self.df)
        col_names = [n+label for n in self.probs]
        self.df['pred'+label] = np.argmax(np.array(self.df[col_names]),axis=1)
        return len(self.df[self.df['actual'] == self.df['pred'+label]])/len(self.df)


    def split_cal_set(self, df, cal_size):
        '''
        given a calibration set size, split the dataframe into calibration (df_cal)
        and test (df) sets
        '''
        self.df_cal = df.sample(n=cal_size, replace=False, random_state=0).copy()
        self.df = df.drop(self.df_cal.index, axis=0).copy()
        self.df_cal.reset_index(drop=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
    def get_lineage(self, tree, feature_names):
        '''
        Parse a sklearn tree and return the rules corresponding to the leaf nodes
        '''
        if tree.tree_.node_count == 1:
            return []
        left = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] if i != -2 else -2 for i in tree.tree_.feature]

        idx = np.argwhere(left == -1)[:,0]     

        def recurse(left, right, child, lineage=None):          
            if lineage is None:
                lineage = [child]
            if child in left:
                parent = np.where(left == child)[0].item()
                split = 'l'
            else:
                parent = np.where(right == child)[0].item()
                split = 'r'

            lineage.append((split, threshold[parent], features[parent]))

            if parent == 0:
                lineage.reverse()
                return lineage
            else:
                return recurse(left, right, parent, lineage)
            
        boxes = []
        new_box = []
        for child in idx:
            for node in recurse(left, right, child):
                if type(node) == np.int64:
                    boxes.append(new_box)
                    new_box = []
                else:
                    new_box.append(node)
        return boxes


    def get_splits(self, var, max_depth=2, min_samples_leaf=10):
        '''
        Given a train and test set, generate decision trees for each variable
            and return the boxes corresponding to the leaf nodes
        '''
        boxes = []
        for var in self.df_cal[[var]].columns:
            sub_train = self.df_cal[[var]]; sub_test = self.df[[var]]
            clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            clf = clf.fit(sub_train, self.df_cal['actual'])
            clf.score(sub_test, self.df['actual'])
            boxes.extend(self.get_lineage(clf, [var]))

        s = set(np.array([item for sublist in boxes for item in sublist]).flatten())
        s.remove('l'); s.remove('r'); s.remove(var)
        return sorted([float(i) for i in s])


    def get_calibrated_probs(self, train, test, how, var):
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

       
    def calibrate(self, label, how, var=None):
        '''
        for a given method (how), calibrate the entire test set using a calibrator
        trained on the calibration set. then add appropriate columns to the dataframe.
        '''
        
        calibrated_probs = self.get_calibrated_probs(self.df_cal, self.df, how, var)

        col_names = [n+label for n in self.probs]
        tmp = pd.DataFrame(calibrated_probs, columns=col_names)

        self.df = pd.merge(self.df, tmp, left_index=True, right_index=True)
        self.df['pred_error'+label] = self.df.apply(lambda x: 1-np.max(x[col_names]), axis=1)
        self.df['err_diff'+label] = self.df['incorrect'].astype(int) - self.df['pred_error'+label]


    def split_calibrate(self, var, splits, how='beta'):
        '''
        perform variable-based tree-based recalibration, given var, a variable to split
        on, splits, breakpoints in the data learned via a decision tree, and how, the
        type of recalibration to apply to each split
        '''
        
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
            new_probs = self.get_calibrated_probs(df_cals[i].copy(), dfs[i].copy(), how=how, var=None)
            df_tmp = pd.DataFrame(new_probs, columns=col_names, index=dfs[i].index)
            updated_dfs.append(pd.merge(dfs[i], df_tmp, left_index=True, right_index=True))

        # recombine
        df = pd.concat(updated_dfs)
        df['pred_error_split'] = df.apply(lambda x: 1-np.max(x[col_names]), axis=1)
        df['err_diff_split'] = df['incorrect'].astype(int) - df['pred_error_split'] 
        self.df = df.copy()
        
        
    def augmented_z_calibrate(self, var, label, degree=1, how='beta'):
        '''
        perform variable-wise augmented-beta (or augmented-logistic) calibration for a particular
        variable var
        degree parameter changes the degree of the corresponding logistic regression wrt var
        how='logistic' performs augmented-logistic instead of augmented-beta
        '''
        train_probs = self.df_cal['prob_1'].apply(lambda x: x+0.0001 if x==0 else (x-0.0001 if x==1 else x))
        test_probs = self.df['prob_1'].apply(lambda x: x+0.0001 if x==0 else (x-0.0001 if x==1 else x))
        if how=='beta':
            s1 = np.log(train_probs); s2 = -1*np.log(1-train_probs)
            cols = [s1, s2]
        elif how=='logistic':
            cols = [train_probs]
        for i in range(1,degree+1):
            cols.append(self.df_cal[var]**i)
        X = np.column_stack(tuple(cols))
        
        lr = LogisticRegression().fit(X, self.df_cal['actual'])
        if how=='beta':
            a, b, c, = lr.coef_[0][0], lr.coef_[0][1], lr.intercept_[0]
            prob_term = np.power(test_probs, a)/np.power((1-test_probs),b)
            extra=1
        elif how=='logistic':
            a, c = lr.coef_[0][0], lr.intercept_[0]
            prob_term = np.exp(test_probs*a)
            extra=0
        var_term = np.exp(c)
        for i in range(1,degree+1):
            var_term *= np.exp(lr.coef_[0][i+extra]*(self.df[var]**i))
        
        new_probs = np.array(1/(1+1/(var_term * prob_term))).reshape(-1,1)
        calibrated_probs = np.hstack([1-new_probs, new_probs])
    
        col_names = [n+label for n in self.probs]
        tmp = pd.DataFrame(calibrated_probs, columns=col_names)

        self.df = pd.merge(self.df, tmp, left_index=True, right_index=True)
        self.df['pred_error'+label] = self.df.apply(lambda x: 1-np.max(x[col_names]), axis=1)
        self.df['err_diff'+label] = self.df['incorrect'].astype(int) - self.df['pred_error'+label]
        


    def compute_VECE(self, var, label='', num_bins=10):
        '''
        compute the expected variable-based calibration error for a given variable var
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
        for a given variable, determine the maximum variable-based difference between 
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
    
    def gen_plot_lowess(self, var, s=0.75, d=-0.009, d2=-5, d3=0, label="", 
                        filename=None, hist=None, bins=15, a=0.05, loc="upper left"):
        ''' 
        plot smoothed actual and predicted model error for a given variable var
        can choose a smoothing factor s or a calibration method label
        option to set y limits from previous graphs to standardize axes
        if hist=True, adds a histogram to the bottom of the plot,
        where y axis start locations can be chosen with parameters 
        d (error) and d2 (density), and error end location with d3
        '''

        x_min, x_max = np.quantile(self.df[var], [a, 1-a])
        xerr, yerr, ll, ul = self.lowess_smooth(var, 'incorrect',x_min, x_max,s)
        xperr, yperr, pll, pul = self.lowess_smooth(var, 'pred_error'+label,x_min, x_max,s)
        
        f, ax1 = plt.subplots(1, 1, figsize=(9,7))
        ax1.plot(xerr,yerr, color='blue', label='Actual Error')
        ax1.fill_between(xerr,ll,ul,color='blue',alpha=0.3)
        ax1.plot(xperr,yperr, color='red', label="Predicted Error")
        ax1.fill_between(xperr,pll,pul,color='red',alpha=0.3)

        ax1.set_ylim(d2, ax1.get_ylim()[1]+d3)
            
        xlim = ax1.get_xlim()
        
        if hist:
            ax2 = ax1.twinx()
            (counts, bins) = np.histogram(self.df[var], bins=bins, density=True)
            ax2.hist(bins[:-1], bins, weights=counts, color='black', alpha=0.4, label="P("+var.split(' ')[0]+")")
            ax2.set_ylim(d, ax2.get_ylim()[1]*8)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            
            ax1.legend(lines + lines2, labels + labels2, prop={'size': 24}, loc=loc)
            ax2.tick_params(axis='y', which='both', right=False, labelright=False)
            ax1.set_xlim(xlim); ax2.set_xlim(xlim)
            
            labs = ax1.get_yticks()
            ax1.set_yticks(labs.tolist())
            ax1.set_yticklabels([int(j) if int(j)>=0 else "" for j in labs])
            
        ax1.set_xlabel(var, fontsize=28); ax1.set_ylabel('% Error', fontsize=28)
        ax1.tick_params(axis='both', which='major', labelsize=24)
        ax1.tick_params(axis='both', which='minor', labelsize=24)
            
        if filename:
            plt.savefig(filename,bbox_inches="tight")
        
        plt.show()

        
    def plot_compare(self, var, label1="", label2="_beta", label3="_split",
                     title1="Uncalibrated", title2="Beta calibration", title3="Variable-based calibration",
                     s=0.9, d=-0.009, d2=-5, d3=0,
                     filename=None, bins=15, a=0.05):
        ''' 
        generate 3 variable-based calibration plots for var with label1, label2, label3
        y axis start locations can be chosen with parameters 
        d (error) and d2 (density), and error end location with d3
        '''

        x_min, x_max = np.quantile(self.df[var], [a, 1-a])
        xerr, yerr, ll, ul = self.lowess_smooth(var, 'incorrect',x_min, x_max,s)
        counts, bins = np.histogram(self.df[var], bins=bins, density=True)
        
        f, axes = plt.subplots(1, 3, figsize=(11*3,7))
        
        for lab, ax, title, let in zip((label1, label2, label3), axes, (title1, title2, title3),('a','b','c')):
            xperr, yperr, pll, pul = self.lowess_smooth(var, 'pred_error'+lab,x_min, x_max,s)
            ax.plot(xerr,yerr, color='blue', label='Actual Empirical Error')
            ax.fill_between(xerr,ll,ul,color='blue',alpha=0.3)
            ax.plot(xperr,yperr, color='red', label="Model's Predicted Error")
            ax.fill_between(xperr,pll,pul,color='red',alpha=0.3)
            
            xlim = ax.get_xlim()
            ax2 = ax.twinx()
            ax2.hist(bins[:-1], bins, weights=counts, color='black', alpha=0.4, label="P("+var+")")
            ax2.set_ylim(d, ax2.get_ylim()[1]*8)
            ax2.tick_params(axis='y', which='both', right=False, labelright=False)
            
            ax.set_xlabel(var, fontsize=36)
            ax.tick_params(axis='both', which='major', labelsize=32)
            ax.tick_params(axis='both', which='minor', labelsize=32)
            ax.set_xlim(xlim); ax2.set_xlim(xlim)
            ax.set_title("("+let+") "+title, fontsize=40, pad=15)

        ylim = (d2, axes[0].get_ylim()[1]+d3)
        axes[0].set_ylim(ylim)
        
        labs = axes[0].get_yticks()
        axes[0].set_yticks(labs.tolist())
        axes[0].set_yticklabels([int(j) if int(j)>=0 else "" for j in labs])
        axes[1].tick_params(labelleft=False); axes[2].tick_params(labelleft=False)
        
        for ax in axes:
            ax.set_ylim(ylim)
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        f.legend(lines + lines2, labels + labels2, prop={'size': 36}, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=3)
            
        axes[0].set_ylabel('% Error', fontsize=36)
        #f.tight_layout()
            
        if filename:
            plt.savefig(filename,bbox_inches="tight")
        
        plt.show()
        
    def rd_compare(self, label1="", label2="_beta", label3="_split",
                     title1="Uncalibrated", title2="Beta calibration", title3="Variable-based calibration",
                     hist_weight=0.00005, filename=None):
        ''' 
        generate 3 reliability diagrams with calibration methods label1, label2, label3
        hist_weight resizes the histogram
        '''
        def get_info(label, b=[0.1*i for i in range(5,11)]):
            df = self.df.copy()
            df['confidence'+label] = 1-df['pred_error'+label]
            df['correct'] = df['incorrect'].apply(lambda x: 0 if x==1 else 1)
            df['bin'] = pd.cut(df['confidence'+label], bins=b, duplicates='raise')
            grouped = df.groupby('bin').aggregate({
                'confidence'+label:'mean',
                'correct':'mean',
                'prob_0':'count'
                })
            y = grouped['correct']
            s = grouped["confidence"+label]
            x = [(b[i] +b[i+1])/2 for i in range(len(b)-1)]
            d = df['confidence'+label]
            return d,s,x,y
        
        f, axes = plt.subplots(1, 3, figsize=(11*3,7))
        
        for lab, ax, title in zip((label1, label2, label3), axes, (title1, title2, title3)):
            
            d,s,x,y = get_info(lab)
            
            ax.plot([0.5, 1], [0.5, 1], ":", label="Perfectly calibrated", color='gray',linewidth=2.5)
            ax.plot(s, y, color='red',linewidth=2, label="Model")
            ax.scatter(s, y, color='red',marker="D",s=15)
            ax.bar(x, y, width=0.1, color='blue', alpha=0.25)

            (counts, bins) = np.histogram(d, bins=30)
            ax.hist(bins[:-1], bins, weights=hist_weight*counts, color='black', label="Density")

            ax.set_xlabel("Confidence", fontsize=36)
            ax.set_ylim([-0.05, 1.05])
            ax.set_title(title, fontsize=40, pad=15)

            ax.tick_params(axis='both', which='major', labelsize=32)
            ax.tick_params(axis='both', which='minor', labelsize=32)


        axes[1].tick_params(labelleft=False); axes[2].tick_params(labelleft=False)
        axes[0].set_ylabel('Accuracy', fontsize=36)
        lines, labels = axes[0].get_legend_handles_labels()
        f.legend(lines, labels, prop={'size': 36}, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=3)

        if filename:
            plt.savefig(filename,bbox_inches="tight")
        
        plt.show()