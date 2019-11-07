from scipy.stats import f
import numpy as np
import pandas as pd


class ANOVA:
    # 方差分析
    def __init__(self,df):
        self.df = df

    def analyse(self,feature='',label='',alpha = 0.05):
        if label == '':
            label_series = self.df.iloc[:,-1]
        else:
            label_series = self.df[label]
        label_cnt = set(label_series)
        t_mean = np.mean(self.df[feature])
        SST = np.var(self.df[feature]) * len(self.df)
        SSR = 0
        for lab in label_cnt:
            tmp = self.df[label_series == lab]
            SSR += len(tmp) * np.square(np.mean(tmp[feature]) - t_mean)
        DFR = (len(label_cnt) - 1)
        DFE = (len(self.df) - len(label_cnt))
        MSR = SSR / DFR
        MSE = (SST - SSR) / DFE
        F_val = MSR / MSE
        p_val = f.sf(F_val, DFR, DFE)
        print("F的值:",F_val,"p值:",p_val)
        # expect_val = f.ppf(alpha,DFR,DFE)
        if p_val < alpha:
            print("显著,拒绝零假设")
            return p_val
        return False