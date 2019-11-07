import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,f1_score,roc_curve,auc,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from .tools import *
import traceback


def Dt_Gini_Bin_mono(df, col, target='bad', max_bin=5, min_percent=0.15, special=9999):
    '''
    :需要将特殊值处理为空值，分组之后，再将特殊值单独拿来处理
    :param df: 需要分组的数据集
    :param col: 数值型变量名的集合
    :param target: 好坏标签
    :param max_bins: 最大分组数量
    :param min_percent:最小分组占比
    :return: 分组结果
    '''
    cut_off_dict = {}
    Less_Bin_Var = []
    IndexError_Var = []

    dict_bin = {}
    for c in col:
        max_bins = max_bin
        try:
            print("{} is in processing".format(c))
            colLevels = sorted(list(set(df[c])))
            N_distinct = len(colLevels)
            if N_distinct <= max_bins:  # 如果原始属性的取值个数低于max_interval，不执行这段函数
                print("The number of original levels for {} is less than or equal to max intervals".format(c))
                Less_Bin_Var.append(c)
            else:
                DF = df[df[c].notnull()]
                DF = DF[DF[c] != special]
                X, y = pd.concat([DF[c], DF[c]], axis=1), DF[target]
                X, y = DF[c].values.reshape(-1,1), DF[target].values
                clf = DecisionTreeClassifier(max_leaf_nodes=max_bins,
                                             min_samples_leaf=min_percent).fit(X, y)
                n_nodes = clf.tree_.node_count
                children_left = clf.tree_.children_left
                children_right = clf.tree_.children_right
                threshold = clf.tree_.threshold
                boundary = []
                for i in range(n_nodes):
                    if children_left[i] != children_right[i]:
                        boundary.append(threshold[i])
                sorted_b = sorted(boundary)
                df[c + '_Bin'] = df[c].map(lambda x: assignBin(x, sorted_b, special_attribute=[]))
                monotone = badRateMonotone(df, c + '_Bin', target)
                while (not monotone):
                    # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
                    max_bins -= 1
                    clf = DecisionTreeClassifier(max_leaf_nodes=max_bins,
                                                 min_samples_leaf=min_percent).fit(X, y)
                    n_nodes = clf.tree_.node_count
                    children_left = clf.tree_.children_left
                    children_right = clf.tree_.children_right
                    threshold = clf.tree_.threshold
                    boundary = []
                    for i in range(n_nodes):
                        if children_left[i] != children_right[i]:
                            boundary.append(threshold[i])
                    sorted_b = sorted(boundary)
                    df[c + '_Bin'] = df[c].map(lambda x: assignBin(x, sorted_b, special_attribute=[special]))
                    if max_bins == 2:
                        # 当分箱数为2时，必然单调
                        break
                    monotone = badRateMonotone(df, c + '_Bin', target, special_attribute=['Bin_-1'])
                dict_bin.update({c: sorted_b})
        except IndexError:
            # print("{} is IndexError".format(c))
            traceback.print_exc()
            IndexError_Var.append(c)
        except ZeroDivisionError:
            # print("{} is ZeroDivisionError".format(c))
            traceback.print_exc()
            IndexError_Var.append(c)
        except TypeError:
            # print("{} is TypeError".format(c))
            traceback.print_exc()
            IndexError_Var.append(c)
        dict_bin['Less_Bin_Var'] = Less_Bin_Var
        dict_bin['IndexError_Var'] = IndexError_Var
    return dict_bin


def merge_pureness_by_chi2(group, problem_index, sorted_b):
    print(problem_index)
    index = group.index.tolist()
    index = sorted(index, key=lambda x: int(x.split('_')[1]))
    group = group.loc[index, :]
    if len(problem_index) == 0:
        return sorted_b
    if problem_index[0] == 'Bin_-1':
        raise TypeError("feature need to check")
    if problem_index[0] == index[0]:
        sorted_b = sorted_b[1:]
        group.loc[index[1], :] += group.loc[index[0], :]
        group = group.drop(index[0])
    elif problem_index[0] == index[-1]:
        sorted_b = sorted_b[:-1]
        group.loc[index[-2], :] += group.loc[index[-1], :]
        group = group.drop(index[-1])
    else:
        pos = index.index(problem_index[0])
        chi2_1 = calcChi2(group.iloc[pos - 1:pos + 1, :])
        chi2_2 = calcChi2(group.iloc[pos:pos + 2, :])
        if chi2_1 < chi2_2:
            sorted_b.pop(pos - 1)
            group.loc[index[pos], :] += group.loc[index[pos - 1], :]
            group = group.drop(index[pos - 1])
        else:
            sorted_b.pop(pos)
            group.loc[index[pos + 1], :] += group.loc[index[pos], :]
            group = group.drop(index[pos])
    problem_index.pop(0)
    if len(problem_index) > 0:
        return merge_pureness_by_chi2(group, problem_index, sorted_b)
    return sorted_b


# 基于决策树的最优分组方法
def Dt_Gini_Bin(df, col, target='bad', max_bin=5, min_percent=0.15, special=9999):
    '''
    :需要将特殊值处理为空值，分组之后，再将特殊值单独拿来处理
    :param df: 需要分组的数据集
    :param col: 数值型变量名的集合
    :param target: 好坏标签
    :param max_bins: 最大分组数量
    :param min_percent:最小分组占比
    :return: 分组结果
    '''
    cut_off_dict = {}
    Less_Bin_Var = []
    IndexError_Var = []

    dict_bin = {}
    for c in col:
        max_bins = max_bin
        try:
            print("{} is in processing".format(c))
            colLevels = sorted(list(set(df[c])))
            N_distinct = len(colLevels)
            if N_distinct <= max_bins:  # 如果原始属性的取值个数低于max_interval，不执行这段函数
                print("The number of original levels for {} is less than or equal to max intervals".format(c))
                Less_Bin_Var.append(c)
            else:
                DF = df[df[c].notnull()]
                DF = DF[DF[c] != special]
                X, y = DF[c].values.reshape(-1,1), DF[target].values
                clf = DecisionTreeClassifier(max_leaf_nodes=max_bins,
                                             min_samples_leaf=min_percent).fit(X, y)
                n_nodes = clf.tree_.node_count
                children_left = clf.tree_.children_left
                children_right = clf.tree_.children_right
                threshold = clf.tree_.threshold
                boundary = []
                for i in range(n_nodes):
                    if children_left[i] != children_right[i]:
                        boundary.append(threshold[i])
                sorted_b = sorted(boundary)
                df[c + '_Bin'] = df[c].map(lambda x: assignBin(x, sorted_b, special_attribute=[special]))
                group = df.loc[:, [c + '_Bin', target]].groupby(c + '_Bin').agg([sum, len])[target]
                group['check'] = group['sum'] * (group['len'] - group['sum'])

                problem_index = group[group['check'] == 0].index.tolist()
                index = group.index.tolist()
                index = sorted(index, key=lambda x: int(x.split('_')[1]))
                if 'Bin_-1' in index:
                    group = group.loc[index[1:], :]
                else:
                    group = group.loc[index, :]
                if len(problem_index) > 0:
                    print(problem_index)
                    sorted_b = merge_pureness_by_chi2(group, problem_index, sorted_b)  # 根据卡方值合并
                df[c + '_Bin'] = df[c].map(lambda x: assignBin(x, sorted_b, special_attribute=[special]))
                dict_bin.update({c: sorted_b})
        except IndexError:
            # print("{} is IndexError".format(c))
            traceback.print_exc()
            IndexError_Var.append(c)
        except ZeroDivisionError:
            # print("{} is ZeroDivisionError".format(c))
            traceback.print_exc()
            IndexError_Var.append(c)
        except TypeError:
            # print("{} is TypeError".format(c))
            traceback.print_exc()
            IndexError_Var.append(c)
        dict_bin['Less_Bin_Var'] = Less_Bin_Var
        dict_bin['IndexError_Var'] = IndexError_Var
    return dict_bin