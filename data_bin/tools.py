import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def assignBin(x, cutOffPoints,special_attribute=[]):
    '''
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :param special_attribute:  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    '''
    numBin = len(cutOffPoints) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin_{}'.format(0-i)
    if len(cutOffPoints) == 0:
        return 'Bin_0'
    if x <= cutOffPoints[0]:
        return 'Bin_0'
    elif x > cutOffPoints[-1]:
        return 'Bin_{}'.format(numBin-1)
    else:
        for i in range(0,numBin-1):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return 'Bin_{}'.format(i+1)


## 判断某变量的坏样本率是否单调
def badRateMonotone(df, sortByVar, target,special_attribute = []):
    '''
    :param df: 包含检验坏样本率的变量，和目标变量
    :param sortByVar: 需要检验坏样本率的变量
    :param target: 目标变量，0、1表示好、坏
    :param special_attribute: 不参与检验的特殊值
    :return: 坏样本率单调与否
    '''
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = binBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateNotMonotone_list = [badRate[i]<badRate[i+1] and badRate[i] < badRate[i-1] or badRate[i]>badRate[i+1] and badRate[i] > badRate[i-1]
                       for i in range(1,len(badRate)-1)]
    if True in badRateNotMonotone_list:
        return False
    else:
        return True


def binBadRate(df, col, target, grantRateIndicator=0):
    '''
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left') # 每箱的坏样本数，总样本数
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1) # 加上一列坏样本率
    dicts = dict(zip(regroup[col],regroup['bad_rate'])) # 每箱对应的坏样本率组成的字典
    if grantRateIndicator==0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)


def calcChi2(df):
    df2 = df.copy()
    goodRate = sum(df2['sum'])*1.0/sum(df2['len'])
    # 当全部样本只有好或者坏样本时，卡方值为0
    if goodRate in [0,1]:
        return 0
    df2['bad'] = df2.apply(lambda x: x['len'] - x['sum'], axis = 1)
    badRate = sum(df2['bad']) * 1.0 / sum(df2['len'])
    # 期望坏（好）样本个数＝全部样本个数*平均坏（好）样本占比
    df2['badExpected'] = df['len'].apply(lambda x: x*badRate)
    df2['goodExpected'] = df['len'].apply(lambda x: x * goodRate)
    badCombined = zip(df2['badExpected'], df2['bad'])
    goodCombined = zip(df2['goodExpected'], df2['sum'])
    badChi = [(i[0]-i[1])**2/i[0] for i in badCombined]
    goodChi = [(i[0] - i[1]) ** 2 / i[0] for i in goodCombined]
    chi2 = sum(badChi) + sum(goodChi)
    return chi2


def calcWoeIV(df,col,label):
    '''
    :param df: 原始数据
    :param col: 分bin之后的标签
    :return: woe，iv
    '''
    gf = df.loc[:,[label,col]].groupby(col).agg([sum,len])
    gf['good'],gf['total'] = gf.iloc[:,0],gf.iloc[:,1]
    gf['bad'] = gf.iloc[:,1] - gf.iloc[:,0]
    # gf['woe'] = np.log((gf.iloc[:, 0] / sum(gf.iloc[:, 0]+1e-32)) / (gf.iloc[:, 2] / sum(gf.iloc[:, 2]+1e-32)))
    gf['woe'] = np.log((gf['good'] / sum(gf['good'] + 1e-32)) / (gf['bad'] / sum(gf['bad'] + 1e-32)))
    gf['iv'] = ((gf['good'] / sum(gf['good']+1e-32)) - (gf['bad'] / sum(gf['bad']+1e-32))) * gf['woe']
    return gf['woe'],sum(gf['iv'])


def calcKS(model,x,y):
    '''
    :param model: 训练后的模型
    :param x: 输入数据集
    :param y: 目标变量
    :return: KS值
    '''
    score = pd.DataFrame(model.predict_proba(x))
    score = score.rename(columns={1:'score'})
    score = score.reset_index()
    target = pd.DataFrame(y)
    target['target'] = target.iloc[:,0]
    target = target.iloc[:,1]
    target = target.reset_index()
    score = pd.merge(score,target,how='inner')
    K_S = KS(score,'score','target')
    return K_S

def KS(df, score, target):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    df_all = pd.DataFrame({'total':total, 'bad':bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all.sort_index(inplace=True)
    df_all.index = range(len(df_all))
    df_all['badCumRate'] = df_all['bad'].cumsum() / df_all['bad'].sum()
    df_all['goodCumRate'] = df_all['good'].cumsum() / df_all['good'].sum()
    KS = df_all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return max(np.abs(KS))


def corrFilter(df, p):
    '''
    用于查看变量之间的相关系数
    :param df:
    :param p:
    :return:相关性超过p的两两组合[['a','b',0.66],['a','c',0.9]]
    '''
    corr = df.corr()  # 计算各变量的相关性系数
    cc = corr.values
    columns = corr.index
    row, col = np.nonzero((cc != 1) & (np.abs(cc) >= p))
    xticks = []  # x轴标签
    for i in range(len(df.columns[1:])):
        xticks.append('x%s' % i)
    yticks = list(corr.index)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1,
                annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
    ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
    plt.show()
    rr = []
    for i in range(len(row) // 2):
        print(columns[row[i]], '&', columns[col[i]], '=', cc[row[i], [col[i]]])
        rr.append([columns[row[i]], columns[col[i]], cc[row[i], [col[i]]]])
    return rr


def splitData(df, col, numOfSplit, special_attribute=[]):
    '''
    :param df: 按照col排序后的数据集
    :param col: 待分箱的变量
    :param numOfSplit: 切分的组别数
    :param special_attribute: 在切分数据集的时候，某些特殊值需要排除在外
    :return: 把原始细粒度的col重新划分成粗粒度的值，便于分箱中的合并处理
    '''
    df2 = df.copy()
    if special_attribute != []:
        df2 = df.loc[~df[col].isin(special_attribute)]
    N = df2.shape[0]
    n = int(N/numOfSplit)
    splitPointIndex = [i*n for i in range(1,numOfSplit)]
    rawValues = sorted(list(df2[col]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))
    return splitPoint # col中“切分点“右边第一个值


def assignGroup(x, bin):
    '''
    :param x: 某个变量的某个取值
    :param bin: 上述变量的分箱结果
    :return: x在分箱结果下的映射
    '''
    N = len(bin)
    if x<=min(bin):
        return min(bin)
    elif x>max(bin):
        return 10e10
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]

