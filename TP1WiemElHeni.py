import numpy as np
from mlxtend.frequent_patterns import apriori
import itertools

def a_Priori(cross_tab, MIN_SUP, MAX):
    s = cross_tab.shape[0]
    allProducts = np.sort(pd.unique(df.tProduct))
    end = False
    n = 0
    itemset = list(itertools.product(allProducts[:]))
    while not end :
        if n != 0 :
            t = []
            itemset = pd.unique(C.itemsets.values.tolist())
            for i in range(0, len(itemset)):
                for j in range(0, n):
                    t.extend(itemset[i][:])      
            itemset = pd.unique(list(itertools.combinations(pd.unique(t), n+1)))
        support = np.zeros((len(itemset)))
        for i in range(0, len(itemset)) :
            tab = []
            for j in range(0, len(itemset[i])) :
                tab.append(cross[itemset[i][j]].values.tolist())
            if n ==0 :
                support[i] = sum(cross[allProducts[i]].values.tolist()) / s
            else : 
                tab = [sum(x) for x in zip(*tab)]
                tab = list(filter(lambda x: x == n+1 , tab))
                support[i] = len(tab) / s
        C = pd.DataFrame({'itemsets': itemset, 'support': support})
        C = C.drop(C[C.support < MIN_SUP].index)
        if n == 0 :
            apriori = C 
        else :
            apriori = pd.concat([apriori, C], axis=0)
        n = n + 1
        if C.size == 0 or n == MAX:
            end = True
    apriori = apriori.reset_index(drop=True)
    return apriori

import pandas as pd
from google.colab import files
uploaded = files.upload()

df = pd.read_table('market_basket.txt', header=0, names=['ID', 'tProduct'])

print(df.head(10))

print(df.shape)

def binary(idCad, df):
    unique_products = np.sort(pd.unique(df.tProduct))
    products = df.query('ID == '+idCad)['tProduct'].values.tolist()
    res = np.zeros([len(unique_products)])
    for i in range(0, len(unique_products)) :
        if products.__contains__(unique_products[i]) :
            res[i] = 1
    return res

def table_binaire(df):
    ids = pd.unique(df.ID)
    res = []
    for i in range(0, len(ids)):
        res.append(binary("{}".format(ids[i]), df))
    return res

print(np.array(table_binaire(df)))

cross = pd.crosstab(df.ID, df.tProduct)
print(cross)

print(pd.crosstab(df.ID, df.tProduct).iloc[:30,:3])

aprioriRes = a_Priori(cross, 0.025, 4)

print(aprioriRes.head(15))

def is_inclus(x,items):
    return items.issubset(x)

print(aprioriRes.loc[np.where(aprioriRes.itemsets.apply(is_inclus, items={'Aspirin'}))])

print(aprioriRes.loc[np.where(aprioriRes.itemsets.apply(lambda x,ensemble:ensemble.issubset(x),ensemble={'Aspirin', 'Eggs'}))])

from mlxtend.frequent_patterns import association_rules
regles = association_rules(aprioriRes, metric="confidence", min_threshold=0.75)

print(regles.head(5))

print(regles[regles['lift'].ge(7)])

print(regles[regles['consequents'].eq({'2pct_Milk'})])