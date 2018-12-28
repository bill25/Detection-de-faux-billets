# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:53:35 2018

@author: Aouissat_salsabil
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from fonctions import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  roc_auc_score, accuracy_score
billets=pd.read_csv('billets.csv',sep=",",encoding = "latin1")
billets.info()

billets['is_genuine'].replace({
        True : 1,
        False : 0}, inplace=True)

billets.info()
pca = PCA(n_components=2).fit(billets)
display_scree_plot(pca)
pcs = pca.components_
pcs
display_circles(pcs, 2, pca, [(0,1)], labels = billets.columns)
X_projected = pca.transform(billets)
display_factorial_planes(X_projected, 2, pca, [(0,1)], labels = billets.index)
kmeans=KMeans(n_clusters=2)
kmeans.fit(billets)
display_factorial_planes(X_projected, 2, pca,
                         [(0,1)],labels = billets.index, 
                         illustrative_var = kmeans.labels_)
len(kmeans.labels_)
is_genuine=billets.is_genuine
billets=billets.drop('is_genuine',axis=1)
newdataframe=billets
newdataframe['is_genuine_Pre']=kmeans.labels_
newdataframe['is_genuine_Pre'].replace({
        0 : 1,
        1 : 0}, inplace=True)

logr=LogisticRegression()
train, test = train_test_split(newdataframe, test_size=0.3, random_state=1)
Xtrain=train.drop('is_genuine_Pre',axis=1)
Ytrain=train.is_genuine_Pre
Xtest=test.drop('is_genuine_Pre',axis=1)
Ytest=test.is_genuine_Pre
logr.fit(Xtrain,Ytrain)
YLogpredict=logr.predict(Xtest)
print('erreur de prédiction :', 1-accuracy_score(Ytest,YLogpredict))
probas = logr.predict_proba(Xtest)
print('AUC :', roc_auc_score(Ytest, probas[:,1])) 

exemple=pd.read_csv('exemple.csv',sep=",",encoding = "latin1")
id1=exemple.id
exemple=exemple.drop('id',axis=1)
newprobas=logr.predict_proba(exemple)
d=0
for i,j in newprobas:
    
    if i>j:
        exemple.at[d, 'is_genuine_Pre'] = 0
       
    else:
        exemple.at[d, 'is_genuine_Pre'] = 1
    d+=1

writer = pd.ExcelWriter('Resultat_Exemple.xlsx')
exemple.to_excel(writer,'Resultat_Exemple')
writer.save()



