# -*- coding: utf-8 -*-
#https://habrahabr.ru/post/202090/

from pandas import read_csv, DataFrame, Series
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pylab as pl


data = read_csv('pacient.csv', sep=';', header=None)

target = data[2]
train = data.drop([0, 1, 2], axis=1) #из исходных данных убираем Id пассажира и флаг спасся он или нет

kfold = 5 #количество подвыборок для валидации
itog_val = {} #список для записи результатов кросс валидации разных алгоритмов


model_rfc = RandomForestClassifier(n_estimators = 70) #в параметре передаем кол-во деревьев
model_knc = KNeighborsClassifier(n_neighbors = 18) #в параметре передаем кол-во соседей
model_lr = LogisticRegression(penalty='l1', tol=0.01)
model_svc = svm.SVC() #по умолчанию kernek='rbf'

scores = cross_validation.cross_val_score(model_rfc, train, target, cv = kfold)
itog_val['RandomForestClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(model_knc, train, target, cv = kfold)
itog_val['KNeighborsClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(model_lr, train, target, cv = kfold)
itog_val['LogisticRegression'] = scores.mean()
scores = cross_validation.cross_val_score(model_svc, train, target, cv = kfold)
itog_val['SVC'] = scores.mean()

print itog_val
# DataFrame.from_dict(data = itog_val, orient='index').plot(kind='bar', legend=False)

# pl.clf()
# pl.figure(figsize=(8, 6))


ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.25)
# ROCtestTRG = label_binarize(ROCtestTRG, classes=[1, 2, 3, 4])
# n_classes = ROCtestTRG.shape[1]
n_classes = 4
# #SVC
# model_svc.probability = True
# probas = model_svc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
# fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1], pos_label=1)
# roc_auc  = auc(fpr, tpr)
# pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC', roc_auc))
#RandomForestClassifier
probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
# print probas[:, 1]
# fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1], pos_label=1)
# roc_auc  = auc(fpr, tpr)
# Compute ROC curve and ROC area for each class

fpr = dict()
tpr = dict()
roc_auc = dict()
# print ROCtestTRG[:, 1]
# print probas[:, 1]
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(ROCtestTRG, probas[:, i], pos_label=2)
    roc_auc[i] = auc(fpr[i], tpr[i])
    pl.plot(fpr[i], tpr[i], label='%s ROC class %d (area = %0.2f)' % ('RandomForest', i, roc_auc[i]))
# Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(ROCtestTRG.ravel(), probas.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# pl.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          linewidth=2)
# # #KNeighborsClassifier
# probas = model_knc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
# fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1], pos_label=1)
# roc_auc  = auc(fpr, tpr)
# pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier',roc_auc))
# #LogisticRegression
# probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
# fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1], pos_label=1)
# roc_auc  = auc(fpr, tpr)
# pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_auc))
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc=0, fontsize='small')
pl.show()