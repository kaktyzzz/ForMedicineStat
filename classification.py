# -*- coding: utf-8 -*-
#https://habrahabr.ru/post/202090/
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

from pandas import read_csv, DataFrame, Series
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import numpy as np
import pylab as pl


def ROCanalize(classificator_name, test, prob):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = test.shape[1]
    pl.figure()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test[:, i], prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        pl.plot(fpr[i], tpr[i], label='class %d (area = %0.2f)' % (i, roc_auc[i]))

    fpr["micro"], tpr["micro"], _ = roc_curve(test.ravel(), prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    pl.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["micro"]),
            linewidth=2)

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    pl.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)

    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC ' + classificator_name)
    pl.legend(loc=0, fontsize='small')
    pl.show()


data = read_csv('pacient.csv', sep=';', header=None)

target = data[2]
train = data.drop([0, 1, 2], axis=1) #из исходных данных убираем

kfold = 5 #количество подвыборок для валидации
itog_val = {} #список для записи результатов кросс валидации разных алгоритмов


model_rfc = RandomForestClassifier(n_estimators = 70) #в параметре передаем кол-во деревьев
model_knc = KNeighborsClassifier(n_neighbors = 18) #в параметре передаем кол-во соседей
model_lr = LogisticRegression(penalty='l1', tol=0.01)
model_svc = svm.SVC() #по умолчанию kernek='rbf'
model_ovrc = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=np.random.RandomState(0)))

scores = cross_validation.cross_val_score(model_rfc, train, target, cv = kfold)
itog_val['RandomForestClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(model_knc, train, target, cv = kfold)
itog_val['KNeighborsClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(model_lr, train, target, cv = kfold)
itog_val['LogisticRegression'] = scores.mean()
scores = cross_validation.cross_val_score(model_svc, train, target, cv = kfold)
itog_val['SVC'] = scores.mean()
scores = cross_validation.cross_val_score(model_ovrc, train, target, cv = kfold)
itog_val['OneVsRestClassifier'] = scores.mean()

print itog_val

ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.7)
ROCtestTRG = label_binarize(ROCtestTRG, classes=[1, 2, 3, 4])

# #SVC
model_svc.probability = True
#probas = model_svc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
# fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1], pos_label=1)
# roc_auc  = auc(fpr, tpr)
# pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC', roc_auc))

#RandomForestClassifier
probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
ROCanalize('RandomForestClassifier', ROCtestTRG, probas)

#KNeighborsClassifier
probas = model_knc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
ROCanalize('KNeighborsClassifier', ROCtestTRG, probas)

#LogisticRegression
probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
ROCanalize('LogisticRegression', ROCtestTRG, probas)

#OneVsRestClassifier
probas = model_ovrc.fit(ROCtrainTRN, ROCtrainTRG).decision_function(ROCtestTRN)
ROCanalize('OneVsRestClassifier', ROCtestTRG, probas)

# Прогнозируем при помощи построенной модели
# model_rfc.fit(train, target)
# result.insert(1,'Survived', model_rfc.predict(test))
# result.to_csv('Kaggle_Titanic/Result/test.csv', index=False)