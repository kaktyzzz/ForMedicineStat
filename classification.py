# -*- coding: utf-8 -*-
#https://habrahabr.ru/post/202090/
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

from pandas import read_csv, DataFrame, Series
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from scipy import interp
import numpy as np
import pylab as pl


def ROCanalize(classificator_name, test, prob, pred):
    fpr = dict()
    tpr = dict()
    trhd = dict()
    roc_auc = dict()

    pl.figure()

    print 'Classificator report for ' + classificator_name
    print classification_report(test, pred, target_names=['pancreatitis class 1', 'pancreatitis class 2', 'pancreatitis class 3', 'pancreatitis class 4',])

    test_bin = label_binarize(test, classes=[1, 2, 3, 4])
    n_classes = test_bin.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], trhd[i] = roc_curve(test_bin[:, i], prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        pl.plot(fpr[i], tpr[i], label='class %d (area = %0.2f)' % (i, roc_auc[i]))

    fpr["micro"], tpr["micro"], trhd["micro"] = roc_curve(test_bin.ravel(), prob.ravel())
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
    pl.savefig('ROC/' + classificator_name + '.png')
    # pl.show()


data = read_csv('pacient.csv', sep=';', header=None)

target = data[2]
excludeXbegin = 50
excludeXlst = [i for i in range(excludeXbegin, len(data.count()))]
train = data.drop([0, 1, 2] , axis=1) #из исходных данных убираем
kfold = 5 #количество подвыборок для валидации
itog_val = {} #список для записи результатов кросс валидации разных алгоритмов

models = {}
models['RandomForestClassifier'] = RandomForestClassifier(n_estimators = 70) #в параметре передаем кол-во деревьев
models['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors = 18) #в параметре передаем кол-во соседей
models['LogisticRegression'] = LogisticRegression(penalty='l1', tol=0.01)
models['SVC'] = svm.SVC() #по умолчанию kernek='rbf'
models['SVC'].probability = True
# models['OneVsRestClassifier'] = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#                                  random_state=np.random.RandomState(0)))

for name, model in models.items():
    scores = cross_validation.cross_val_score(model, train, target, cv = kfold)
    itog_val[name] = scores.mean()
print 'Кросс-валидация:'
print itog_val
print ''

ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.5)

for name, model in models.items():
    fit = model.fit(ROCtrainTRN, ROCtrainTRG)
    probas = fit.predict_proba(ROCtestTRN)
    pred = fit.predict(ROCtestTRN)
    ROCanalize(name, ROCtestTRG, probas, pred)

feature_importance = models['RandomForestClassifier'].feature_importances_
print 'Влияние факторов, %:'
for v in feature_importance.tolist():
    print "%10f" % v
print 'max: ' + str(feature_importance.max()) + ' min: ' + str(feature_importance.min())
# print models['RandomForestClassifier'].feature_importances_.tolist()
# #SVC
# model_svc.probability = True
# probas = model_svc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
# fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1], pos_label=1)
# roc_auc  = auc(fpr, tpr)
# pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC', roc_auc))


# Прогнозируем при помощи построенной модели
# model_rfc.fit(train, target)
# result.insert(1,'Survived', model_rfc.predict(test))
# result.to_csv('Kaggle_Titanic/Result/test.csv', index=False)