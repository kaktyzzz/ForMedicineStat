# -*- coding: utf-8 -*-
#https://habrahabr.ru/post/202090/
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

from pandas import read_csv, DataFrame, Series
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from scipy import interp
import numpy as np
import pylab as pl

def stat(true, pred, classificator_name):
    class_names = ['class 1 OGSK', 'class 2 Pancreonekros', 'class 3 Pancreotogenniy abscess', 'class 4 Kista', ]
    print 'Classificator report for || ' + classificator_name
    print classification_report(true, pred, target_names=class_names)
    confusion = confusion_matrix(true, pred)
    for i in range(0, len(confusion)):
        tp = confusion[i][i]
        tn = 0
        fp = 0
        fn = 0
        for j in range(0, len(confusion)):
            tn += confusion[j][j]
            fn += confusion[i][j]
            fp += confusion[j][i]
        tn -= tp
        fp -= tp
        fn -= tp

        se = tp / (tp + fn)  # = tpr
        sp = tn / (tn + fp)  # = tnr
        ac = (tp + tn) / (tp + tn + fp + fn)
        pvp = tp / (tp + fp)  # precision
        pvn = tn / (tn + fn)

        print '%s: se = %5f sp = %5f ac = %5f +pv = %5f -pv = %5f' % (class_names[i], se, sp, ac, pvp, pvn)


def ROCanalize(classificator_name, test, prob, pred):
    fpr = dict()
    tpr = dict()
    trhd = dict()
    roc_auc = dict()

    pl.figure()

    stat(test, pred, classificator_name)

    test_bin = label_binarize(test, classes=[1, 2, 3, 4])
    n_classes = test_bin.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], trhd[i] = roc_curve(test_bin[:, i], prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        pl.plot(fpr[i], tpr[i], label='%s (area = %0.2f)' % (class_names[i], roc_auc[i]))

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
    pl.savefig('ROC4/' + classificator_name + '.png')
    # pl.show()

def selectKImportance(model, X, k=5):
    return X[:, model.feature_importances_.argsort()[::-1][:k]]

groups_factors_on_enter = {}
groups_factors_on_enter['01:an'] = range(11, 17)
groups_factors_on_enter['02:zh'] = [17, 18, 20] + range(21-30) + [31, 33]
groups_factors_on_enter['03:obj'] = [35, 37, 39, 41, 43, 45, 47] + range(49, 55) + [61, 63, 65, 67, 69, 71, 73, 75, 77]
groups_factors_on_enter['04:pon'] = range(79, 85)
groups_factors_on_enter['05:oak'] = [85, 87, 89, 91, 93, 95, 97]
groups_factors_on_enter['06:bak'] = [99, 101, 103, 105, 107, 109, 111, 113]
groups_factors_on_enter['07:am'] = [115, 117]
groups_factors_on_enter['08:cr'] = [119]
groups_factors_on_enter['09:orobk'] = range(129, 137)
groups_factors_on_enter['10:uzi'] = [137] + range(139, 142) + [212]
groups_factors_on_enter['11:egdc'] = range(219, 228)
# groups_factors_on_enter['euzi']
groups_factors_on_enter['12:pabbi'] = range(323, 325)
groups_factors_on_enter['13:pabci'] = range(325, 332)
groups_factors_on_enter['14:pabaa'] = [332]
# groups_factors_on_enter['krg']
# groups_factors_on_enter['bim']
groups_factors_before_operate = {}
groups_factors_before_operate['01:an'] = range(11, 17)
groups_factors_before_operate['02:zh'] = [17, 18, 20] + range(21-29) + [30, 32, 34]
groups_factors_before_operate['03:obj'] = [36, 38, 40, 42, 44, 46, 48] + range(55, 61) + [62, 64, 66, 68, 70, 72, 74, 76, 78]
groups_factors_before_operate['04:pon'] = range(79, 85)
groups_factors_before_operate['05:oak'] = [86, 88, 90, 92, 94, 96, 98]
groups_factors_before_operate['06:bak'] = [100, 102, 104, 106, 108, 110, 112, 114]
groups_factors_before_operate['07:am'] = [116, 118]
groups_factors_before_operate['08:cr'] = [120]
groups_factors_before_operate['09:orogk'] = range(121, 129)
groups_factors_before_operate['10:orobk'] = range(129, 137)
groups_factors_before_operate['11:uzi'] = [137, 138] + range(139, 142) + range(142, 219)
groups_factors_before_operate['12:egdc'] = range(219, 228)
groups_factors_before_operate['13:euzi'] = range(228, 323)
groups_factors_before_operate['14:pabbi'] = range(323, 325)
groups_factors_before_operate['15:pabci'] = range(325, 332)
groups_factors_before_operate['16:pabaa'] = [332]
groups_factors_before_operate['17:krg'] = range(333, 340)
groups_factors_before_operate['18:bim'] = range(340, 389)

data = read_csv('pacient.csv', sep=';', header=0)
# header = read_csv('pacient-header.csv', sep=';', header=None)

target = data.iloc[:, 2]
# target1 = data['Forma ODP']
kfold = 5 #количество подвыборок для валидации
itog_val = {} #список для записи результатов кросс валидации разных алгоритмов
models = {}
getlist = []
name_groups = ''
# train = data.iloc[:, 2:]
data_droped = data.drop(['nomer'] , axis=1) #из исходных данных убираем
for name_group in sorted(groups_factors_before_operate.keys()):
    getlist += groups_factors_before_operate[name_group]
    name_groups += '+' + name_group
    train = data_droped.iloc[:, getlist]
    # print train.values
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
        ROCanalize(name_groups + ' | ' + name, ROCtestTRG, probas, pred)

    feature_importance = models['RandomForestClassifier'].feature_importances_
    names_feature = list(ROCtrainTRN.columns.values)
    f_i_zipped = zip(names_feature, feature_importance.tolist())
    f_i_zipped.sort(key = lambda t:t[1], reverse=True)
    print 'Влияние факторов [top 10], %:'
    for n, f in f_i_zipped[:10]:
        print " %10f - %s" % (f, n)
    print 'max: ' + str(feature_importance.max()) + ' min: ' + str(feature_importance.min())
    print '---' * 10
    print ''
    print ''
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