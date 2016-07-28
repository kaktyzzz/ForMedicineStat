# -*- coding: utf-8 -*-

from collections import deque

import numpy
from scipy.cluster import *
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from shared import get_data


def kmeans_export(centroids, data, labels):
    """Export kmeans result"""

    res = [[] for i in xrange(len(centroids))]
    d = cdist(numpy.array(data), centroids, 'euclidean')
    for i, l in enumerate(d):
        res[l.tolist().index(l.min())].append((labels[i], data[i]))

    return res


# def kmeans_draw(clusters):
#     """Drawing kmeans clustering result"""
#
#     colors = deque(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
#     fig = plt.figure()
#
#     # Prior to version 1.0.0, the method of creating a 3D axes was different. For those using older versions of matplotlib,
#     # change ax = fig.add_subplot(111, projection='3d') to ax = Axes3D(fig).
#     ax = Axes3D(fig)
#     for cluster in clusters:
#         color = colors.popleft()
#         for name, coord in cluster:
#             x, y, z = coord
#             ax.plot3D([x], [y], [z], marker='o', c=color)
#
#     ax.set_xlabel(u'Белки')
#     ax.set_ylabel(u'Жиры')
#     ax.set_zlabel(u'Углеводы')
#     plt.show()


if __name__ == '__main__':
    names, data = get_data(5, 336) # 13 ,19, 37, 81, 86, 100, 116, 120, 138, 161, 169, 209, 215, 224, 319, 329, 336, до конца
    centroids = vq.kmeans(numpy.array(data), 4, iter=100)[0]
    K_res = kmeans_export(centroids, data, names)

    true_list = [];
    pre_list = [];

    for i, group in enumerate(K_res):
        for item in group:
            label, data = item;

            labelsplit = label.split('|')[1].split(' ');
            true_list.append(i+1);
            pre_list.append(int(labelsplit[0]));

            print i+1, label;
    print 'Точность: ' + str(accuracy_score(true_list, pre_list)) #In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    print 'Специфичность!: ' + str(precision_score(true_list, pre_list, average='binary')) #tp / (tp + fp)
    print 'Чувствительность!: ' + str(recall_score(true_list, pre_list, average='binary')) # tp / (tp + fn)
    print 'f1: ' + str(f1_score(true_list, pre_list, average='binary')) # F1 = 2 * (precision * recall) / (precision + recall)

            # kmeans_draw(K_res)