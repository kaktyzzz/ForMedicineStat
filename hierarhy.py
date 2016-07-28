# -*- coding: utf-8 -*-

from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

from shared import get_data


def hierarchy_draw(Z, labels, level, file_id):
    #Рисуем дендрограмму и сохраняем ее
    plt.figure()
    hierarchy.dendrogram(Z, labels=labels, color_threshold=level, leaf_font_size=5, count_sort=True)
    plt.savefig('hierarhy/dendrogram'+ str(file_id) +'.svg');
    # plt.show()


if __name__ == '__main__':

    for i in {13,19, 37, 81, 86, 100, 116, 120, 138, 161, 169, 209, 215, 224, 319, 329, 336, None}:
        names, data = get_data(4, i, False)

        dist = pdist(data, 'euclidean')
        plt.hist(dist, 500, color='green', alpha=0.5)
        plt.savefig('hierarhy/hist'+ str(i) +'.svg');
        Z = hierarchy.linkage(dist, method='average')

        hierarchy_draw(Z, names, .25, i)