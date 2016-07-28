# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def get_data(from_x = 4, to_x = None, label_full=True):
    """Возвращает списки идентификаторов объектов и матрицу значений"""
    source = [row.strip().split(';') for row in file('pacient.csv')]
    if label_full:
        names = [ '|' + row[2].decode('UTF-8') + ' ' + row[3].decode('UTF-8') + '|' + row[0].decode('UTF-8') + '-' + row[1].decode('UTF-8') for row in source[0:]]
    else:
        names = [row[0].decode('UTF-8') for row in source[0:]]
    data = [map(float, row[from_x:to_x]) for row in source[0:]]
    # print data[:2]
    return  names, norm(data)

def norm(data):
    """Нормирование данных"""
    matrix = np.array(data, 'f')
    len_val = len(matrix[1, :])
    for i in range(len_val):
        local_min = matrix[:, i].min()
        if local_min !=  0.0:
            matrix[:, i] -= local_min
        local_max = matrix[:, i].max()
        if local_max !=  0.0:
            matrix[:, i] /= local_max
    return matrix.tolist()