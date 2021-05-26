import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from pyds import MassFunction
from sklearn.neighbors import KDTree
import math
from sklearn.datasets import make_classification
from collections import Counter
from scipy.spatial import distance


def get_centers(maj_data, min_data):
    center_maj = np.mean(maj_data)
    center_min = np.mean(min_data)
    center_meta = np.mean([center_maj, center_min], axis=0)
    return center_maj, center_min, center_meta


def get_max_mass(m):
    max_mass = 0
    lbl_max = ''
    for label in list(m.focal()):
        if m[label] > max_mass:
            max_mass = m[label]
            lbl_max = label
    return max_mass, [list(x) for x in lbl_max]


def getpl_max_from_m(mass):
    plmax = 0
    lbl_max = ''
    for m in list(mass.focal()):
        if mass.pl(m) > plmax:
            plmax = mass.pl(m)
            lbl_max = m
    # print('label max: ', lbl_max)

    return plmax, lbl_max


def getbel_max_from_m(mass):
    belmax = 0
    lbl_max = ''
    for m in list(mass.core()):
        if mass.bel(m) > belmax:
            belmax = mass.bel(m)
            lbl_max = m
    return belmax, lbl_max


def undersampling(bbas):
    counter = 0
    for bba in bbas:
        if bba['a'] < get_max_mass(bba):
            print(bba)
            plmax, lbl_max = getpl_max_from_m(bba)
            print('pl de ', lbl_max, 'est: ', plmax)
            # print('Plmax: ', getpl_max_from_m(bba))
            # print('pl(c): ', bba.pl('c'), 'pl(a): ', bba.pl('a'), 'pl(b): ', bba.pl('b'), 'pl(ab): ', bba.pl('ab'))
            counter += 1
    # print("number of removed objects: ", counter)


# def _getpl_max_from_m(self, mass):
#     plmax = 0
#     for m in list(mass.frame()):
#
#         if mass.pl(m) > plmax:
#             plmax = mass.pl(m)
#
#     return plmax


class Evus:

    def __init__(self, X, y, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t
        self.X = pd.DataFrame(data=X[0:, 0:], index=[i for i in range(X.shape[0])],
                              columns=['df' + str(i) for i in range(X.shape[1])])
        self.y = pd.DataFrame(data=y, index=[i for i in range(y.shape[0])],
                              columns=['label'])
        self.X['label'] = y
        self.lambd = self.compute_lambda()

        # splitting the data into classes
        self.maj_data, self.min_data = self.split_classes(X)
        self.maj_data = self.maj_data.drop('label', axis=1)
        self.min_data = self.min_data.drop('label', axis=1)
        self.center_maj, self.center_min, self.center_meta = get_centers(self.maj_data, self.min_data)
        self.bbas = self._evidential_editing(self.maj_data, self.t)
        # print(self.bbas)
        # print(self.min_data)
        # print("Removed bbas: ")

    def resampling(self):
        overlap_count = 0
        outlier_count = 0
        label_noise = 0
        to_remove = []
        before = len(self.maj_data)
        for index, m in enumerate(self.bbas):
            plmax, lbl_max = getpl_max_from_m(m)
            # belmax, lbl_max2 = getbel_max_from_m(m)
            print(lbl_max, ': ', plmax)
            print(type(lbl_max))
            max_mass, lbl = get_max_mass(m)
            # c1 = 0
            # c2 = 0
            # c3 = 0
            # if lbl_max == 'a':
            #     c1 += 1
            # elif lbl_max == 'b':
            #     c2 += 1
            # elif lbl_max == 'c':
            #     c3 += 1

            # print(lbl)
            # print(len(lbl))
            # print(self.maj_data.index[index])
            if len(lbl) > 1:
                # print('overlapping')
                # self.maj_data.drop(self.maj_data.index[index], inplace=True)
                to_remove.append(index)
                overlap_count += 1
            elif lbl[0][0] == 'c':
                # print('outlier')
                # self.maj_data.drop(self.maj_data.index[index], inplace=True)
                to_remove.append(index)
                outlier_count += 1
            elif lbl[0][0] == 'b':
                # print('label noise')
                # self.maj_data.drop(self.maj_data.index[index], inplace=True)
                to_remove.append(index)
                label_noise += 1
            if outlier_count + overlap_count + label_noise >= len(self.min_data):
                break

        # print("the amount of majority: ", c1)
        # print("the amount of minority: ", c2)
        # print("the amount of outliers: ", c3)
        self.maj_data.drop(self.maj_data.index[to_remove], inplace=True)
        print('before: ', before, 'after: ', len(self.maj_data))
        print('overlap: ', overlap_count, 'outliers: ', outlier_count, 'label noise: ', label_noise)
        self.maj_data['label'] = 1
        self.min_data['label'] = 0
        # print(self.min_data)
        new_df = self.maj_data.append(self.min_data, ignore_index=True)
        y_data = new_df['label'].values
        x_data = new_df.drop('label', axis=1).values
        # print(sorted(Counter(y_data).items()))
        return x_data, y_data

    def split_classes(self, data):
        classes = {}
        by_class = self.X.groupby('label')
        maj_data = by_class.get_group(1)
        min_data = by_class.get_group(0)
        return maj_data, min_data

    def compute_lambda(self):
        lambd = self.beta * (2 ** self.alpha)
        return lambd

    def _evidential_editing(self, df, t):
        bbas = []
        for index, row in df.iterrows():
            dist_maj = np.linalg.norm(row - self.center_maj)
            dist_min = np.linalg.norm(row - self.center_min)
            dist_meta = np.linalg.norm(row - self.center_meta)
            # compute gamma
            max_dist = dist_min
            min_dist = dist_maj
            if dist_maj > dist_min:
                max_dist = dist_maj
                min_dist = dist_min
            gamma = max_dist / min_dist

            mass_maj = math.exp(-dist_maj)
            mass_min = math.exp(-dist_min)
            mass_meta = math.exp(-gamma * self.lambd * dist_meta)
            mass_outlier = math.exp(-t)

            m = MassFunction()
            m['a'] = mass_maj
            m['b'] = mass_min
            m['ab'] = mass_meta
            m['c'] = mass_outlier

            # normalization
            m_sum = 0
            for label in list(m.focal()):
                m_sum += m[label]
            for label in list(m.focal()):
                m[label] = m[label] / m_sum

            bbas.append(m)
            # print(m)

        return bbas
