# BSAS
import pyclustering

from pyclustering.core.wrapper import ccore_library
from pyclustering.core.bsas_wrapper import bsas as bsas_wrapper
from pyclustering.core.metric_wrapper import metric_wrapper

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.encoder import type_encoding

from pyclustering.utils.metric import type_metric, distance_metric


from sklearn.datasets import load_iris
from numpy import *


class bsas_visualizer: # Gera o gráfico

    @staticmethod
    def show_clusters(sample, clusters, representatives, **kwargs):

        figure = kwargs.get('figure', None)
        display = kwargs.get('display', True)
        offset = kwargs.get('offset', 0)

        visualizer = cluster_visualizer()
        visualizer.append_clusters(clusters, sample, canvas=offset)

        for cluster_index in range(len(clusters)):
            visualizer.append_cluster_attribute(offset, cluster_index, [representatives[cluster_index]], '*', 10) # visualizer.append_cluster_attribute(offset, cluster_index, [representatives[cluster_index]], '*', 10)

        return visualizer.show(figure=figure, display=display)


class bsas:

    def __init__(self, data, maximum_clusters, threshold, ccore=True, **kwargs):

        self._data = data
        self._amount = maximum_clusters
        self._threshold = threshold
        self._metric = kwargs.get('metric', distance_metric(type_metric.EUCLIDEAN))
        self._ccore = ccore and self._metric.get_type() != type_metric.USER_DEFINED

        self._clusters = []
        self._representatives = []

        if self._ccore is True:
            self._ccore = ccore_library.workable()

    def process(self):

        if self._ccore is True:
            self.__process_by_ccore()
        else:
            self.__prcess_by_python()

    def __process_by_ccore(self):
        ccore_metric = metric_wrapper.create_instance(self._metric)
        self._clusters, self._representatives = bsas_wrapper(self._data, self._amount, self._threshold,
                                                             ccore_metric.get_pointer())

    def __prcess_by_python(self):
        self._clusters.append([0])
        self._representatives.append(self._data[0])

        for i in range(1, len(self._data)):
            point = self._data[i]
            index_cluster, distance = self._find_nearest_cluster(point)

            if (distance > self._threshold) and (len(self._clusters) < self._amount):
                self._representatives.append(point)
                self._clusters.append([i])
            else:
                self._clusters[index_cluster].append(i)
                self._update_representative(index_cluster, point)

    def get_clusters(self):

        return self._clusters

    def get_representatives(self):

        return self._representatives

    def get_cluster_encoding(self):

        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION

    def _find_nearest_cluster(self, point):  # Cálculo do cluster mais próximo

        index_cluster = -1
        nearest_distance = float('inf')

        for index in range(len(self._representatives)):
            distance = self._metric(point, self._representatives[index])
            if distance < nearest_distance:
                index_cluster = index
                nearest_distance = distance

        return index_cluster, nearest_distance

    def _update_representative(self, index_cluster, point):  # Cálculo dos novos centróides

        length = len(self._clusters[index_cluster])
        rep = self._representatives[index_cluster]

        for dimension in range(len(rep)):
            rep[dimension] = ((length - 1) * rep[dimension] + point[dimension]) / length


if __name__ == "__main__":

    iris = load_iris()
    aux = iris.data

    sample = []

    for i in range(len(aux)): # Cálculo da área da pétala
        area = aux[i][2] * aux[i][3]
        s = aux[i][:-2]
        s = insert(s, 2, area)
        sample.append(s)

    max_clusters = 3
    threshold = 4.0  # limiar de dissimilaridade

    bsas_instance = bsas(sample, max_clusters, threshold)
    bsas_instance.process()

    clusters = bsas_instance.get_clusters()
    representatives = bsas_instance.get_representatives()

    print('\nCentróides: ', representatives)  # Últimos representantes de cada cluster

    print('\n\nSetosa: ',len(clusters[0]) ,'     ',clusters[0])
    print('\nVirginica: ',len(clusters[1]) ,'  ',clusters[1])
    print('\nVersicolor: ',len(clusters[2]) ,' ',clusters[2])

    cont0 = 0
    cont1 = 0
    cont2 = 0
    for i in range(len(clusters)):  # Cálculo dos acertos
        for j in range(len(clusters[i])):

            if i == 0 and clusters[i][j] >= 0 and clusters[i][j] <= 49:
                cont0 += 1
            elif i == 1 and clusters[i][j] >= 50 and clusters[i][j] <= 99:
                cont1 += 1
            elif i == 2 and clusters[i][j] >= 100 and clusters[i][j] <= 149:
                cont2 += 1

    print("\n\nAcertos Setosa:", cont0)
    print("Acertos Virginica:", cont1)
    print("Acertos Versicolor:", cont2)
    print("\nTotal acertos:", cont0 + cont1 + cont2)

    bsas_visualizer.show_clusters(sample, clusters, representatives)







'''Atributos:
   1. comprimento da sépala em cm
   2. largura da sépala em cm
   3. área da pétala em cm'''


