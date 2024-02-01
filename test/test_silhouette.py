# write your silhouette score unit tests here
import numpy as np
from cluster import KMeans, make_clusters, Silhouette
import pytest
from sklearn.metrics import silhouette_score

#test silhouette scores range
def test_clusters():
    clusters, labels = make_clusters(k = 5, scale = 1, seed = 10)
    km = KMeans(k = 5)
    km.fit(clusters)
    predictions = km.predict(clusters)
    scores = Silhouette().score(clusters, predictions)
    assert len(np.unique(predictions)) == 5
    assert max(scores) <= 1
    assert min(scores) >= -1

