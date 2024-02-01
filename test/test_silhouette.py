# write your silhouette score unit tests here
import numpy as np
from cluster import KMeans, make_clusters, Silhouette
import pytest
from sklearn.metrics import silhouette_score

#test silhouette scores range
def test_clusters():
    clusters, labels = make_clusters(k = 9, scale = 1, seed = 13)
    km = KMeans(k = 9)
    km.fit(clusters)
    predictions = km.predict(clusters)
    scores = Silhouette().score(clusters, predictions)
    # check number of clusters is correct, expect 9
    assert len(np.unique(predictions)) == 9
    # check if the scores are in an appropriate range between -1 and 1, also checked below 
    assert max(scores) <= 1
    assert min(scores) >= -1

#test sklearn silhouette scores
def test_score():
    clusters, labels = make_clusters(k = 6, scale = 1, seed = 12)
    km = KMeans(k = 5)
    km.fit(clusters)
    predictions = km.predict(clusters)
    scores = Silhouette().score(clusters, predictions)
    # average score to compare to sklearn score average
    averaged_score = float(sum(scores) / len(scores))
    sklearn_score = silhouette_score(clusters, np.ravel(predictions))
    #with a tolerance of 10%, are the scores close
    assert np.isclose(averaged_score, sklearn_score, rtol = 0.1)