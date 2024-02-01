# Write your k-means unit tests here
import numpy as np
from cluster import KMeans, make_clusters
import pytest

# test to see if the correct number of clusters is returned, using a stable seed
def test_clusters():
    clusters, labels = make_clusters(k = 5, scale = 1, seed = 10)
    km = KMeans(k = 5)
    km.fit(clusters)
    predictions = km.predict(clusters)
    assert len(np.unique(predictions)) == 5

#test exception raise for inappropriate k
def test_assert_kmeans():
    with pytest.raises(Exception) as Error:
        km = KMeans(k = 0)
    
    assert "k must be positive" in str(Error.value)

#test exception raise for non integer k
def test_assert_kmeans2():
    with pytest.raises(Exception) as Error:
        km = KMeans(k = "0")

    assert "k must be an integer" in str(Error.value)