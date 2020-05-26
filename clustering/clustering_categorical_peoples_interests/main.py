from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import preprocessing

np.random.seed(42)
start = time()

# X, y = load_digits(return_X_y=True)
raw_data = pd.read_csv('./data/kaggle_Interests_group.csv')
X = raw_data.drop(columns = ['group', 'grand_tot_interests'])
y = raw_data['group']
X = dict(X)
for key in X:
  for i in range(len(X[key])):
    if X[key][i] != 1:
      X[key][i] = 0
X = pd.DataFrame(X)
data = scale(X)
le = preprocessing.LabelEncoder()
fit_list = y
le.fit(fit_list)
y = le.transform(y)
labels = y

n_samples, n_features = data.shape
n_clusters = 10

print("n_clusters: %d, \t n_samples %d, \t n_features %d" % (n_clusters, n_samples, n_features))

print(82 * '_')
print('init\ttime\tinertia\tsilhouette')

def bench_k_means(estimator, name, data):
  t0 = time()
  estimator.fit(data)
  silhouette_score = metrics.silhouette_score(data, estimator.labels_, metric='euclidean')
  print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
    % (name, (time() - t0), estimator.inertia_,
      metrics.homogeneity_score(labels, estimator.labels_),
      metrics.completeness_score(labels, estimator.labels_),
      metrics.v_measure_score(labels, estimator.labels_),
      metrics.adjusted_rand_score(labels, estimator.labels_),
      metrics.adjusted_mutual_info_score(labels, estimator.labels_),
      silhouette_score))
  filename = './model/' + str(start) + '_' + name + '_n_clusters_' + str(n_clusters) + '_silhouette_score_' + str(silhouette_score) + '.sav'
  joblib.dump(estimator, filename)

pca = PCA(n_components=n_clusters).fit(data)
estimators = dict()
estimators['k-means_k-means++'] = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
estimators['k-means_random'] = KMeans(init='random', n_clusters=n_clusters, n_init=10)
estimators['k-means_PCA-based'] = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)

for name in estimators:
  # name = 'kmeans k-means++'
  estimator = estimators[name]
  bench_k_means(estimator=estimator, name=name, data=data)

print(82 * '_')

# ###########################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z,
  interpolation='nearest',
  extent=(xx.min(), xx.max(), yy.min(), yy.max()),
  cmap=plt.cm.Paired,
  aspect='auto',
  origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
  marker='x', s=169, linewidths=3,
  color='w', zorder=10)
plt.title('K-means clustering on the kaggle_Interests_group dataset (PCA-reduced data)\n'
  'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()