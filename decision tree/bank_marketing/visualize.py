import numpy as np
import pandas as pd

num_classes = 2

err_count = np.ndarray(shape=(num_classes, num_classes), dtype=int)
err_rate = np.ndarray(shape=(num_classes, num_classes), dtype=float)

data = pd.read_csv('./log/1590442005.1459477_accuracy_0.8718039458550827_test.csv')
y_true = [int(x) for x in data['true class']]
y_pred = [int(x) for x in data['predicted class']]

num_samples = len(y_true)

for i in range(num_classes):
	for j in range(num_classes):
		err_count[i, j] = 0

for i in range(num_samples):
	err_count[y_true[i], y_pred[i]] += 1

for i in range(num_classes):
	for j in range(num_classes):
		err_rate[i, j] = err_count[i, j] / sum(err_count[i, :])

print("Error Count")
print(err_count)
print("Error Rate")
print(err_rate)