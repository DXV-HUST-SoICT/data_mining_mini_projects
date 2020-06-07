import numpy as np
import pandas as pd

num_classes = 5

err_count = np.ndarray(shape=(num_classes, num_classes), dtype=int)
err_rate = np.ndarray(shape=(num_classes, num_classes), dtype=float)

data = pd.read_csv('./log/test_time_1590997089.5220308_accuracy_0.9730700179533214.csv')
y_true = [int(x) for x in data['category']]
y_pred = [int(x) for x in data['prediction']]

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