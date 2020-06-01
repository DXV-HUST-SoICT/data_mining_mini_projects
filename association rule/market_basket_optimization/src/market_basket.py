import csv
from efficient_apriori import apriori

def read_data(file_path='../data/Market_Basket_Optimisation.csv'):
	transactions = []
	with open(file_path) as csv_file:
		for row in csv.reader(csv_file):
			transactions.append(tuple(row))
	return transactions

def extract_association_with_apriori(transactions, min_sup, min_conf):
	itemsets, rules = apriori(transactions, min_support=min_sup, min_confidence=min_conf)
	return itemsets, rules

def main():
	file_path = '../data/Market_Basket_Optimisation.csv'
	transaction_dict = read_data(file_path)
	itemsets, rules = extract_association_with_apriori(transaction_dict, min_sup=0.01, min_conf=0.3)
	# print('total learned rules = ', len(rules))
	# for i, rule in enumerate(rules):
	# 	print('Rule #{}: {}'.format(i+1, rule))
	result_file_path = '../result/leanred_rules.txt'
	with open(result_file_path, 'w+') as f:
		f.write('total learned rules = {}\n'.format(len(rules)))
		f.write('\n')
		for i, rule in enumerate(rules):
			f.write('Rule #{}: {}\n'.format(i+1, rule))

if __name__ == '__main__':
	main()