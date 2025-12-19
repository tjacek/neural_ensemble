from tabpfn import TabPFNClassifier
import base,dataset,utils

def exp(in_path):
	data=dataset.read_csv(in_path)
	clf = TabPFNClassifier()
	
print(dir(TabPFNClassifier))