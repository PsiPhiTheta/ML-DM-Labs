import numpy as np
import sklearn.feature_extraction as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from subprocess import call
import random
import pdb

def accuracy_score(x, y):
	total = 0
	for i in range(len(x)):
		if (x[i] == y[i]):
			total = total+1
	acc = total/len(x)
	return acc

def load_data():
	############ 1.1 Create the data structures ############
	data_real = []
	label_real = [] #filled with ones only (true)
	data_fake = []
	label_fake = [] #filled with zeros only (not true)

	############ 1.2 Import & label the data ############
	with open("clean_real.txt", "r") as f1:
		data_real = f1.read().splitlines()

	for x1 in range(len(data_real)):
		label_real.append("1")

	with open("clean_fake.txt", "r") as f2:
		data_fake = f2.read().splitlines()
	
	for x2 in range(len(data_fake)):
		label_fake.append("0")

	############ 1.3 Merge the imported data to a dataset ############
	dataset = list(range(len(data_real)))
	for i in range(len(data_real)):
		dataset[i] = [data_real[i], label_real[i]]

	dataset_bottom = list(range(len(data_fake)))		
	for j in range(len(data_fake)):
		dataset_bottom[j] = [data_fake[j], label_fake[j]]
		dataset.append(dataset_bottom[j])

	############ 1.4 Shuffle the dataset ############
	np_dataset = np.array(dataset) #check the shape is (3266,2) using print(np_dataset.shape)
	np.random.seed(5) #prevents tree changing every time
	np.random.shuffle(np_dataset)
	dataset = np_dataset.tolist()

	############ 1.5 Vectorize the news headlines ############
	np_headlines = np.zeros(3266) #3266 lines	
	np_headlines = np_dataset[:, 0]
	headlines = np_headlines.tolist()

	np_labels = np.zeros(3266) #3266 lines	
	np_labels = np_dataset[:, 1]
	labels = np_labels.tolist()

	vectorizer = sk.text.CountVectorizer() #Could either use feature_extraction.text.CountVectorizer([data_real]) or use TfidfVectorizer or later vectorizer.fit_transform(headlines)
	vectorizer.fit(headlines) #check contents of vectorizer using: print(vectorizer.vocabulary_)

	headlines_vector = vectorizer.transform(headlines) # encodes the vectorizer, shape: (3266, 5799), type: sparse matrix - check using: print(type(headlines_vector))

	############ 1.6 Split the vectorized dataset ############
	headlines_vector_training = headlines_vector[0:2286,:] #headlines_vector_training should be 70% which is 2286
	labels_training = labels[0:2286]
	
	headlines_vector_validation = headlines_vector[2286:2776,:] #headlines_vector_validation should be 15% which is 490
	labels_validation = labels[2286:2776]
	
	headlines_vector_test = headlines_vector[2776:3266,:]#headlines_vector_test should be 15% which is 490
	labels_test = labels[2776:3266]

	#pdb.set_trace() #used for debugging

	return (vectorizer, headlines_vector_training, labels_training, headlines_vector_validation, labels_validation)

def select_model(vectorizer, headlines_vector_training, labels_training, headlines_vector_validation, labels_validation): #cannot use the score function included in the DecisionTreeClassifier API
	############ 2.1 Build 10 models with different max_depth values for gini and entropy criterions ############
	DecisionTree_1 = DecisionTreeClassifier(criterion="gini", max_depth=2, random_state=0)
	#print(cross_val_score(DecisionTree_1, headlines_vector_training, labels_training, cv=10)) #not allowed
	DecisionTree_1.fit(headlines_vector_training, labels_training)
	predictions1 = DecisionTree_1.predict(headlines_vector_validation)
	accuracy1 = accuracy_score(predictions1, labels_validation)
	print(accuracy1)

	DecisionTree_2 = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=0)
	#print(cross_val_score(DecisionTree_2, headlines_vector_training, labels_training, cv=10)) #not allowed
	DecisionTree_2.fit(headlines_vector_training, labels_training)
	predictions2 = DecisionTree_2.predict(headlines_vector_validation)
	accuracy2 = accuracy_score(predictions2, labels_validation)
	print(accuracy2)

	DecisionTree_3 = DecisionTreeClassifier(criterion="gini", max_depth=50, random_state=0)
	#print(cross_val_score(DecisionTree_3, headlines_vector_training, labels_training, cv=10)) #not allowed
	DecisionTree_3.fit(headlines_vector_training, labels_training)
	predictions3 = DecisionTree_3.predict(headlines_vector_validation)
	accuracy3 = accuracy_score(predictions3, labels_validation)
	print(accuracy3)

	DecisionTree_4 = DecisionTreeClassifier(criterion="gini", max_depth=200, random_state=0)
	#print(cross_val_score(DecisionTree_4, headlines_vector_training, labels_training, cv=10)) #not allowed
	DecisionTree_4.fit(headlines_vector_training, labels_training)
	predictions4 = DecisionTree_4.predict(headlines_vector_validation)
	accuracy4 = accuracy_score(predictions4, labels_validation)
	print(accuracy4)

	DecisionTree_5 = DecisionTreeClassifier(criterion="gini", max_depth=1000, random_state=0)
	#print(cross_val_score(DecisionTree_5, headlines_vector_training, labels_training, cv=10)) #not allowed
	DecisionTree_5.fit(headlines_vector_training, labels_training)
	predictions5 = DecisionTree_5.predict(headlines_vector_validation)
	accuracy5 = accuracy_score(predictions5, labels_validation)
	print(accuracy5)

	DecisionTree_6 = DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=0)
	#print(cross_val_score(DecisionTree_6, headlines_vector_training, labels_training, cv=10)) #not allowed
	DecisionTree_6.fit(headlines_vector_training, labels_training)
	predictions6 = DecisionTree_6.predict(headlines_vector_validation)
	accuracy6 = accuracy_score(predictions6, labels_validation)
	print(accuracy6)

	DecisionTree_7 = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=0)
	#print(cross_val_score(DecisionTree_7, headlines_vector_training, labels_training, cv=10)) #not allowed
	DecisionTree_7.fit(headlines_vector_training, labels_training)
	predictions7 = DecisionTree_7.predict(headlines_vector_validation)
	accuracy7 = accuracy_score(predictions7, labels_validation)
	print(accuracy7)

	DecisionTree_8 = DecisionTreeClassifier(criterion="entropy", max_depth=50, random_state=0)
	#print(cross_val_score(DecisionTree_8, headlines_vector_training, labels_training, cv=10)) #not allowed
	DecisionTree_8.fit(headlines_vector_training, labels_training)
	predictions8 = DecisionTree_8.predict(headlines_vector_validation)
	accuracy8 = accuracy_score(predictions8, labels_validation)
	print(accuracy8)

	DecisionTree_9 = DecisionTreeClassifier(criterion="entropy", max_depth=200, random_state=0)
	#print(cross_val_score(DecisionTree_9, headlines_vector_training, labels_training, cv=10)) #not allowed
	DecisionTree_9.fit(headlines_vector_training, labels_training)
	predictions9 = DecisionTree_9.predict(headlines_vector_validation)
	accuracy9 = accuracy_score(predictions9, labels_validation)
	print(accuracy9)

	DecisionTree_10 = DecisionTreeClassifier(criterion="entropy", max_depth=1000, random_state=0)
	#print(cross_val_score(DecisionTree_10, headlines_vector_training, labels_training, cv=10)) #not allowed
	DecisionTree_10.fit(headlines_vector_training, labels_training)
	predictions10 = DecisionTree_10.predict(headlines_vector_validation)
	accuracy10 = accuracy_score(predictions10, labels_validation)
	print(accuracy10)

	############ 2.2 Print the maximum performance and number of tree that yielded that performance ############
	print("Maximum is:")
	accuracy = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10]
	print(max(accuracy))
	print(accuracy.index(max(accuracy))+1)

	############ 2.3 Extract and print the optimum tree ############
	#Run the following in the environment to extract the correct tree, run using python3 -i filename.py --> gives you into the python environment
	vocab = list(vectorizer.get_feature_names())
	export_graphviz(DecisionTree_9, max_depth=2, out_file='best_tree.dot', class_names=['fake', 'real'], feature_names=vocab, rounded=True, filled=True)
	#call('dot -Tpng best_tree.dot -o best_tree.png', shell=True)

def compute_information_gain(word, vectorizer, headlines_vector_training, labels_training):
	############ 3.1 Sum up the totals ############
	np_headlines_vector_training = headlines_vector_training.toarray()
	idx = vectorizer.vocabulary_.get(word)
	word_present = (np_headlines_vector_training[:, idx] == 1)
	int_labels_training = np.array(list(map(int, labels_training)))

	print(sum(word_present))

	real_word = sum(int_labels_training[word_present])+1 #for donald 606
	print(real_word)
	real_notword = sum(int_labels_training[np.logical_not(word_present)])-1 #for donald 782
	print(real_notword)
	fake_word = sum(1-int_labels_training[word_present])  #for donald 160
	print(fake_word)
	fake_notword = sum(1-int_labels_training[np.logical_not(word_present)]) #for donald 738 
	print(fake_notword)

	real_total = real_word + real_notword
	fake_total = fake_word + fake_notword
	word_total = real_word + fake_word
	notword_total = real_notword + fake_notword
	total = real_word + real_notword + fake_word + fake_notword

	############ 3.2 Calculate the entropies ############
	entrop_root = -(fake_total/total)*np.log2(fake_total/total) - (real_total/total)*np.log2(real_total/total)
	print(entrop_root) #for donald 0.9654168827384615
	entrop_given_word = -(real_word/word_total)*np.log2(real_word/word_total) - (fake_word/word_total)*np.log2(fake_word/word_total)
	print(entrop_given_word) #for donald 0.7373396972207713
	entrop_given_notword = -(real_notword/notword_total)*np.log2(real_notword/notword_total) - (fake_notword/notword_total)*np.log2(fake_notword/notword_total)
	print(entrop_given_notword) #for donald 0.9992131140819875
	entrop_given_either = (word_total/total)*entrop_given_word + (notword_total/total)*entrop_given_notword
	print(entrop_given_either) #for donald 0.9113091369434047

	############ 3.3 Calculate the information gain ############
	information_gain = entrop_root - entrop_given_either #compare with mutual_info_classif from scikit

	############ 3.4 Print the information gain ############
	print("Information gain is:")
	print(information_gain) #for donald 0.05410774579505684 - check with (word_total/total)*entrop_given_word + (notword_total/total)*entrop_given_notword
	print("--------------------")

if __name__== "__main__":
	print("############ START ############")

	print("############ 1/3 Loading Data ############")
	vectorizer, headlines_vector_training, labels_training, headlines_vector_validation, labels_validation = load_data()

	print("############ 2/3 Selecting Model ############")
	select_model(vectorizer, headlines_vector_training, labels_training, headlines_vector_validation, labels_validation)
	
	print("############ 3/3 Computing Information Gain ############")
	compute_information_gain("donald", vectorizer, headlines_vector_training, labels_training)
	compute_information_gain("the", vectorizer, headlines_vector_training, labels_training)
	compute_information_gain("trumps", vectorizer, headlines_vector_training, labels_training)
	compute_information_gain("hillary", vectorizer, headlines_vector_training, labels_training)
	compute_information_gain("coal", vectorizer, headlines_vector_training, labels_training)
	compute_information_gain("trump", vectorizer, headlines_vector_training, labels_training)	

	print("############ END ############")
