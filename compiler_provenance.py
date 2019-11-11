import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
import preprocessing as pr

print ("*** READING THE DATASET ***")
json_D = pr.preprocess("train_dataset.jsonl")

D = pd.DataFrame(json_D)                                                        # Make dataset

print("************************************** OPTIMIZATION DETECTION **************************************")

print ("*** VECTORIZE INSTRUCTIONS ***")

#n_gram = (1,1)
n_gram = (1,3)

vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=n_gram, max_features=10000)

X_all = vectorizer.fit_transform(D.instructions)
y_all = D.opt

print(X_all.shape)
print(y_all.shape)


print ("*** SPLIT IN TRAIN SET AND VALIDATION SET ***")
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33333, stratify = y_all)		# stratify mantains distribution

print("Train: %d - Test: %d" %(X_train.shape[0],X_test.shape[0]))

print ("*** CREATE MODEL ***")

#opt_model = BernoulliNB()
#print('Bernoulli Model created')

#opt_model = MultinomialNB()
#print('Multinomial Model created')

opt_model = svm.LinearSVC(C=1)
print('LinearSVC model created')


opt_model.fit(X_train, y_train)

print("*** EVALUATION ***")
y_pred = opt_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("*** CROSS VALIDATION ***")
cv = ShuffleSplit(n_splits=5, test_size=0.333, random_state=120)
scores = cross_val_score(opt_model, X_all, y_all, cv=cv)
print(scores)
print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print("*** PREDICT ***")
#{"instructions": ["xor edx cmp rdi rsi mov rax seta rdx cmovae rax rdx ret" "opt": "H", "compiler": "gcc"}
FUNCTION_new1 = np.array(["xor edx cmp rdi rsi mov rax seta rdx cmovae rax rdx ret"])
xnew1 = vectorizer.transform(FUNCTION_new1)
ynew1 = opt_model.predict(xnew1)
print('%s %s,   right = H' %(FUNCTION_new1,ynew1))


print("************************************** COMPILER DETECTION **************************************")
print ("*** VECTORIZE INSTRUCTIONS ***")

#n_gram = (1,1)
n_gram = (1,3)

vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=n_gram, max_features = 10000)

X_all = vectorizer.fit_transform(D.instructions)
y_all = D.compiler

print(X_all.shape)
print(y_all.shape)

print ("*** SPLIT IN TRAIN SET AND VALIDATION SET ***")
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33333, stratify = y_all)      # stratify mantains distribution

print ("*** CREATE MODEL ***")

#compiler_model = BernoulliNB()
#print('Bernoulli Model created')

#compiler_model = MultinomialNB()
#print('Multinomial Model created')

compiler_model = svm.LinearSVC(C=1)
print("LinearSVC model created")

compiler_model.fit(X_train, y_train)

print("*** EVALUATION ***")
y_pred = compiler_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("*** CROSS VALIDATION ***")
cv = ShuffleSplit(n_splits=5, test_size=0.333, random_state=120)
scores = cross_val_score(compiler_model, X_all, y_all, cv=cv)
print(scores)
print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("*** PREDICT ***")
#{"instructions": ["xor edx cmp rdi rsi mov rax seta rdx cmovae rax rdx ret" "opt": "H", "compiler": "gcc"}
FUNCTION_new1 = np.array(["xor edx cmp rdi rsi mov rax seta rdx cmovae rax rdx ret"])
xnew1 = vectorizer.transform(FUNCTION_new1)
ynew1 = compiler_model.predict(xnew1)
print('%s %s,   right = gcc' %(FUNCTION_new1,ynew1))

'''
print("*** BLIND SET PREDICTION ***")
res = open("res.txt", "w")
json_D = pr.preprocess("test_dataset_blind.jsonl")

for function in json_D:
	FUNCTION_new = np.array([function["instructions"]])
	xnew = vectorizer.transform(FUNCTION_new)
	y_opt = opt_model.predict(xnew)
	y_compiler = compiler_model.predict(xnew)
	res.write('%s, %s\n' %(y_compiler[0], y_opt[0]))
res.close()
print("*** END ***")
'''
