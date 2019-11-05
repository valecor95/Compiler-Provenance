import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm
from sklearn.tree import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels




def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# L = 17780
# H = 12220

print ("*** READING THE DATASET ***")

f = open("train_dataset.jsonl")
lines = f.readlines()
f.close()
dataset = set(map(lambda x: x, lines))									# scarta i doppioni REPORT
json_D = []

# Save only mnemonics
for l in dataset:
	function = json.loads(l)
	for i in range(len(function["instructions"])):
		instruction = function["instructions"][i].split(" ")[0]
		function["instructions"][i] = instruction
	json_D.append(function)
		
	# Make string for vectorizer
	function["instructions"] = ' '.join(function["instructions"])

# Make dataset
D = pd.DataFrame(json_D)
		
print ("*** VECTORIZE INSTRUCTIONS ***")		

#n_gram = (1,1)
n_gram = (1,2)
#n_gram = (1,3)
#n_gram = (2,3)
#n_gram = (3,4)															# REPORT
min_df = 5																# REPORT
max_df = 1.																# REPORT

#vectorizer = HashingVectorizer() # multivariate

#vectorizer = CountVectorizer() # multinomial
   
vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=n_gram, max_df=max_df, min_df=min_df)

X_all = vectorizer.fit_transform(D.instructions)
y_all = D.opt

print(X_all.shape)
print(y_all.shape)


print ("*** SPLIT IN TRAIN SET AND VALIDATION SET ***")
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
          test_size=0.33333, stratify = y_all)							# stratify mantains distribution REPORT

print("Train: %d - Test: %d" %(X_train.shape[0],X_test.shape[0]))


print ("*** CREATE MODEL ***")

#model = BernoulliNB()
#print('Bernoulli Model created')

model = MultinomialNB()
print('Multinomial Model created')

#model = DecisionTreeClassifier()
#print('Decision Tree Classifier Model created')

#model = svm.SVC()
#print('SVM Model created')

#model = LogisticRegression()
#print('Logistic Regression Model created')

model.fit(X_train, y_train)

print("*** EVALUATION ***")
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
class_names = np.array(["H", "L"])
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=False)

'''
print("*** PREDICT ***")

FUNCTION_new1 = np.array(['Hello, did you solve ML exercise?'])
xnew1 = vectorizer.transform(FUNCTION_new1)
ynew1 = model.predict(xnew1)
print('%s %s' %(FUNCTION_new1,ynew1))

FUNCTION_new2 = np.array(['You just won $1,000! Call now 18001234567'])
xnew2 = vectorizer.transform(FUNCTION_new2)
ynew2 = model.predict(xnew2)
print('%s %s' %(FUNCTION_new2,ynew2))
'''



