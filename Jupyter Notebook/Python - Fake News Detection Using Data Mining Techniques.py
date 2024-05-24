from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import metrics
import itertools
fake_dataset = pd.read_csv(r"Fake_News.csv");
true_dataset = pd.read_csv(r"Real_News.csv");
print(fake_dataset.shape)
print(true_dataset.shape)
fake_dataset['target'] = 'fake'
true_dataset['target'] = 'true'
print(fake_dataset.head())
print(true_dataset.head())
dataset = pd.concat([fake_dataset, true_dataset]).reset_index(drop = True)
print(dataset.shape)
print(dataset.head(5))
print(dataset.tail(5))
dataset = shuffle(dataset)
dataset = dataset.reset_index(drop=True)
print(dataset.head())
print(dataset.info())
dataset.drop(["date"],axis=1,inplace=True)
print(dataset.head())
dataset.drop(["title"],axis=1,inplace=True)
print(dataset.head())
dataset['text'] = dataset['text'].apply(lambda x: x.lower())
print(dataset.head())
import string
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str
dataset['text'] = dataset['text'].apply(punctuation_removal)
print(dataset.head())
print(dataset.tail())
print(dataset.groupby(['subject'])['text'].count())
dataset.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()
print(dataset.groupby(['target'])['text'].count())
dataset.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()
def plot_confusion_matrix_from_data(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
X_train,X_test,y_train,y_test = train_test_split(dataset['text'], dataset.target, test_size=0.2, random_state=42)
print(X_test.head())
print(y_train.head())
pipeline = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])
decision_tree_model = pipeline.fit(X_train, y_train)
model_prediction = decision_tree_model.predict(X_test)
print("Decision Tree Model Accuracy: {}%".format(round(accuracy_score(y_test, model_prediction)*100,2)))
confusion_metrix = metrics.confusion_matrix(y_test, model_prediction)
plot_confusion_matrix_from_data(confusion_metrix, classes=['Fake', 'Real'])