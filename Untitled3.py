
# coding: utf-8

# In[1]:


-pip install --user matplotlib pandas numpy scipy sklearn


# In[2]:


get_ipython().system('pip install --user matplotlib pandas numpy scipy sklearn')


# In[3]:


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import requests 
from io import BytesIO

iris_url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
resp = requests.get(iris_url)
print(resp.content)

data=BytesIO(resp.content)


# In[6]:


import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

df=pd.read_csv(data)
print(df.head())


# In[7]:


print(df.head()['sepal_length'])


# In[8]:


print(df.iloc[0])


# In[11]:


target = df[df.columns[-1]]
target = target.astype('category')
numeric_data=df._get_numeric_data()
print(target.head())
print(" ")
print(numeric_data.head())


# In[14]:


#print(target.cat.codes)
training_data,testing_data,training_label,testing_label=train_test_split(numeric_data,target.cat.codes)

print('training_data')
print(training_data.head())

print(len(training_data))
print(len(testing_data))


# In[15]:


tree_model=tree.DecisionTreeClassifier()
tree_model.fit(training_data,training_label)

print(tree_model)


# In[17]:


predict_result=tree_model.predict(testing_data)
score_result=tree_model.predict_proba(testing_data)

print(predict_result[0:5])
print(score_result[0:5])


# In[19]:


matrix=confusion_matrix(testing_label,predict_result)
report=classification_report(testing_label,predict_result,target_names=target.cat.categories)
acc=accuracy_score(testing_label,predict_result)

print(matrix)
print("===")
print(report)
print("===")
print(acc)


# In[20]:


print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

