import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('fetal_health.csv')
df.drop_duplicates(inplace=True)
from sklearn.model_selection import train_test_split as tts
xtrain, xtest, ytrain, ytest = tts(df.iloc[:,:-1], df.iloc[:,-1:], random_state=42)
from sklearn.tree import DecisionTreeClassifier as dt
model = dt()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import classification_report, confusion_matrix
clsreport = classification_report(ytest, ypred)
print(clsreport)
cm = confusion_matrix(ytest, ypred)

plt.figure(figsize=(3, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()
from sklearn import tree
plt.figure(figsize=(50,20))
tree.plot_tree(model,filled=True, fontsize=10)
plt.show()