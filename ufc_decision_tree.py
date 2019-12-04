import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import ufc_utils

ufc_data = pd.read_csv('data/data.csv')

differences = ufc_utils.create_diffs(ufc_data)

x_train, x_test, y_train, y_test = train_test_split(
    differences.drop('BlueFightWin', axis=1),
    differences['BlueFightWin'],
    test_size=0.30,
    random_state=1
)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
predictions = tree.predict(x_test)

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


