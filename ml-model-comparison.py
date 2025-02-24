import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load the dataset
data = pd.read_csv('example.csv')
X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rfc = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rfc = grid_search.best_estimator_

y_pred = best_rfc.predict(X_test)
print(classification_report(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred, average='weighted'))


svc = SVC(random_state=42)
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_svc = grid_search.best_estimator_

mlp = MLPClassifier(random_state=42)
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_mlp = grid_search.best_estimator_


print('Random Forest Classifier:')
print(classification_report(y_test, best_rfc.predict(X_test)))
print('F1 score:', f1_score(y_test, best_rfc.predict(X_test), average='weighted'))

print('Support Vector Machine:')
print(classification_report(y_test, best_svc.predict(X_test)))
print('F1 score:', f1_score(y_test, best_svc.predict(X_test), average='weighted'))

print('Multi-Layer Perceptron:')
print(classification_report(y_test, best_mlp.predict(X_test)))
print('F1 score:', f1_score(y_test, best_mlp.predict(X_test), average='weighted'))
