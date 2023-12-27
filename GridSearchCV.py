from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'model__learning_rate': [0.01, 0.1, 0.3],
    'model__max_depth': [3, 5, 7],
    'model__n_estimators': [50, 100, 200]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, scoring='accuracy', verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train, model__sample_weight=sample_weights)

# Get the best estimator
best_clf = grid_search.best_estimator_

# Preprocessing of validation data, get predictions
best_predictions = best_clf.predict(X_test)

print(classification_report(y_test, best_predictions))
