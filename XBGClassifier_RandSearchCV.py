from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

# Create a custom scorer
f1_scorer = make_scorer(f1_score)

# Define the parameter grid to search
param_grid = {
    'model__learning_rate': [0.001, 0.01, 0.05],
    'model__max_depth': [9, 11, 13, 17],
    'model__n_estimators': [420, 500, 550, 600],
    'model__subsample': [0.65, 0.75, 0.85, 0.9],
    'model__colsample_bytree': [0.65, 0.75, 0.85, 0.9],
}

# Create the GridSearchCV object
grid_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, 
                           cv=3, n_jobs=-1, scoring=f1_scorer, verbose=2, n_iter = 30)

# Fit the grid search to the data
grid_search.fit(X_train, y_train, model__sample_weight=sample_weights)

# Get the best estimator
best_clf = grid_search.best_estimator_

# Preprocessing of validation data, get predictions
best_predictions = best_clf.predict(X_test)

print(classification_report(y_test, best_predictions))
