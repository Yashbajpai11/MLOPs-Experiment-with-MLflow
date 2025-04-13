from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

#load the breast cancer dataset
data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target,name='target')

#splitting into training and testing dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#creating the randomforest model
rf = RandomForestClassifier(random_state=42)

#Defining the parameters grid for GridSearchCV
param_grid = {
    'n_estimators':[10,50,100],
    'max_depth':[None,10,20,30]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)

#run without MLflow from here
#grid_search.fit(x_train,y_train)

#display the best params and best score
#best_params = grid_search.best_params_
#best_score = grid_search.best_score_

#print(best_params)
#print(best_score)

mlflow.set_experiment('breast-cancer-rf-hp')

with mlflow.start_run():
    grid_search.fit(x_train,y_train)

    # Display the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log params
    mlflow.log_params(best_params)

    # Log metrics
    mlflow.log_metric('accuracy',best_score)

    # Log training data
    train_df = x_train.copy()
    train_df['target'] = y_train
    mlflow.log_input(mlflow.data.from_pandas(train_df), 'training')

    #log test data
    test_df = x_test.copy()
    test_df['target'] = y_test  # ← y_train ki jagah y_test hoga
    mlflow.log_input(mlflow.data.from_pandas(test_df), 'testing')

    #log score code 
    mlflow.log_artifact(__file__)

    #log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_,'random_forest')

    #set tags
    mlflow.set_tag('author', 'yash')

    print(best_params)
    print(best_score)


    