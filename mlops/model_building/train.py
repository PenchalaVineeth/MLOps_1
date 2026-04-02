import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import joblib
import os
from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

X_train_path = 'hf://datasets/vineeth32/Bank-Customer-Churn-1/X_train.csv'
X_test_path = 'hf://datasets/vineeth32/Bank-Customer-Churn-1/X_test.csv'
y_train_path = 'hf://datasets/vineeth32/Bank-Customer-Churn-1/y_train.csv'
y_test_path = 'hf://datasets/vineeth32/Bank-Customer-Churn-1/y_test.csv'

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)

numeric_features = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]

categorical_features = [
    'Geography',
]

class_weight = y_train.value_counts()[0]/y_train.value_counts()[1]
class_weight

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    model_path = 'best_churn_model_1.joblib'
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path='model')
    print(f"Model saved as artifact at: {model_path}")

    repo_id = 'vineeth32/churn-model-1'
    repo_type = 'model'

    api = HfApi(token=os.getenv('HF_TOKEN'))

    try:
      api.repo_info(repo_id=repo_id, repo_type=repo_type)
      print(f'Model Space {repo_id} already exists. Using it.')
    except RepositoryNotFoundError:
      print(f"Model Space '{repo_id}' not found. Creating new space...")
      create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
      print(f"Model Space '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj='best_churn_model_1.joblib',
        path_in_repo='best_churn_model_1.joblib',
        repo_id=repo_id,
        repo_type=repo_type,
    )
