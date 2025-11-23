"""
File Modelling Tuning
Melatih model ML dengan hyperparameter tuning dan manual logging
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    log_loss, matthews_corrcoef
)
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(train_path, test_path):
    """Load data yang sudah di-preprocessing"""
    print("Loading preprocessed data...")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    print(f" Training data: {X_train.shape}")
    print(f" Testing data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_with_tuning_and_manual_logging(X_train, X_test, y_train, y_test, 
                                         model_type="RandomForest", use_dagshub=False):
    """
    Melatih model dengan hyperparameter tuning dan manual logging
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : Data
    model_type : str
        Jenis model yang akan dilatih
    use_dagshub : bool
        Apakah menggunakan DagsHub untuk remote tracking
    """
    
    # Setup MLflow tracking
    if use_dagshub:
        # Untuk Advanced level - gunakan DagsHub
        import dagshub
        dagshub.init(repo_owner='<your_username>', repo_name='<your_repo>', mlflow=True)
        print("✓ Using DagsHub for remote tracking")
    else:
        # Untuk Skilled level - gunakan lokal
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print(" Using local MLflow tracking")
    
    # Set experiment
    experiment_name = "ML_Training_with_Tuning"
    mlflow.set_experiment(experiment_name)
    
    print(f"\n{'='*60}")
    print(f"Training {model_type} with Hyperparameter Tuning")
    print(f"{'='*60}")
    
    # Define model and parameter grid
    if model_type == "RandomForest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga']
        }
    else:
        raise ValueError(f"Model type {model_type} tidak dikenali")
    
    # Grid Search dengan Cross Validation
    print("\nMelakukan Grid Search...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Start MLflow run dengan manual logging
    with mlflow.start_run(run_name=f"{model_type}_tuning_manual") as run:
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"\n Best Parameters: {best_params}")
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # MANUAL LOGGING - Metrics yang sama dengan autolog
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Training score
        train_score = best_model.score(X_train, y_train)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_score", train_score)
        
        # ADDITIONAL METRICS
        # 1. ROC AUC Score
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            mlflow.log_metric("roc_auc_score", roc_auc)
            print(f"  ROC AUC Score: {roc_auc:.4f}")
        except:
            pass
        
        # 2. Log Loss
        try:
            logloss = log_loss(y_test, y_pred_proba)
            mlflow.log_metric("log_loss", logloss)
            print(f"  Log Loss: {logloss:.4f}")
        except:
            pass
        
        # 3. Matthews Correlation Coefficient
        try:
            mcc = matthews_corrcoef(y_test, y_pred)
            mlflow.log_metric("matthews_corrcoef", mcc)
            print(f"  Matthews Correlation Coef: {mcc:.4f}")
        except:
            pass
        
        # 4. Cross Validation Score
        cv_mean_score = grid_search.best_score_
        mlflow.log_metric("cv_mean_score", cv_mean_score)
        print(f"  CV Mean Score: {cv_mean_score:.4f}")
        
        # 5. Number of Features
        n_features = X_train.shape[1]
        mlflow.log_metric("n_features", n_features)
        
        # 6. Training/Testing samples
        mlflow.log_metric("n_training_samples", len(X_train))
        mlflow.log_metric("n_testing_samples", len(X_test))
        
        # LOG PARAMETERS
        # Log best parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log model type
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("grid_search", True)
        mlflow.log_param("cv_folds", 5)
        
        # LOG MODEL
        # Log model dengan signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        mlflow.sklearn.log_model(
            best_model,
            "model",
            signature=signature
        )

        # LOG ARTIFACTS
        # 1. Best parameters JSON
        with open("best_parameters.json", "w") as f:
            json.dump(best_params, f, indent=4)
        mlflow.log_artifact("best_parameters.json")
        
        # 2. Classification report
        class_report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(class_report)
        mlflow.log_artifact("classification_report.txt")
        
        # 3. Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.savetxt("confusion_matrix.csv", cm, delimiter=",", fmt='%d')
        mlflow.log_artifact("confusion_matrix.csv")
        
        # 4. Feature importance (jika ada)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
        
        # 5. Grid search results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv("grid_search_results.csv", index=False)
        mlflow.log_artifact("grid_search_results.csv")
        
        # PRINT SUMMARY
        
        print("Training Summary")
        print(f"Model Type: {model_type}")
        print(f"Best Parameters: {best_params}")
        print(f"\nMetrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Training Score: {train_score:.4f}")
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")
        
        return best_model, best_params


def main():
    """Fungsi utama untuk menjalankan training dengan tuning"""
    
    # Configuration
    TRAIN_DATA_PATH = "dataset_preprocessing/train_data.csv"
    TEST_DATA_PATH = "dataset_preprocessing/test_data.csv"
    MODEL_TYPE = "RandomForest"  # Options: RandomForest, GradientBoosting, LogisticRegression
    
    # Untuk Advanced level, set USE_DAGSHUB = True
    # Pastikan sudah setup DagsHub dan kredensial
    USE_DAGSHUB = False  # Set True untuk Advanced level
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data(
        TRAIN_DATA_PATH,
        TEST_DATA_PATH
    )
    
    # Train dengan tuning dan manual logging
    best_model, best_params = train_with_tuning_and_manual_logging(
        X_train, X_test, y_train, y_test,
        model_type=MODEL_TYPE,
        use_dagshub=USE_DAGSHUB
    )
    
    print("\n Training completed!")
    print("\nUntuk melihat hasil training:")
    if USE_DAGSHUB:
        print("  Buka DagsHub repository Anda")
    else:
        print("  mlflow ui --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    main()