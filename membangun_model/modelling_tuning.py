"""
File Modelling Tuning 
Melatih model ML dengan hyperparameter tuning dan manual logging
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, max_error
)
import json
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(train_path, test_path):
    """Load data yang sudah di-preprocessing"""
    print("="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Testing data: {X_test.shape}")
    print(f"✓ Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    return X_train, X_test, y_train, y_test


def train_with_tuning_and_manual_logging(X_train, X_test, y_train, y_test, 
                                         model_type="RandomForest", use_dagshub=False):
    """
    Melatih model dengan hyperparameter tuning dan manual logging
    """
    
    # Setup MLflow tracking
    if use_dagshub:

        print("SETTING UP DAGSHUB")

        
        try:
            import dagshub
            dagshub.init(
                repo_owner='aryamaulana110803',
                repo_name='Eksperimen_SML_AryaMaulana', 
                mlflow=True
            )
            print(" DagsHub initialized successfully")
            print(" Using remote tracking (DagsHub)")
        except Exception as e:
            print(f" Error initializing DagsHub: {e}")
            print("Falling back to local tracking...")
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print(" Using local MLflow tracking")
    
    # Set experiment
    experiment_name = "Diabetes_Regression_Tuning"
    mlflow.set_experiment(experiment_name)
    
    print(f"\n{'='*60}")
    print(f"TRAINING {model_type.upper()} WITH HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    
    # Define model and parameter grid (REGRESSION MODELS)
    if model_type == "RandomForest":
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif model_type == "Ridge":
        model = Ridge(random_state=42)
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }
    else:
        raise ValueError(f"Model type {model_type} tidak dikenali")
    
    # Grid Search dengan Cross Validation
    print("\nPerforming Grid Search with 5-Fold Cross Validation...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',  # For regression
        n_jobs=-1,
        verbose=1
    )
    
    # Start MLflow run dengan manual logging
    with mlflow.start_run(run_name=f"{model_type}_Tuning_Manual") as run:
        
        print("\nTraining model...")
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"\n Best Parameters: {best_params}")
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_train_pred = best_model.predict(X_train)
        
        # ==================================================
        # MANUAL LOGGING - REGRESSION METRICS
        # ==================================================
        
        # Basic regression metrics (sama dengan autolog)
        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Log basic metrics (equivalent to autolog)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2_score", test_r2)
        
        mlflow.log_metric("training_mse", train_mse)
        mlflow.log_metric("training_rmse", train_rmse)
        mlflow.log_metric("training_mae", train_mae)
        mlflow.log_metric("training_r2_score", train_r2)
        
        # ==================================================
        # ADDITIONAL METRICS (ADVANCED LEVEL - 2+ extra)
        # ==================================================
        
        # 1. MAPE (Mean Absolute Percentage Error)
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mlflow.log_metric("test_mape", mape)
            print(f"  MAPE: {mape:.4f}")
        except:
            pass
        
        # 2. Max Error
        try:
            max_err = max_error(y_test, y_pred)
            mlflow.log_metric("test_max_error", max_err)
            print(f"  Max Error: {max_err:.4f}")
        except:
            pass
        
        # 3. Explained Variance Score
        from sklearn.metrics import explained_variance_score
        evs = explained_variance_score(y_test, y_pred)
        mlflow.log_metric("explained_variance_score", evs)
        print(f"  Explained Variance: {evs:.4f}")
        
        # 4. Cross Validation Score
        cv_mean_score = grid_search.best_score_
        mlflow.log_metric("cv_mean_r2_score", cv_mean_score)
        print(f"  CV Mean R²: {cv_mean_score:.4f}")
        
        # 5. Residual Statistics
        residuals = y_test - y_pred
        mlflow.log_metric("residual_mean", np.mean(residuals))
        mlflow.log_metric("residual_std", np.std(residuals))
        
        # 6. Number of Features
        mlflow.log_metric("n_features", X_train.shape[1])
        
        # 7. Training/Testing samples
        mlflow.log_metric("n_training_samples", len(X_train))
        mlflow.log_metric("n_testing_samples", len(X_test))
        
        # ==================================================
        # LOG PARAMETERS
        # ==================================================
        
        # Log best parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log model configuration
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("grid_search", True)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring_metric", "r2")
        
        # ==================================================
        # LOG MODEL
        # ==================================================
        
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        mlflow.sklearn.log_model(
            best_model,
            "model",
            signature=signature,
            registered_model_name=f"{model_type}_Diabetes_Regression"
        )
        
        # ==================================================
        # LOG ARTIFACTS
        # ==================================================
        
        # 1. Best parameters JSON
        with open("best_parameters.json", "w") as f:
            json.dump(best_params, f, indent=4)
        mlflow.log_artifact("best_parameters.json")
        
        # 2. Metrics summary
        metrics_summary = {
            "test_metrics": {
                "rmse": float(test_rmse),
                "mae": float(test_mae),
                "r2_score": float(test_r2),
                "mse": float(test_mse)
            },
            "training_metrics": {
                "rmse": float(train_rmse),
                "mae": float(train_mae),
                "r2_score": float(train_r2)
            }
        }
        with open("metrics_summary.json", "w") as f:
            json.dump(metrics_summary, f, indent=4)
        mlflow.log_artifact("metrics_summary.json")
        
        # 3. Predictions vs Actual
        predictions_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred,
            'residual': residuals
        })
        predictions_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")
        
        # 4. Feature importance (if available)
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
        
        # 6. Model parameters summary
        model_params = {
            "model_type": model_type,
            "best_params": best_params,
            "n_features": X_train.shape[1],
            "n_training_samples": len(X_train),
            "n_testing_samples": len(X_test)
        }
        with open("model_config.json", "w") as f:
            json.dump(model_params, f, indent=4)
        mlflow.log_artifact("model_config.json")
        
        # ==================================================
        # PRINT SUMMARY
        # ==================================================
        

        print("TRAINING SUMMARY")

        print(f"Model Type: {model_type}")
        print(f"Best Parameters: {best_params}")
        print(f"\nTest Metrics:")
        print(f"  RMSE:     {test_rmse:.4f}")
        print(f"  MAE:      {test_mae:.4f}")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"\nTraining Metrics:")
        print(f"  RMSE:     {train_rmse:.4f}")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"\nMLflow Info:")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Artifact URI: {run.info.artifact_uri}")
        print("="*60)
        
        return best_model, best_params


def main():
    """Fungsi utama untuk menjalankan training dengan tuning"""
    
    print("\n" + "="*60)
    print("MLFLOW TRAINING - ADVANCED LEVEL (DAGSHUB)")
    print("="*60)
    
    # Configuration
    TRAIN_DATA_PATH = "../preprocessing/dataset_preprocessing/train_data.csv"
    TEST_DATA_PATH = "../preprocessing/dataset_preprocessing/test_data.csv"
    MODEL_TYPE = "RandomForest"  # Options: RandomForest, GradientBoosting, Ridge
    
    # ADVANCED LEVEL: Use DagsHub
    USE_DAGSHUB = True
    
    try:
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
        
        print("\n" + "="*60)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if USE_DAGSHUB:
            print("\nNext Steps:")
            print("1. Buka DagsHub repository Anda:")
            print("   https://dagshub.com/aryamaulana110803/Eksperimen_SML_AryaMaulana")
            print("2. Klik tab 'Experiments'")
            print("3. Lihat experiment 'Diabetes_Regression_Tuning'")
            print("4. Screenshot dashboard dan artifacts dari DagsHub")
            print("\n5. Buat file DagsHub.txt berisi link DagsHub Anda")
        else:
            print("\nBuka MLflow UI: http://127.0.0.1:5000")
        
        print("="*60)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Pastikan dataset ada dan format benar")
        print("2. Check DagsHub credentials (dagshub login)")
        print("3. Check internet connection untuk DagsHub")


if __name__ == "__main__":
    main()