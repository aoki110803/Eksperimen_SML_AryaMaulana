"""
File Modelling - Auto Detect Classification/Regression
Melatih model ML menggunakan MLflow dengan autolog
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,  # Classification
    mean_squared_error, mean_absolute_error, r2_score  # Regression
)
import warnings
warnings.filterwarnings('ignore')


def detect_problem_type(y):
    """
    Deteksi apakah ini classification atau regression problem
    
    Parameters:
    -----------
    y : Series/array
        Target variable
        
    Returns:
    --------
    str : 'classification' atau 'regression'
    """
    unique_values = y.nunique() if hasattr(y, 'nunique') else len(np.unique(y))
    
    # Jika unique values sedikit dan semuanya integer, kemungkinan classification
    if unique_values <= 20:
        # Check jika semua nilai adalah integer
        if np.all(y == y.astype(int)):
            return 'classification'
    
    # Jika banyak unique values atau float, kemungkinan regression
    return 'regression'


def load_preprocessed_data(train_path, test_path):
    """
    Load data yang sudah di-preprocessing
    """
    print("="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Pisahkan features dan target (kolom terakhir adalah target)
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    print(f" Training data: {X_train.shape}")
    print(f" Testing data: {X_test.shape}")
    
    # Detect problem type
    problem_type = detect_problem_type(y_train)
    print(f"\n Problem Type Detected: {problem_type.upper()}")
    print(f"  - Target unique values: {y_train.nunique()}")
    print(f"  - Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    if problem_type == 'classification':
        print(f"  - Target classes: {sorted(y_train.unique())}")
    
    return X_train, X_test, y_train, y_test, problem_type


def train_classification_model(X_train, X_test, y_train, y_test):
    """
    Train classification model
    """

    print("TRAINING CLASSIFICATION MODEL")

    
    with mlflow.start_run(run_name="RandomForest_Classification") as run:
        
        # Create classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        print("\nTraining RandomForest Classifier...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        

        print("CLASSIFICATION RESULTS")

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        

        print("MLFLOW INFO")

        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")
        
        return model


def train_regression_model(X_train, X_test, y_train, y_test):
    """
    Train regression model
    """

    print("TRAINING REGRESSION MODEL")

    
    with mlflow.start_run(run_name="RandomForest_Regression") as run:
        
        # Create regressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        print("\nTraining RandomForest Regressor...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        

        print("REGRESSION RESULTS")

        print(f"RMSE:      {rmse:.4f}")
        print(f"MAE:       {mae:.4f}")
        print(f"R² Score:  {r2:.4f}")
        print(f"MSE:       {mse:.4f}")
        

        print("MLFLOW INFO")

        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")
        
        return model


def train_model_with_autolog(X_train, X_test, y_train, y_test, problem_type):
    """
    Melatih model menggunakan MLflow autolog
    Automatically choose classifier or regressor based on problem type
    """
    # Set tracking URI ke lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Set experiment
    experiment_name = "Diabetes_ML_Experiment"
    mlflow.set_experiment(experiment_name)
    

    print("MLFLOW SETUP")

    print(f" Tracking URI: http://127.0.0.1:5000")
    print(f" Experiment: {experiment_name}")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    print(f" Autolog enabled")
    
    # Train based on problem type
    if problem_type == 'classification':
        model = train_classification_model(X_train, X_test, y_train, y_test)
    else:  # regression
        model = train_regression_model(X_train, X_test, y_train, y_test)
    

    print(" TRAINING COMPLETED!")

    
    return model


def main():
    """
    Fungsi utama untuk menjalankan training
    """

    print("MLFLOW MODEL TRAINING - BASIC LEVEL")

    
    # Path ke preprocessed data
    TRAIN_DATA_PATH = "../preprocessing/dataset_preprocessing/train_data.csv"
    TEST_DATA_PATH = "../preprocessing/dataset_preprocessing/test_data.csv"
    
    try:
        # Load data dan detect problem type
        X_train, X_test, y_train, y_test, problem_type = load_preprocessed_data(
            TRAIN_DATA_PATH, 
            TEST_DATA_PATH
        )
        
        # Train model
        model = train_model_with_autolog(
            X_train, X_test, y_train, y_test,
            problem_type=problem_type
        )
        

        print("NEXT STEPS")

        print("1. Buka browser: http://127.0.0.1:5000")
        print("2. Refresh page (F5)")
        print("3. Klik experiment 'Diabetes_ML_Experiment'")
        print("4. Lihat run yang baru dibuat")
        print("5. Screenshot:")
        print("   - Dashboard (showing runs list)")
        print("   - Artifacts (showing model folder)")

        
    except FileNotFoundError as e:
        print(f"\n ERROR: File tidak ditemukan!")
        print(f"Detail: {e}")
        print("\nPastikan:")
        print("1. File train_data.csv dan test_data.csv ada")
        print("2. Path ke file sudah benar")
        print(f"3. Anda menjalankan script dari folder: membangun_model/")
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check dataset format (CSV)")
        print("2. Pastikan preprocessing sudah selesai")
        print("3. Check MLflow UI sudah running")


if __name__ == "__main__":
    # PENTING: Pastikan MLflow UI sudah running di terminal lain:
    # mlflow ui --host 127.0.0.1 --port 5000
    
    main()