"""
File Modelling 
Melatih model ML menggunakan MLflow dengan autolog
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(train_path, test_path):
    """
    Load data yang sudah di-preprocessing
    
    Parameters:
    -----------
    train_path : str
        Path ke file training data
    test_path : str
        Path ke file testing data
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    print("Loading preprocessed data...")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Pisahkan features dan target
    # Asumsi kolom terakhir adalah target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    print(f" Training data: {X_train.shape}")
    print(f" Testing data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model_with_autolog(X_train, X_test, y_train, y_test, model_name="RandomForest"):
    """
    Melatih model menggunakan MLflow autolog
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : Data
    model_name : str
        Nama model yang akan dilatih
    """
    # Set tracking URI ke lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Set experiment
    experiment_name = "ML_Training_Experiment"
    mlflow.set_experiment(experiment_name)
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} Model")
    print(f"{'='*60}")
    
    with mlflow.start_run(run_name=f"{model_name}_autolog") as run:
        
        # Pilih model
        if model_name == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_name == "LogisticRegression":
            model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Model {model_name} tidak dikenali")
        
        # Train model
        print("Training model")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics (autolog akan mencatat ini otomatis)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n Model trained successfully!")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Artifact URI: {run.info.artifact_uri}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"\nUntuk melihat hasil training, buka MLflow UI:")
    print(f"  mlflow ui --host 127.0.0.1 --port 5000")


def main():
    """
    Fungsi utama untuk menjalankan training
    """
    # Path ke preprocessed data
    TRAIN_DATA_PATH = "dataset_preprocessing/train_data.csv"
    TEST_DATA_PATH = "dataset_preprocessing/test_data.csv"
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data(
        TRAIN_DATA_PATH, 
        TEST_DATA_PATH
    )
    
    # Train model dengan autolog
    train_model_with_autolog(
        X_train, X_test, y_train, y_test,
        model_name="RandomForest"
    )


if __name__ == "__main__":
    # Jalankan MLflow UI di terminal lain dengan:
    # mlflow ui --host 127.0.0.1 --port 5000
    
    main()