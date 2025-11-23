"""
File Automasi Preprocessing Data
Untuk Kriteria 1 - Skilled Level

File ini mengotomatisasi seluruh proses preprocessing yang telah
dilakukan pada notebook eksperimen.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Class untuk melakukan preprocessing data secara otomatis
    """
    
    def __init__(self, target_column='target'):
        """
        Inisialisasi DataPreprocessor
        
        Parameters:
        -----------
        target_column : str
            Nama kolom target variable
        """
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_numeric = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        self.numeric_cols = []
        self.categorical_cols = []
        
    def load_data(self, filepath):
        """
        Load data dari file CSV
        
        Parameters:
        -----------
        filepath : str
            Path ke file CSV
            
        Returns:
        --------
        df : DataFrame
            Dataset yang dimuat
        """
        print(f"Loading data dari {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✓ Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
        return df
    
    def identify_column_types(self, df):
        """
        Identifikasi tipe kolom (numerik dan kategorikal)
        
        Parameters:
        -----------
        df : DataFrame
            Dataset
        """
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Hapus target dari list jika ada
        if self.target_column in self.numeric_cols:
            self.numeric_cols.remove(self.target_column)
        if self.target_column in self.categorical_cols:
            self.categorical_cols.remove(self.target_column)
            
        print(f"✓ Kolom numerik: {len(self.numeric_cols)}")
        print(f"✓ Kolom kategorikal: {len(self.categorical_cols)}")
    
    def handle_missing_values(self, df):
        """
        Handle missing values
        
        Parameters:
        -----------
        df : DataFrame
            Dataset dengan missing values
            
        Returns:
        --------
        df : DataFrame
            Dataset tanpa missing values
        """
        print("\nHandling missing values...")
        df_clean = df.copy()
        
        # Handle numeric columns
        numeric_cols_with_missing = [col for col in self.numeric_cols if df_clean[col].isnull().any()]
        if numeric_cols_with_missing:
            df_clean[numeric_cols_with_missing] = self.imputer_numeric.fit_transform(
                df_clean[numeric_cols_with_missing]
            )
            print(f"✓ Missing values pada {len(numeric_cols_with_missing)} kolom numerik diisi dengan median")
        
        # Handle categorical columns
        categorical_cols_with_missing = [col for col in self.categorical_cols if df_clean[col].isnull().any()]
        if categorical_cols_with_missing:
            df_clean[categorical_cols_with_missing] = self.imputer_categorical.fit_transform(
                df_clean[categorical_cols_with_missing]
            )
            print(f"✓ Missing values pada {len(categorical_cols_with_missing)} kolom kategorikal diisi dengan modus")
        
        print(f"✓ Total missing values: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def remove_outliers(self, df, columns=None):
        """
        Remove outliers menggunakan IQR method
        
        Parameters:
        -----------
        df : DataFrame
            Dataset
        columns : list, optional
            List kolom yang akan dibersihkan dari outliers
            
        Returns:
        --------
        df : DataFrame
            Dataset tanpa outliers
        """
        print("\nHandling outliers...")
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        if columns is None:
            columns = self.numeric_cols
        
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_rows = initial_rows - len(df_clean)
        print(f"✓ {removed_rows} baris outliers dihapus")
        print(f"✓ Data tersisa: {len(df_clean)} baris")
        
        return df_clean
    
    def encode_categorical(self, df):
        """
        Encode categorical variables menggunakan Label Encoding
        
        Parameters:
        -----------
        df : DataFrame
            Dataset dengan kolom kategorikal
            
        Returns:
        --------
        df : DataFrame
            Dataset dengan kolom kategorikal yang sudah di-encode
        """
        print("\nEncoding categorical variables...")
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
            print(f"✓ {col} di-encode ({len(le.classes_)} kategori)")
        
        return df_encoded
    
    def scale_features(self, X):
        """
        Scale features menggunakan StandardScaler
        
        Parameters:
        -----------
        X : DataFrame
            Features yang akan di-scale
            
        Returns:
        --------
        X_scaled : DataFrame
            Features yang sudah di-scale
        """
        print("\nScaling features...")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        print(f"✓ {X_scaled.shape[1]} features di-scale")
        
        return X_scaled
    
    def split_data(self, X, y=None, test_size=0.2, random_state=42):
        """
        Split data menjadi training dan testing
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series, optional
            Target variable
        test_size : float
            Proporsi data testing
        random_state : int
            Random seed
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : DataFrames/Series
            Data yang sudah di-split
        """
        print("\nSplitting data...")
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            print(f"✓ Data training: {X_train.shape[0]} samples")
            print(f"✓ Data testing: {X_test.shape[0]} samples")
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            print(f"✓ Data training: {X_train.shape[0]} samples")
            print(f"✓ Data testing: {X_test.shape[0]} samples")
            return X_train, X_test, None, None
    
    def save_processed_data(self, X_train, X_test, y_train=None, y_test=None, output_dir='preprocessing/dataset_preprocessing'):
        """
        Save preprocessed data ke file CSV
        
        Parameters:
        -----------
        X_train, X_test : DataFrame
            Training dan testing features
        y_train, y_test : Series, optional
            Training dan testing target
        output_dir : str
            Direktori output
        """
        print(f"\nSaving preprocessed data ke {output_dir}...")
        
        # Buat direktori jika belum ada
        os.makedirs(output_dir, exist_ok=True)
        
        # Gabungkan X dan y
        if y_train is not None:
            train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
            test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
        else:
            train_data = X_train
            test_data = X_test
        
        # Save
        train_path = os.path.join(output_dir, 'train_data.csv')
        test_path = os.path.join(output_dir, 'test_data.csv')
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"✓ train_data.csv: {train_data.shape}")
        print(f"✓ test_data.csv: {test_data.shape}")
    
    def preprocess(self, filepath, remove_outliers_cols=None, output_dir='preprocessing/dataset_preprocessing'):
        """
        Pipeline lengkap preprocessing
        
        Parameters:
        -----------
        filepath : str
            Path ke raw dataset
        remove_outliers_cols : list, optional
            Kolom yang akan dibersihkan dari outliers
        output_dir : str
            Direktori output
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : DataFrames/Series
            Data yang sudah diproses
        """

        print("MEMULAI PREPROCESSING OTOMATIS")
        
        # 1. Load data
        df = self.load_data(filepath)
        
        # 2. Identify column types
        self.identify_column_types(df)
        
        # 3. Handle missing values
        df = self.handle_missing_values(df)
        
        # 4. Remove outliers
        if remove_outliers_cols:
            df = self.remove_outliers(df, remove_outliers_cols)
        
        # 5. Encode categorical
        df = self.encode_categorical(df)
        
        # 6. Separate features and target
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
        else:
            X = df
            y = None
        
        # 7. Scale features
        X_scaled = self.scale_features(X)
        
        # 8. Split data
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y)
        
        # 9. Save data
        self.save_processed_data(X_train, X_test, y_train, y_test, output_dir)
        
        print("\n" + "="*60)
        print("PREPROCESSING SELESAI!")

        
        return X_train, X_test, y_train, y_test


def main():
    """
    Fungsi utama untuk menjalankan preprocessing
    """
    # Konfigurasi
    RAW_DATA_PATH = 'dataset_raw/diabetes.csv'
    TARGET_COLUMN = 'Target'  # Ganti dengan nama kolom target Anda
    OUTPUT_DIR = 'preprocessing/dataset_preprocessing'
    
    # Kolom yang ingin dibersihkan dari outliers (opsional)
    # OUTLIER_COLS = ['col1', 'col2']  # Uncomment dan ganti dengan kolom yang sesuai
    OUTLIER_COLS = None
    
    # Inisialisasi preprocessor
    preprocessor = DataPreprocessor(target_column=TARGET_COLUMN)
    
    # Jalankan preprocessing
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        filepath=RAW_DATA_PATH,
        remove_outliers_cols=OUTLIER_COLS,
        output_dir=OUTPUT_DIR
    )
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()