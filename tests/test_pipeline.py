import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data import load_and_prepare_data, get_season, get_aqi_category
from feature_engineering import create_advanced_features
from config_loader import Config

class TestDataLoading:
    """Test data loading and preprocessing functions"""
    
    def setup_method(self):
        """Setup test data"""
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='H'),
            'Time': ['10:30:00'] * 100,
            'CO (mg/m3)': np.random.uniform(0.1, 5.0, 100),
            'SO2(µg/m³)': np.random.uniform(5, 50, 100),
            'NO2(µg/m³)': np.random.uniform(10, 100, 100),
            'PM2.5(µg/m³)': np.random.uniform(5, 80, 100),
            'PM10(µg/m³)': np.random.uniform(10, 150, 100),
            'Temp(°C)': np.random.uniform(-10, 40, 100),
            'RH (%)': np.random.uniform(20, 90, 100),
            'Location': ['TestLocation'] * 100
        })
    
    def test_get_season(self):
        """Test season assignment"""
        assert get_season(1) == 'Winter'
        assert get_season(4) == 'Spring'
        assert get_season(7) == 'Summer'
        assert get_season(10) == 'Autumn'
        assert get_season(12) == 'Winter'
    
    def test_get_aqi_category(self):
        """Test AQI category assignment"""
        assert get_aqi_category(10) == 'Good'
        assert get_aqi_category(25) == 'Moderate'
        assert get_aqi_category(45) == 'Unhealthy for Sensitive Groups'
        assert get_aqi_category(100) == 'Unhealthy'
        assert get_aqi_category(200) == 'Very Unhealthy'
        assert get_aqi_category(300) == 'Hazardous'
        assert get_aqi_category(np.nan) == 'Unknown'
    
    def test_feature_engineering(self):
        """Test feature engineering functions"""
        enhanced_data = create_advanced_features(self.sample_data)
        
        # Check that new features are created
        assert 'CO_NO2_ratio' in enhanced_data.columns
        assert 'PM25_PM10_ratio' in enhanced_data.columns
        assert 'DayOfWeek' in enhanced_data.columns
        assert 'Month' in enhanced_data.columns
        assert 'IsWeekend' in enhanced_data.columns
        
        # Check that original data is preserved
        assert enhanced_data.shape[0] == self.sample_data.shape[0]
        assert enhanced_data.shape[1] > self.sample_data.shape[1]
    
    def test_data_types(self):
        """Test data type consistency"""
        # Test that numeric columns are numeric
        numeric_cols = ['CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 'PM2.5(µg/m³)']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(self.sample_data[col])
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        # Add some missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0:5, 'PM2.5(µg/m³)'] = np.nan
        
        enhanced_data = create_advanced_features(data_with_missing)
        
        # Check that missing values are handled appropriately
        assert not enhanced_data.empty

class TestConfiguration:
    """Test configuration loading and management"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        # Test with default config
        config = Config()
        
        # Check that default values exist
        assert config.get('data.raw_file') is not None
        assert config.get('training.test_size') is not None
        assert config.get('paths.artifacts_dir') is not None
    
    def test_config_get_set(self):
        """Test configuration get/set operations"""
        config = Config()
        
        # Test setting and getting values
        config.set('test.value', 42)
        assert config.get('test.value') == 42
        
        # Test default values
        assert config.get('nonexistent.key', 'default') == 'default'
    
    def test_config_directory_creation(self):
        """Test directory creation from config"""
        config = Config()
        
        # This should not raise an exception
        config.create_directories()

class TestModelIntegration:
    """Integration tests for model training pipeline"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(1, 2, 100),
            'feature3': np.random.uniform(0, 10, 100)
        })
        
        # Classification target
        self.y_classification = np.random.choice(['A', 'B', 'C'], 100)
        
        # Regression target
        self.y_regression = np.random.normal(50, 15, 100)
    
    def test_model_training_shapes(self):
        """Test that models can be trained with given data shapes"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Classification
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(self.X, self.y_classification)
        predictions = clf.predict(self.X)
        assert len(predictions) == len(self.y_classification)
        
        # Regression
        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(self.X, self.y_regression)
        predictions = reg.predict(self.X)
        assert len(predictions) == len(self.y_regression)
    
    def test_preprocessing_consistency(self):
        """Test that preprocessing doesn't break data consistency"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        # Add some missing values
        X_with_missing = self.X.copy()
        X_with_missing.iloc[0:5, 0] = np.nan
        
        # Apply preprocessing
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_imputed = imputer.fit_transform(X_with_missing)
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Check shapes are consistent
        assert X_scaled.shape == self.X.shape
        assert not np.isnan(X_scaled).any()

class TestPipelineIntegration:
    """End-to-end pipeline integration tests"""
    
    def test_pipeline_components_exist(self):
        """Test that all pipeline components can be imported"""
        try:
            import load_data
            import feature_engineering
            import preprocess
            import models
            import evaluate
            import predict
            import utils
            import config_loader
        except ImportError as e:
            pytest.fail(f"Failed to import pipeline component: {e}")
    
    def test_artifacts_directory_structure(self):
        """Test that artifact directories are created correctly"""
        from config_loader import get_config
        
        config = get_config()
        config.create_directories()
        
        required_dirs = [
            'artifacts',
            'artifacts/models',
            'artifacts/evaluation',
            'artifacts/eda'
        ]
        
        for directory in required_dirs:
            assert os.path.exists(directory), f"Directory {directory} was not created"

if __name__ == '__main__':
    pytest.main([__file__])