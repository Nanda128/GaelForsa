import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

class TurbineFaultPredictor:
    def __init__(self, model_path='fault_predictor.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'vibration_level', 'generator_temp', 'gearbox_oil_temp',
            'power_output', 'wind_speed', 'wind_direction',
            'days_since_last_maintenance', 'operating_hours'
        ]
        self.target_column = 'days_until_failure'

        if os.path.exists(model_path):
            self._load_model()
        else:
            self._create_and_train_model()

    def _create_and_train_model(self):
        print("Creating fault prediction model with synthetic data...")
        np.random.seed(42)
        n_samples = 2000

        data = {
            'vibration_level': np.random.normal(2.0, 0.8, n_samples).clip(0.5, 8.0),
            'generator_temp': np.random.normal(65, 15, n_samples).clip(20, 120),
            'gearbox_oil_temp': np.random.normal(60, 10, n_samples).clip(20, 100),
            'power_output': n.random.normal(1500, 400, n_samples).clip(0, 3000),
            'wind_speed': np.random.normal(8, 3, n_samples).clip(0, 25),
            'wind_direction': np.random.uniform(0, 360, n_samples),
            'days_since_last_maintenance': np.random.exponential(30, n_samples).clip(1, 365),
            'operating_hours': np.random.exponential(100, n_samples).clip(10, 10000)
        }

        df = pd.dataFrame(data)

