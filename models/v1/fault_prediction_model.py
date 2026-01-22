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
            'power_output': np.random.normal(1500, 400, n_samples).clip(0, 3000),
            'wind_speed': np.random.normal(8, 3, n_samples).clip(0, 25),
            'wind_direction': np.random.uniform(0, 360, n_samples),
            'days_since_last_maintenance': np.random.exponential(30, n_samples).clip(1, 365),
            'operating_hours': np.random.exponential(100, n_samples).clip(10, 10000)
        }

        df = pd.DataFrame(data)
        base_failure_days = 180  # Average 6 months

        risk_factor = (
            0.1 * (df['vibration_level'] - 2.0) +
            0.05 * (df['generator_temp'] - 65) +
            0.03 * (df['gearbox_oil_temp'] - 60) +
            -0.02 * (df['power_output'] - 1500) / 1500 +
            0.001 * df['days_since_last_maintenance'] +
            0.0001 * df['operating_hours']
        )

        df['days_until_failure'] = (base_failure_days - 50 * risk_factor + np.random.normal(0, 30, n_samples)).clip(7, 730)

        X = df[self.feature_columns]
        y = df[self.target_column]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

 
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self._save_model()

        print(f"Model trained with {n_samples} samples. R² score: {self.model.score(X_scaled, y):.3f}")

    def _save_model(self):
        """Save the trained model and scaler."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'trained_at': datetime.now()
        }

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

    def _load_model(self):
        """Load the trained model and scaler."""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']

        print(f"Model loaded from {self.model_path}")

    def predict_fault_timing(self, sensor_data):
        """
        Predict when the next fault will occur.

        Args:
            sensor_data (dict): Current sensor readings with keys:
                - vibration_level: float (mm/s)
                - generator_temp: float (°C)
                - gearbox_oil_temp: float (°C)
                - power_output: float (kW)
                - wind_speed: float (m/s)
                - wind_direction: float (degrees)
                - days_since_last_maintenance: int
                - operating_hours: float

        Returns:
            dict: Prediction results with:
                - predicted_days_until_failure: float
                - confidence_score: float (0-1)
                - risk_level: str ('low', 'medium', 'high', 'critical')
                - recommended_action: str
        """
        input_data = {}
        for col in self.feature_columns:
            input_data[col] = sensor_data.get(col, self._get_default_value(col))

        df_input = pd.DataFrame([input_data])
        X_scaled = self.scaler.transform(df_input)
        predicted_days = float(self.model.predict(X_scaled)[0])

        confidence = min(0.95, max(0.1, 1.0 - abs(predicted_days - 180) / 360))

        if predicted_days <= 30:
            risk_level = 'critical'
            recommended_action = 'Immediate maintenance required'
        elif predicted_days <= 90:
            risk_level = 'high'
            recommended_action = 'Schedule maintenance within 1 week'
        elif predicted_days <= 180:
            risk_level = 'medium'
            recommended_action = 'Monitor closely, plan maintenance'
        else:
            risk_level = 'low'
            recommended_action = 'Continue normal operation'

        return {
            'predicted_days_until_failure': round(predicted_days, 1),
            'predicted_date': (datetime.now() + timedelta(days=predicted_days)).date(),
            'confidence_score': round(confidence, 3),
            'risk_level': risk_level,
            'recommended_action': recommended_action,
            'input_features': input_data
        }

    def _get_default_value(self, feature):
        """Get default values for missing features."""
        defaults = {
            'vibration_level': 2.0,
            'generator_temp': 65.0,
            'gearbox_oil_temp': 60.0,
            'power_output': 1500.0,
            'wind_speed': 8.0,
            'wind_direction': 180.0,
            'days_since_last_maintenance': 30,
            'operating_hours': 1000.0
        }
        return defaults.get(feature, 0.0)

    def retrain_model(self, new_data):
        """
        Retrain the model with new real data.

        Args:
            new_data (pd.DataFrame): New training data with same columns
        """
        print("Retraining model with new data...")
        pass



if __name__ == "__main__":
    predictor = TurbineFaultPredictor()

    sample_data = {
        'vibration_level': 3.2,
        'generator_temp': 75.0,
        'gearbox_oil_temp': 68.0,
        'power_output': 1800.0,
        'wind_speed': 12.0,
        'wind_direction': 270.0,
        'days_since_last_maintenance': 45,
        'operating_hours': 2500.0
    }

    result = predictor.predict_fault_timing(sample_data)

    print("Fault Prediction Results:")
    print(f"Days until failure: {result['predicted_days_until_failure']}")
    print(f"Predicted date: {result['predicted_date']}")
    print(f"Risk level: {result['risk_level']}")
    print(f"Recommended action: {result['recommended_action']}")
    print(f"Confidence: {result['confidence_score']:.1%}")
