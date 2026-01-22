"""
Health Prediction Service - Bridges Django models and ML models for real-time predictions.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from django.conf import settings
from django.db.models import Q
from django.utils import timezone

from core.models import Turbine, TurbineLog, OnTurbineReading, HealthPrediction

V1_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', 'models', 'v1')
if os.path.exists(V1_MODEL_PATH):
    sys.path.insert(0, V1_MODEL_PATH)

try:
    from fault_prediction_model import TurbineFaultPredictor
    from scada_tcn_model import SCADATCNModel, SCADATCNTrainer
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML models not available: {e}")
    ML_MODELS_AVAILABLE = False


class HealthPredictionService:
    """
    Service for generating health predictions using ML models.
    Handles data format conversion between Django models and ML models.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the service and load ML models.
        
        Args:
            model_dir: Directory containing trained model files. 
                      Defaults to backend/models/v1/
        """
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'models', 'v1'
        )
        
        self.fault_predictor = None
        self.tcn_model = None
        self.tcn_trainer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Feature mapping from Django model fields to ML model features
        self.feature_mapping = {
            'wind_speed': 'wind_speed',
            'wind_direction': 'wind_direction',
            'power_output': 'power_output',
            'rotor_speed_rpm': 'Rotor Speed (rpm)',
            'generator_rpm': 'Generator Speed (rpm)',
            'blade_pitch_angle': 'Blade Pitch Angle (deg)',
            'gear_oil_temperature': 'Gearbox Oil Temperature (°C)',
            'generator_temperature': 'Generator Winding Temperature (°C)',
            'yaw_position': 'Yaw Position (deg)',
            'reactive_power': 'Reactive Power (kVAR)',
            'vibration_level': 'vibration_level',
            'gearbox_oil_temperature': 'gearbox_oil_temp',
        }
        
        # Window parameters for TCN model
        self.window_length = 100  # L
        self.forecast_horizon = 6  # K
        
        if ML_MODELS_AVAILABLE:
            self._load_models()
    
    def _load_models(self):
        """Load trained ML models."""
        try:
            # Load fault prediction model
            fault_model_path = os.path.join(self.model_dir, 'fault_predictor.pkl')
            if os.path.exists(fault_model_path):
                self.fault_predictor = TurbineFaultPredictor(fault_model_path)
            else:
                print(f"Warning: Fault predictor model not found at {fault_model_path}, using default")
                self.fault_predictor = TurbineFaultPredictor()
        except Exception as e:
            print(f"Warning: Could not load fault predictor: {e}")
            self.fault_predictor = TurbineFaultPredictor()
        
        try:
            tcn_model_path = os.path.join(self.model_dir, 'scada_tcn_model.pth')
            if os.path.exists(tcn_model_path):
                checkpoint = torch.load(tcn_model_path, map_location=self.device)
                config = checkpoint.get('config', {})
                
                self.tcn_model = SCADATCNModel(
                    num_features=config.get('num_features', 20),
                    num_regime_flags=config.get('num_regime_flags', 3),
                    hidden_channels=config.get('hidden_channels', 64),
                    num_tcn_layers=config.get('num_tcn_layers', 4),
                    forecast_horizon=self.forecast_horizon,
                    num_fault_classes=5
                )
                self.tcn_model.load_state_dict(checkpoint['model_state_dict'])
                self.tcn_model.to(self.device)
                self.tcn_model.eval()
                self.tcn_trainer = SCADATCNTrainer(self.tcn_model, self.device)
                print(f"Loaded TCN model from {tcn_model_path}")
            else:
                print(f"Warning: TCN model not found at {tcn_model_path}, TCN predictions will be unavailable")
        except Exception as e:
            print(f"Warning: Could not load TCN model: {e}")
    
    def _get_recent_readings(self, turbine: Turbine, current_log: TurbineLog, 
                            window_size: int = None) -> pd.DataFrame:
        """
        Get recent readings for a turbine to form a window for ML models.
        
        Args:
            turbine: Turbine instance
            current_log: Current TurbineLog to predict for
            window_size: Number of readings to retrieve (default: window_length)
        
        Returns:
            DataFrame with readings ordered by timestamp
        """
        window_size = window_size or self.window_length
        
        logs = TurbineLog.objects.filter(
            turbine=turbine,
            timestamp__lte=current_log.timestamp
        ).order_by('-timestamp')[:window_size]
        
        readings_data = []
        for log in reversed(logs): 
            reading = OnTurbineReading.objects.filter(log=log).first()
            if reading:
                reading_dict = {
                    'timestamp': log.timestamp,
                    'wind_speed': reading.wind_speed,
                    'wind_direction': reading.wind_direction,
                    'power_output': reading.power_output,
                    'rotor_speed_rpm': reading.rotor_speed_rpm,
                    'generator_rpm': reading.generator_rpm,
                    'blade_pitch_angle': reading.blade_pitch_angle,
                    'gear_oil_temperature': reading.gear_oil_temperature or reading.gearbox_oil_temperature,
                    'generator_temperature': reading.generator_temperature,
                    'yaw_position': reading.yaw_position,
                    'reactive_power': reading.reactive_power,
                    'vibration_level': reading.vibration_level,
                    'gearbox_oil_temperature': reading.gearbox_oil_temperature or reading.gear_oil_temperature,
                    'gearbox_oil_pressure': reading.gearbox_oil_pressure,
                    'front_bearing_temperature': reading.front_bearing_temperature,
                    'rear_bearing_temperature': reading.rear_bearing_temperature,
                    'nacelle_temperature': reading.nacelle_temperature,
                    'transformer_temperature': reading.transformer_temperature,
                }
                readings_data.append(reading_dict)
        
        if not readings_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(readings_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def _prepare_tcn_input(self, readings_df: pd.DataFrame) -> Optional[Dict]:
        """
        Prepare input for TCN model from readings DataFrame.
        
        Args:
            readings_df: DataFrame with recent readings
        
        Returns:
            Dict with 'x', 'm_miss', 'flags' tensors, or None if insufficient data
        """
        if len(readings_df) < self.window_length:
            padding_needed = self.window_length - len(readings_df)
            first_row = readings_df.iloc[0:1]
            padding = pd.concat([first_row] * padding_needed, ignore_index=True)
            readings_df = pd.concat([padding, readings_df], ignore_index=True)
        
        # Select window
        window_df = readings_df.tail(self.window_length)
        feature_cols = [
            'wind_speed', 'wind_direction', 'power_output',
            'rotor_speed_rpm', 'generator_rpm', 'blade_pitch_angle',
            'gear_oil_temperature', 'generator_temperature', 'yaw_position',
            'reactive_power', 'vibration_level', 'gearbox_oil_temperature',
            'gearbox_oil_pressure', 'front_bearing_temperature',
            'rear_bearing_temperature', 'nacelle_temperature',
            'transformer_temperature'
        ]
        
        available_cols = [col for col in feature_cols if col in window_df.columns]
        X = window_df[available_cols].values.astype(np.float32)
        m_miss = (~np.isnan(X)).astype(np.float32)
        X_mean = np.nanmean(X, axis=0, keepdims=True)
        X_std = np.nanstd(X, axis=0, keepdims=True) + 1e-8
        X_standardized = (X - X_mean) / X_std
        
        n_samples = len(window_df)
        flags = np.zeros((n_samples, 3), dtype=np.float32)  # [curtailment, maintenance, grid_event]
        
        X_t = X_standardized.T
        m_miss_t = m_miss.T
        flags_t = flags.T
        
        return {
            'x': torch.FloatTensor(X_t).unsqueeze(0),  # (1, F, L)
            'm_miss': torch.FloatTensor(m_miss_t).unsqueeze(0),  # (1, F, L)
            'flags': torch.FloatTensor(flags_t).unsqueeze(0),  # (1, R, L)
        }
    
    def _prepare_fault_predictor_input(self, reading: OnTurbineReading) -> Dict:
        """
        Prepare input for fault prediction model.
        
        Args:
            reading: OnTurbineReading instance
        
        Returns:
            Dict with sensor data for fault predictor
        """
        turbine = reading.log.turbine
        last_maintenance = turbine.maintenance_events.filter(
            event_type__in=['maintenance', 'repair']
        ).order_by('-start_time').first()
        
        days_since_maintenance = 30  # Default
        if last_maintenance:
            days_since_maintenance = (reading.log.timestamp - last_maintenance.start_time).days
        
        operating_hours = 1000.0  # Default
        
        return {
            'vibration_level': reading.vibration_level or 2.0,
            'generator_temp': reading.generator_temperature or 65.0,
            'gearbox_oil_temp': reading.gearbox_oil_temperature or reading.gear_oil_temperature or 60.0,
            'power_output': reading.power_output or 1500.0,
            'wind_speed': reading.wind_speed or 8.0,
            'wind_direction': reading.wind_direction or 180.0,
            'days_since_last_maintenance': days_since_maintenance,
            'operating_hours': operating_hours,
        }
    
    def predict_health(self, log: TurbineLog) -> Optional[HealthPrediction]:
        """
        Generate health prediction for a given TurbineLog.
        
        Args:
            log: TurbineLog instance to predict for
        
        Returns:
            HealthPrediction instance or None if prediction fails
        """
        try:
            reading = OnTurbineReading.objects.filter(log=log).first()
            if not reading:
                return None
            
            turbine = log.turbine
            
            readings_df = self._get_recent_readings(turbine, log)
            if readings_df.empty:
                return None
            
            health_score = 0.8  
            failure_probability = 0.2 
            predicted_failure_window_start = None
            predicted_failure_window_end = None
            features_used = {}
            
            if self.tcn_model and self.tcn_trainer:
                try:
                    tcn_input = self._prepare_tcn_input(readings_df)
                    if tcn_input:
                        predictions = self.tcn_trainer.predict(
                            tcn_input['x'],
                            tcn_input['m_miss'],
                            tcn_input['flags']
                        )
                        
                        p_fault = predictions['p_fault'][0]  # (C,)
                        fault_probs = p_fault.numpy()
                        
        
                        failure_probability = float(np.sum(fault_probs[1:])) 
                        
                        health_score = max(0.0, min(1.0, 1.0 - failure_probability))
                        
                        features_used['tcn_fault_probs'] = fault_probs.tolist()
                        features_used['tcn_model_used'] = True
                except Exception as e:
                    print(f"Error in TCN prediction: {e}")
                    features_used['tcn_error'] = str(e)
            

            if self.fault_predictor:
                try:
                    fault_input = self._prepare_fault_predictor_input(reading)
                    fault_prediction = self.fault_predictor.predict_fault_timing(fault_input)
                    
                    predicted_days = fault_prediction['predicted_days_until_failure']
                    
                    # Update failure probability if higher
                    if predicted_days <= 30:
                        failure_probability = max(failure_probability, 0.7)
                    elif predicted_days <= 90:
                        failure_probability = max(failure_probability, 0.5)
                    
                    # Set failure window
                    predicted_failure_window_start = log.timestamp + timedelta(days=max(1, int(predicted_days - 7)))
                    predicted_failure_window_end = log.timestamp + timedelta(days=int(predicted_days + 7))
                    
                    features_used['fault_predictor'] = fault_prediction
                    features_used['fault_predictor_used'] = True
                except Exception as e:
                    print(f"Error in fault predictor: {e}")
                    features_used['fault_predictor_error'] = str(e)
            
            
            if features_used.get('tcn_model_used') and features_used.get('fault_predictor_used'):
                tcn_weight = 0.6
                fault_weight = 0.4
                failure_probability = (
                    tcn_weight * failure_probability +
                    fault_weight * min(1.0, features_used['fault_predictor']['predicted_days_until_failure'] / 180.0)
                )
                health_score = max(0.0, min(1.0, 1.0 - failure_probability))
            
            # Create or update prediction
            prediction, created = HealthPrediction.objects.update_or_create(
                turbine=turbine,
                log=log,
                defaults={
                    'health_score': health_score,
                    'failure_probability': failure_probability,
                    'predicted_failure_window_start': predicted_failure_window_start,
                    'predicted_failure_window_end': predicted_failure_window_end,
                    'model_version': 'v1',
                    'features_used': features_used,
                }
            )
            
            return prediction
            
        except Exception as e:
            print(f"Error generating health prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def batch_predict(self, logs: List[TurbineLog]) -> List[HealthPrediction]:
        """
        Generate predictions for multiple logs.
        
        Args:
            logs: List of TurbineLog instances
        
        Returns:
            List of HealthPrediction instances
        """
        predictions = []
        for log in logs:
            prediction = self.predict_health(log)
            if prediction:
                predictions.append(prediction)
        return predictions
