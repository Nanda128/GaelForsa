import pandas as pd 
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, Dataloader 

class SCADADataLoader:
    def __init__(self, data_path: str = 'data'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_columns = [
           'Wind Speed (m/s)', 'Wind Direction (deg)', 'Ambient Temperature (°C)',
            'Rotor Speed (rpm)', 'Generator Speed (rpm)', 'Generator Torque (kNm)',
            'Active Power (kW)', 'Reactive Power (kVAR)', 'Blade Pitch Angle (deg)',
            'Gearbox Oil Temperature (°C)', 'Generator Winding Temperature (°C)',
            'Nacelle Orientation (deg)', 'Yaw Position (deg)' 
        ]

        self.angle_columns = ['Wind Direction (deg)', 'Blade Pitch Angle (deg)',
                             'Nacelle Orientation (deg)', 'Yaw Position (deg)']
        try:
            self.raw_data = self.load_real_data(self.data_path)
            print("successfully loaded real data!")
        except Exception as e:
            print(f"could not load real data: {e}")
            print("Using placeholder data instead")
            self.raw_data = self._load_placeholder_data()#

    def _load_placeholder_data(self) -> pd.DataFrame:
        """Load placeholder data for development."""
        print("Loading placeholder SCADA data...")

        np.random.seed(42)
        n_samples = 10000
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='10min')

        data = {'timestamp': timestamps}


        for col in self.feature_columns:
            if 'Temperature' in col:
                data[col] = np.random.normal(60, 15, n_samples).clip(0, 120)
            elif 'Speed' in col and 'rpm' in col:
                data[col] = np.random.normal(1500, 200, n_samples).clip(0, 2000)
            elif 'Power' in col:
                data[col] = np.random.normal(1500, 400, n_samples).clip(0, 3000)
            elif 'Direction' in col or 'Angle' in col or 'Orientation' in col or 'Position' in col:
                data[col] = np.random.uniform(0, 360, n_samples)
            elif 'Torque' in col:
                data[col] = np.random.normal(500, 100, n_samples).clip(0, 1000)
            else:
                data[col] = np.random.normal(10, 3, n_samples)

        data['turbine_id'] = np.random.choice(['T001', 'T002', 'T003'], n_samples)
        data['curtailment'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        data['maintenance'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        data['grid_event'] = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
        data['fault_class'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.8, 0.05, 0.05, 0.05, 0.05])

        return pd.DataFrame(data)

    def load_real_data(self, data_path: str):
        """Load real SCADA data from zip file or directory."""
        import os
        import zipfile

        print(f"Loading real data from {data_path}...")

        all_dataframes = []
        if os.path.isdir(data_path):
            print("Loading from directory...")
            csv_files = []
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.lower().endswith('.csv') and 'turbine_data' in file.lower():
                        csv_files.append(os.path.join(root, file))

            if not csv_files:
                raise ValueError("No Turbine_Data CSV files found in directory")

            print(f"Found {len(csv_files)} Turbine_Data CSV files")

      #load turbine_data
            for csv_file in csv_files:
                print(f"Loading {csv_file}...")
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False, sep=',', quotechar='"', comment='#')
                except (UnicodeDecodeError, pd.errors.ParserError):
                    try:
                        df = pd.read_csv(csv_file, encoding='latin1', low_memory=False, sep=',', quotechar='"', comment='#')
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        df = pd.read_csv(csv_file, encoding='cp1252', low_memory=False, sep=',', quotechar='"', comment='#')

               #turbine id get
                filename = os.path.basename(csv_file)
                turbine_match = filename.split('_')
                if len(turbine_match) >= 3 and turbine_match[2].isdigit():
                    turbine_id = turbine_match[2]
                    df['turbine_id'] = turbine_id

                all_dataframes.append(df)

        elif data_path.lower().endswith('.zip'):

            with zipfile.ZipFile(data_path, 'r') as zip_ref:
                #load files from zip
                file_list = zip_ref.namelist()
                print(f"Files in zip: {file_list}")

                csv_files = [f for f in file_list if f.lower().endswith('.csv') and 'turbine_data' in f.lower()]
                
                if not csv_files:
                    raise ValueError("No Turbine_Data CSV files found in zip")       
                
                #load all data files
                #get turbine id
                #combine dataframes
                #missing values timestamps faults...