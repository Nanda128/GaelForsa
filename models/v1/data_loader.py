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
                for csv_file in cvs_files:
                    print(f"loading {csv_file} ....")
                    with zip_ref.open(csv_file) as file:
                        try:
                            df = pd.read_csv(file, encoding='utf-8', low_memory=False, sep=',', quotechar='"', comment='#')
                        except (UnicodeDecodeError, pd.errors.ParseError):
                            try:
                                df = pd.read_csv(file, encoding='latin1', low_memory=False, sep=',', quotechar='"', comment='#')
                            except (UnicodeDecodeError, pd.errors.ParserError):
                                df = pd.read_csv(file, encoding='cp1252', low_memory=False, sep=',', quotechar='"', comment='#')
                    filename = os.path.basename(csv_file)
                    turbine_match = filename.split('_')
                    if len(turbine_match) >= 3 and turbine_match[2].isdigit():
                        turbine_id = turbine_match[2]
                        df['turbine_id'] = turbine_id

                    all_dataframes.append(df)
                
        else:
            raise ValueError(f"Unsupported data path: {data_path}. Must be a directory or zip file.")
        
        # Basic data cleaning
        timestamp_cols = [col for col in combined_df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            combined_df[timestamp_cols[0]] = pd.to_datetime(combined_df[timestamp_cols[0]], errors='coerce')
            combined_df = combined_df.sort_values(timestamp_cols[0]).dropna(subset=[timestamp_cols[0]])

        # Handle missing values
        combined_df = combined_df.ffill().bfill()

        print(f"Final dataframe: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
        print(f"Columns: {list(combined_df.columns)[:10]}...")  # Show first 10 columns

        return combined_df

    def preprocess_data(self):
        """
        Preprocess data - automatically detect features from real data
        """
        df = self.raw_data.copy()

        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            df[timestamp_cols[0]] = pd.to_datetime(df[timestamp_cols[0]], errors='coerce')
            df = df.sort_values(timestamp_cols[0]).dropna(subset=[timestamp_cols[0]])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_patterns = ['id', 'index', 'timestamp', 'time', 'date']
        feature_cols = [col for col in numeric_cols
                       if not any(pattern in col.lower() for pattern in exclude_patterns)]

        print(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")

        # Handle missing values
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)

        feature_data = df[feature_cols].values

        m_miss = (~np.isnan(df[feature_cols].values)).astype(float)
        X_standardized = self.scaler.fit_transform(feature_data)

        #create synthetic regime flags (since real data might not have them)
        n_samples = len(df)
        flags = np.column_stack([
            np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # curtailment
            np.random.choice([0, 1], n_samples, p=[0.95, 0.05]), # maintenance
            np.random.choice([0, 1], n_samples, p=[0.98, 0.02])  # grid_event
        ])

        self.feature_columns = feature_cols
        self.regime_columns = ['curtailment', 'maintenance', 'grid_event']

        return X_standardized, m_miss, flags

    def create_windows(self, X: np.ndarray, M_miss: np.ndarray, Flags: np.ndarray,
                      window_length: int = 100, forecast_horizon: int = 6) -> List[Dict]:
        """
        Create sliding windows for training.

        Args:
            X: Standardized features (N, F)
            M_miss: Missingness mask (N, F)
            Flags: Regime flags (N, R)
            window_length: L
            forecast_horizon: K

        Returns:
            List of window dictionaries
        """
        windows = []
        n_samples = len(X)

        for i in range(window_length, n_samples - forecast_horizon):
            x_window = X[i-window_length:i].T  # (F, L)
            m_miss_window = M_miss[i-window_length:i].T  # (F, L)
            flags_window = Flags[i-window_length:i].T  # (R, L)
            y_true = X[i:i+forecast_horizon].T  # (F, K) -> will reshape to (K, F)
            if 'fault_class' in self.raw_data.columns:
                fault_label = self.raw_data.iloc[i]['fault_class']
            else:
                #synthetic fault labels
                fault_label = np.random.choice([0, 1, 2, 3, 4], p=[0.8, 0.05, 0.05, 0.05, 0.05])

            windows.append({
                'x': x_window,
                'm_miss': m_miss_window,
                'flags': flags_window,
                'y_true': y_true.T,  # (K, F)
                'y_fault': fault_label
            })

        return windows


class SCADADataset(Dataset):
    """PyTorch Dataset for SCADA windows."""

    def __init__(self, windows: List[Dict]):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]

        return {
            'x': torch.FloatTensor(window['x']),
            'm_miss': torch.FloatTensor(window['m_miss']),
            'flags': torch.FloatTensor(window['flags']),
            'y_true': torch.FloatTensor(window['y_true']),
            'y_fault': torch.LongTensor([window['y_fault']])
        }


def create_data_loaders(data_path: str = '5841834', batch_size: int = 32,
                       window_length: int = 100, forecast_horizon: int = 6,
                       train_split: float = 0.7):
    """
    Create train/val DataLoaders.

    Args:
        data_path: Path to data zip
        batch_size: Batch size
        window_length: Input window length L
        forecast_horizon: Forecast horizon K
        train_split: Train/validation split ratio

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load and preprocess data
    loader = SCADADataLoader(data_path)
    X, M_miss, Flags = loader.preprocess_data()
    windows = loader.create_windows(X, M_miss, Flags, window_length, forecast_horizon)
    n_train = int(len(windows) * train_split)
    train_windows = windows[:n_train]
    val_windows = windows[n_train:]
    train_dataset = SCADADataset(train_windows)
    val_dataset = SCADADataset(val_windows)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_data_loaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    batch = next(iter(train_loader))
    print(f"Batch x shape: {batch['x'].shape}")  # (B, F, L)
    print(f"Batch flags shape: {batch['flags'].shape}")  # (B, R, L)
    print(f"Batch y_true shape: {batch['y_true'].shape}")  # (B, K, F)