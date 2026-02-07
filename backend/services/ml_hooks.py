def on_initial_upload(turbine_id: int, csv_path: str, feature_csv_path: str | None = None) -> None:
    # Placeholder hook for ML team to implement training.
    print(f"[ML Hook] initial upload turbine={turbine_id} data={csv_path} features={feature_csv_path}")


def on_retrain_trigger(turbine_id: int, csv_path: str) -> None:
    # Placeholder hook for ML team to implement retraining.
    print(f"[ML Hook] retrain turbine={turbine_id} data={csv_path}")
