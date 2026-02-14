from __future__ import annotations

import torch

from scada_tcn.inference.calibration import fit_percentile_thresholds
from scada_tcn.inference.scoring import compute_E_rec, top_contributors
from scada_tcn.training.masking import build_model_input, sample_mask
from scada_tcn.data.windowing import split_window_index
from scada_tcn.config_schema import DataConfig


def test_sample_mask_only_masks_present_values() -> None:
    m_miss = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    m_mask = sample_mask(m_miss, p_mask=1.0, generator=torch.Generator().manual_seed(0))

    # present values are fully masked at p_mask=1.0
    assert torch.equal(m_mask[..., 0], torch.zeros_like(m_mask[..., 0]))
    # missing values are forced keep=1 and never masked
    assert torch.equal(m_mask[..., 1], torch.ones_like(m_mask[..., 1]))


def test_build_model_input_channel_order_and_shape() -> None:
    B, L, F, R = 2, 4, 3, 2
    x = torch.randn(B, L, F)
    m_miss = torch.ones(B, L, F)
    m_mask = torch.zeros(B, L, F)
    flags = torch.randn(B, L, R)

    x_in = build_model_input(x, m_miss, m_mask, flags)
    assert x_in.shape == (B, L, F + F + F + R)

    assert torch.equal(x_in[..., :F], x)
    assert torch.equal(x_in[..., F : 2 * F], m_miss)
    assert torch.equal(x_in[..., 2 * F : 3 * F], m_mask)
    assert torch.equal(x_in[..., 3 * F :], flags)


def test_reconstruction_error_only_on_scored_positions() -> None:
    x = torch.tensor([[[2.0, 4.0]]])
    x_hat = torch.tensor([[[1.5, 10.0]]])
    m_score = torch.tensor([[[0.0, 1.0]]])

    e_rec = compute_E_rec(x, x_hat, m_score)
    assert torch.allclose(e_rec, torch.tensor([[[0.5, 0.0]]]))


def test_top_contributors_clamps_when_topn_exceeds_feature_count() -> None:
    contrib = torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32)
    idx, vals = top_contributors(contrib, topn=10)
    assert idx.shape == (1, 3)
    assert vals.shape == (1, 3)


def test_calibration_percentile_range_validation() -> None:
    import pandas as pd

    df = pd.DataFrame({"s_now": [0.1, 0.2], "turbine_id": ["T1", "T1"], "regime_key": ["none", "none"]})
    try:
        fit_percentile_thresholds(df, percentile=1.2)
        raise AssertionError("Expected ValueError for invalid percentile")
    except ValueError:
        pass


def test_time_auto_split_per_turbine_has_each_partition() -> None:
    import pandas as pd

    rows = []
    ts = pd.date_range('2020-01-01', periods=10, freq='h')
    for tid in ['T1', 'T2']:
        for i, t in enumerate(ts):
            rows.append({'turbine_id': tid, 'start_pos': i, 'end_pos': i + 1, 'target_start_pos': i + 1, 'target_end_pos': i + 2, 'start_time': t, 'has_future': True})
    df_idx = pd.DataFrame(rows)

    cfg = DataConfig(
        paths={},
        sampling={'dt_minutes': 10},
        windowing={'L': 1, 'K': 1, 'stride': 1},
        splits={'method': 'time_auto', 'ratios': [0.7, 0.2, 0.1]},
        identifiers={'turbine_id_col': 'turbine_id', 'time_col': 'timestamp'},
        labels={'enabled': False},
        qc={},
    )
    out = split_window_index(df_idx, cfg)
    assert len(out['train']) > 0
    assert len(out['val']) > 0
    assert len(out['test']) > 0
    assert set(out.keys()) == {'train', 'val', 'test'}
