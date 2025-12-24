# tests/test_stitch_boundaries.py

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from spcproc.config import DEFAULT_STITCH_BOUNDS


class TestStitchBoundaries: 
    """验证拼接边界与 R 代码的一致性"""
    
    @pytest.fixture
    def psi_data(self):
        """加载 PSI 参考数据"""
        data_path = Path("data/reference/FTIR_raw_spectra.csv")
        if not data_path.exists():
            pytest.skip("PSI reference data not found")
        return pd.read_csv(data_path)
    
    def test_stitch_boundary_indices(self, psi_data):
        """测试拼接边界的索引是否与 R 代码一致"""
        Wn = psi_data['Wavenumber'].values
        low_mid_wn = DEFAULT_STITCH_BOUNDS[1]  # 1336.5996
        mid_high_wn = DEFAULT_STITCH_BOUNDS[0]  # 2069.5114
        
        idx_low_mid = np.argmin(np.abs(Wn - low_mid_wn))
        idx_mid_high = np.argmin(np. abs(Wn - mid_high_wn))
        
        # R 代码中的索引 (1-based) 转换为 Python (0-based)
        expected_low_mid = 1379
        expected_mid_high = 999
        
        assert idx_low_mid == expected_low_mid, \
            f"low_mid index mismatch: got {idx_low_mid}, expected {expected_low_mid}"
        
        assert idx_mid_high == expected_mid_high, \
            f"mid_high index mismatch: got {idx_mid_high}, expected {expected_mid_high}"
    
    def test_stitch_boundary_wavenumbers(self, psi_data):
        """测试边界点的实际波数值"""
        Wn = psi_data['Wavenumber'].values
        
        # 检查 R 代码硬编码的索引对应的波数
        assert np.isclose(Wn[1379], 1336.5996, atol=0.001), \
            f"Wn[1379] = {Wn[1379]}, expected ≈ 1336.60"
        
        assert np.isclose(Wn[999], 2069.5114, atol=0.001), \
            f"Wn[999] = {Wn[999]}, expected ≈ 2069.51"
    
    def test_mask_continuity(self, psi_data):
        """测试 mask 是否连续覆盖整个数组"""
        Wn = psi_data['Wavenumber'].values
        low_mid_wn = DEFAULT_STITCH_BOUNDS[1]
        mid_high_wn = DEFAULT_STITCH_BOUNDS[0]
        
        low_mask = Wn <= low_mid_wn
        mid_mask = (Wn > low_mid_wn) & (Wn <= mid_high_wn)
        high_mask = Wn > mid_high_wn
        
        # 每个点应该且只应该在一个 mask 中
        total_coverage = low_mask.sum() + mid_mask.sum() + high_mask.sum()
        assert total_coverage == len(Wn), \
            f"Mask coverage mismatch: {total_coverage} != {len(Wn)}"
        
        # 检查没有重叠
        assert not np.any(low_mask & mid_mask), "low_mask and mid_mask overlap"
        assert not np.any(mid_mask & high_mask), "mid_mask and high_mask overlap"
        assert not np.any(low_mask & high_mask), "low_mask and high_mask overlap"
    
    def test_data_length(self, psi_data):
        """测试数据长度是否为预期的 1866"""
        assert len(psi_data) == 1866, \
            f"Data length is {len(psi_data)}, expected 1866"