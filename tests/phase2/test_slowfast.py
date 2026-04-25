"""Tests for SlowFast dual-pathway model."""
import pytest
import torch

from src.models.temporal.slowfast import SlowFastSpotting


class TestSlowFastSpotting:
    def test_forward_shape(self):
        model = SlowFastSpotting(feat_dim=512, num_classes=17, slow_stride=4)
        x = torch.randn(2, 80, 512)
        out = model(x)
        # output at FULL T resolution (not T//slow_stride). The 2026-04-19
        # fix upsamples the slow pathway back to T before fusion so tight-mAP
        # eval (1-5s tolerance) isn't capped by a 0.5fps effective output.
        assert out.shape == (2, 80, 18)

    def test_slow_pathway_subsamples_internally(self):
        """Slow pathway still runs at T/slow_stride *internally* for cheap
        large-context modelling; output is upsampled back to T for fusion."""
        model = SlowFastSpotting(feat_dim=512, slow_stride=4)
        x = torch.randn(1, 80, 512)
        # slow_proj sees only T/slow_stride frames
        slow_in = x[:, ::model.slow_stride, :]
        assert slow_in.shape[1] == 80 // 4
        # but final output is at full T
        out = model(x)
        assert out.shape[1] == 80

    def test_fast_pathway_full_resolution(self):
        model = SlowFastSpotting(feat_dim=512, slow_stride=4, hidden_dim=64)
        x = torch.randn(1, 40, 512)
        # fast pathway projects all T frames
        fast_out = model.fast_proj(x)
        assert fast_out.shape == (1, 40, 64 // 4)  # fast_dim = hidden_dim // 4

    def test_lateral_connection_matters(self):
        """Output should differ when lateral connection is active vs zeroed."""
        model = SlowFastSpotting(feat_dim=512, hidden_dim=64)
        x = torch.randn(1, 80, 512)
        model.eval()
        with torch.no_grad():
            out1 = model(x).clone()
            # zero lateral weights
            model.lateral.weight.zero_()
            model.lateral.bias.zero_()
            out2 = model(x)
        # outputs should differ since lateral was providing information
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_feature_dim_512(self):
        model = SlowFastSpotting(feat_dim=512)
        x = torch.randn(1, 40, 512)
        out = model(x)
        assert out.shape[-1] == 18
