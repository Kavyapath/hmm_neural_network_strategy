"""Assembles each ablation variant's per-ticker feature frame from three
independently-switchable blocks: technical (always on), regime (Step 3's
HMM), volatility (Step 9's GARCH(1,1) vol forecast). See
STEP_8_BASELINES_ABLATION.md#feature-block-switching-per-ticker."""
from __future__ import annotations

import pandas as pd

from causal_hmm import CausalHMM
from regime_features import build_regime_features
from volatility_features import build_volatility_features, GarchParams
from technical_features import build_technical_features

# Which blocks feed which ablation variant (STEP_8_BASELINES_ABLATION.md's
# variant table). A has no model at all — every candidate is taken
# unconditionally — so it needs no feature frame.
VARIANT_BLOCKS: dict[str, frozenset[str]] = {
    "A_rule_based": frozenset(),
    "B_logreg": frozenset({"technical", "regime", "volatility"}),
    "C_gru_no_regime": frozenset({"technical", "volatility"}),
    "D_gru_full": frozenset({"technical", "regime", "volatility"}),
    "E_gru_no_volatility": frozenset({"technical", "regime"}),
}


def build_feature_frame(
    close: pd.Series,
    blocks: frozenset[str],
    regime_model: CausalHMM | None = None,
    volatility_model: GarchParams | None = None,
) -> pd.DataFrame:
    frames = []
    if "technical" in blocks:
        frames.append(build_technical_features(close))
    if "regime" in blocks:
        if regime_model is None:
            raise ValueError("'regime' block requested but no regime_model was given")
        frames.append(build_regime_features(close, regime_model))
    if "volatility" in blocks:
        if volatility_model is None:
            raise ValueError("'volatility' block requested but no volatility_model was given")
        frames.append(build_volatility_features(close, volatility_model))

    if not frames:
        return pd.DataFrame(index=close.index)

    combined = pd.concat(frames, axis=1, join="inner")
    return combined.dropna()