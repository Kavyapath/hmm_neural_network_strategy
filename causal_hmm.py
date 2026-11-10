from __future__ import annotations

import numpy as np


class CausalHMM:
    def __init__(self, n_states: int = 3, random_state: int = 42, n_iter: int = 200):
        self.n_states = n_states
        self.random_state = random_state
        self.n_iter = n_iter
        self.pi_: np.ndarray | None = None      # (K,) initial state distribution
        self.A_: np.ndarray | None = None        # (K, K) row-stochastic transition matrix
        self.means_: np.ndarray | None = None     # (K,) emission means
        self.vars_: np.ndarray | None = None      # (K,) emission variances
        self._filtered: np.ndarray | None = None   # running filtered state vector
        self._log_likelihood: float = 0.0

    # ------------------------------------------------------------------
    # Fitting — the ONLY place parameters are allowed to change.
    # ------------------------------------------------------------------
    def fit(self, train_observations: np.ndarray) -> "CausalHMM":
        from hmmlearn import hmm  # lazy import, same convention as the plugin

        x = np.asarray(train_observations, dtype=float).reshape(-1, 1)
        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(x)

        self.pi_ = model.startprob_.copy()
        self.A_ = model.transmat_.copy()
        self.means_ = model.means_.reshape(-1).copy()
        self.vars_ = np.clip(model.covars_.reshape(-1).copy(), 1e-12, None)

        self._reset_filter()
        return self

    def _check_fitted(self) -> None:
        if self.pi_ is None:
            raise RuntimeError("CausalHMM.fit() must be called before filtering.")

    # ------------------------------------------------------------------
    # Emission density: diagonal Gaussian per state, vectorized over K.
    # ------------------------------------------------------------------
    def _emission_prob(self, obs_t: float) -> np.ndarray:
        diff = obs_t - self.means_
        return np.exp(-0.5 * diff * diff / self.vars_) / np.sqrt(2.0 * np.pi * self.vars_)

    # ------------------------------------------------------------------
    # Online scaled forward algorithm
    # ------------------------------------------------------------------
    def _reset_filter(self) -> None:
        self._filtered = None
        self._log_likelihood = 0.0

    def filter_step(self, obs_t: float) -> np.ndarray:
        """Advance the filter by exactly one observation. Returns filtered_t,
        shape (K,). Depends only on obs_t and the previous internal state —
        never on any future observation."""
        self._check_fitted()
        b = self._emission_prob(obs_t)

        if self._filtered is None:
            alpha = self.pi_ * b          # t = 1: alpha_1[k] = pi[k] * b_k(obs_1)
        else:
            alpha = b * (self._filtered @ self.A_)   # t > 1: scaled forward recursion

        scale = alpha.sum()
        if not np.isfinite(scale) or scale <= 0:
            # Degenerate observation (zero density under every state) —
            # fall back to uniform instead of propagating NaNs/zeros.
            filtered = np.full(self.n_states, 1.0 / self.n_states)
            scale = 1.0
        else:
            filtered = alpha / scale

        self._log_likelihood += np.log(scale)
        self._filtered = filtered
        return filtered

    def filter_sequence(self, observations: np.ndarray) -> np.ndarray:
        """Causal filter over a whole array, shape (T, K).

        filter_sequence(obs)[:t] must be bit-identical to
        filter_sequence(obs[:t]) — that equivalence is the core causality
        test in test_causal_hmm.py.
        """
        self._check_fitted()
        self._reset_filter()
        observations = np.asarray(observations, dtype=float).reshape(-1)
        out = np.empty((len(observations), self.n_states), dtype=float)
        for t, obs_t in enumerate(observations):
            out[t] = self.filter_step(obs_t)
        return out

    @property
    def log_likelihood(self) -> float:
        """Accumulated log-likelihood from the most recent filter_sequence()
        call. Should closely match hmmlearn's own model.score(X) on the same
        data/parameters — use that as a correctness check on this forward
        pass (see Test 4 in STEP_2_CAUSAL_HMM_ADAPTER.md)."""
        return self._log_likelihood

    # ------------------------------------------------------------------
    # Diagnostics / derived quantities
    # ------------------------------------------------------------------
    def transition_matrix(self) -> np.ndarray:
        self._check_fitted()
        return self.A_.copy()

    def next_state_probs(self, filtered_t: np.ndarray) -> np.ndarray:
        """filtered_t pushed one step through A — a probability-weighted
        average over current-state uncertainty. NOT the observable model's
        single-row lookup, and NOT Viterbi's single most-likely next state."""
        self._check_fitted()
        return np.asarray(filtered_t, dtype=float) @ self.A_

    def stationary_distribution(self) -> np.ndarray:
        """Diagnostic only — long-run state mix. Never use as a live feature."""
        self._check_fitted()
        eigvals, eigvecs = np.linalg.eig(self.A_.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        vec = np.abs(np.real(eigvecs[:, idx]))
        return vec / vec.sum()

    def order_by_mean_return(self) -> np.ndarray:
        """Permutation that sorts states by ascending emission mean (for
        human-readable Bear/Sideways/Bull labeling). Cosmetic only — apply
        consistently when reporting, but feed the classifier the raw
        filtered_t vector in the fixed order decided once per fit()."""
        self._check_fitted()
        return np.argsort(self.means_)


def _demo_synthetic_causality_check() -> None:
    """Self-contained smoke test with synthetic regime-switching data (no
    network, no real asset — not evidence of predictive performance, just a
    pipeline check). Generates returns from a KNOWN 3-state Gaussian HMM,
    fits CausalHMM on it, and proves the causality property numerically."""
    rng = np.random.default_rng(0)
    true_A = np.array([[0.95, 0.05, 0.00],
                        [0.03, 0.94, 0.03],
                        [0.00, 0.05, 0.95]])
    true_means = np.array([-0.01, 0.0005, 0.012])
    true_stds = np.array([0.012, 0.006, 0.010])

    n = 1500
    state = 1
    obs = np.empty(n)
    for t in range(n):
        obs[t] = rng.normal(true_means[state], true_stds[state])
        state = rng.choice(3, p=true_A[state])

    model = CausalHMM(n_states=3, random_state=0).fit(obs[:1000])  # train on first 1000 only

    filtered_short = model.filter_sequence(obs[:100])
    filtered_long = model.filter_sequence(obs[:200])

    causal_ok = np.allclose(filtered_short, filtered_long[:100], atol=1e-10)
    rowsum_ok = np.allclose(model.transition_matrix().sum(axis=1), 1.0)

    print(f"causality check (prefix-independence): {'PASS' if causal_ok else 'FAIL'}")
    print(f"transition matrix row-stochastic:       {'PASS' if rowsum_ok else 'FAIL'}")
    print(f"log-likelihood over first 200 obs:       {model.log_likelihood:.3f}")
    print(f"state order by mean return:              {model.order_by_mean_return()}")


if __name__ == "__main__":
    _demo_synthetic_causality_check()