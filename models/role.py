"""
ROLE: Refinement-based Outlier-Logit Enhancement
=================================================

Two-phase few-shot open-set classification.

"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# NumPy helpers
# ---------------------------------------------------------------------------

def _l2_norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _softmax_np(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# ROLE
# ---------------------------------------------------------------------------

class ROLE:
    """
    Refinement-based Outlier-Logit Enhancement.

    Args:
        temperature : cosine logit temperature tau (default 15.0)
        lr          : Adam learning rate (default 1e-3)
        b           : outlier prior -- should match the episode outlier ratio
        T_boot      : Phase-1 BCD iterations (default 3)
        lambda_xi   : Phase-1 inlierness temperature (default 0.2)
        T_cal       : Phase-2 Adam iterations (default 3)
        lambda_q    : Phase-2 conditional-entropy weight (default 0.1)
        lambda_c    : Phase-2 marginal-entropy weight (default 1.0)
    """

    def __init__(
        self,
        temperature: float = 15.0,
        lr: float = 1e-3,
        b: float = 0.5,
        T_boot: int = 3,
        lambda_xi: float = 0.2,
        T_cal: int = 3,
        lambda_q: float = 0.1,
        lambda_c: float = 1.0,
        **_legacy_kwargs,   # absorb old params (num_iter, use_bcd, oslo_T_warm, ...)
    ):
        self.temperature = temperature
        self.lr          = lr
        self.b           = b
        self.T_boot      = T_boot
        self.lambda_xi   = lambda_xi
        self.T_cal       = T_cal
        self.lambda_q    = lambda_q
        self.lambda_c    = lambda_c

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _phase1_bcd(
        self,
        Zs: np.ndarray,   # (N_s, D)  L2-normalised support
        Ys: np.ndarray,   # (N_s,)
        Zq: np.ndarray,   # (N_q, D)  L2-normalised queries
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run T_boot BCD steps with b-prior shift.

        Returns:
            mu : (K, D)  refined prototypes, L2-normalised
            xi : (N_q,)  inlierness in (0, 1)
        """
        K, N_q, D = int(Ys.max()) + 1, Zq.shape[0], Zs.shape[1]
        tau   = self.temperature
        prior = math.log((1.0 - self.b) / (self.b + 1e-12))

        mu_sup = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            mask = (Ys == k)
            if mask.any():
                mu_sup[k] = Zs[mask].sum(0)

        mu = _l2_norm(mu_sup.copy())
        xi = np.ones(N_q, dtype=np.float32)

        for _ in range(self.T_boot):
            s  = tau * (Zq @ mu.T)
            pi = _softmax_np(s)
            Z  = _softmax_np(xi[:, None] * s)
            xi = _sigmoid_np(
                (Z * np.log(pi + 1e-12)).sum(axis=1) / self.lambda_xi + prior
            )
            mu = _l2_norm(mu_sup + (xi[:, None] * Z).T @ Zq)

        return mu, xi

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _phase2_xi_gated(
        self,
        Zs_raw:  np.ndarray,   # (N_s, D)  raw support
        Ys:      np.ndarray,   # (N_s,)
        Zq_raw:  np.ndarray,   # (N_q, D)  raw queries
        mu_init: np.ndarray,   # (K, D)    Phase-1 prototypes (L2-norm space)
        xi:      np.ndarray,   # (N_q,)    fixed inlierness weights
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimise prototypes with xi-weighted losses, then score outliers.

        Returns:
            predictions    : (N_q,) int
            probs          : (N_q, K) float
            outlier_scores : (N_q,) float  -- higher = more outlier
        """
        Zs = _l2_norm(Zs_raw.astype(np.float32))
        Zq = _l2_norm(Zq_raw.astype(np.float32))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        K     = int(Ys.max()) + 1
        log_K = math.log(K)
        tau   = self.temperature

        # Episode-level centring
        phi   = np.concatenate([Zs, Zq], axis=0).mean(axis=0, keepdims=True).astype(np.float32)
        sup_t = torch.from_numpy(Zs - phi).to(device)
        qry_t = torch.from_numpy(Zq - phi).to(device)
        lbl_t = torch.from_numpy(Ys.astype(np.int64)).to(device)
        xi_t  = torch.from_numpy(xi.clip(1e-6, 1.0 - 1e-6)).to(device)

        proto = F.normalize(
            torch.from_numpy(mu_init - phi).to(device), dim=1
        ).clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([proto], lr=self.lr)

        oh = torch.zeros(len(Ys), K, device=device)
        oh.scatter_(1, lbl_t.unsqueeze(1), 1.0)

        for _ in range(self.T_cal):
            opt.zero_grad()
            pn = F.normalize(proto, dim=1)

            # Support cross-entropy
            ls = tau * (F.normalize(sup_t, dim=1) @ pn.T)
            ce = -(oh * F.log_softmax(ls, dim=1)).sum() / len(Ys)

            # Query posteriors
            lq  = tau * (F.normalize(qry_t, dim=1) @ pn.T)
            pjk = F.softmax(lq, dim=1)

            # xi-weighted conditional entropy (minimise)
            Hj   = -(pjk * torch.log(pjk + 1e-12)).sum(dim=1)
            xi_s = xi_t.sum() + 1e-12
            l_ce = self.lambda_q * (xi_t * Hj).sum() / xi_s

            # xi-weighted marginal entropy (maximise)
            p_hat = (xi_t.unsqueeze(1) * pjk).sum(0) / xi_s
            l_me  = self.lambda_c * (p_hat * torch.log(p_hat + 1e-12)).sum()

            (ce + l_ce + l_me).backward()
            opt.step()
            with torch.no_grad():
                proto.data = F.normalize(proto.data, dim=1)

        with torch.no_grad():
            pn  = F.normalize(proto, dim=1)
            lf  = tau * (F.normalize(qry_t, dim=1) @ pn.T)
            lse = torch.logsumexp(lf, dim=1)
            out = torch.sigmoid(-lse + log_K - math.log(self.b + 1e-12))

        return (
            lf.argmax(dim=1).cpu().numpy().astype(int),
            F.softmax(lf, dim=1).cpu().numpy().astype(np.float32),
            out.cpu().numpy().astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        support_embeddings: np.ndarray,   # (N_s, D)
        support_labels:     np.ndarray,   # (N_s,)
        query_embeddings:   np.ndarray,   # (N_q, D)
        **_legacy_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Single-episode inference: Phase-1 BCD -> Phase-2 xi-gated optimisation.

        Returns:
            predictions    : (N_q,) int
            probs          : (N_q, K) float
            outlier_scores : (N_q,) float  -- higher = more outlier
        """
        Zs = _l2_norm(support_embeddings.astype(np.float32))
        Zq = _l2_norm(query_embeddings.astype(np.float32))
        mu, xi = self._phase1_bcd(Zs, support_labels, Zq)
        return self._phase2_xi_gated(
            support_embeddings, support_labels, query_embeddings, mu, xi
        )
