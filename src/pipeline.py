"""
Full prediction pipeline orchestration.

Implements the six-stage sequential dependency chain from architecture Section 2:

    Stage 1 → load_data()           Data loading and tiering
    Stage 2 → build_elo()           ELO model construction
    Stage 3 → compute_style_axes()  ELO-weighted feature construction (per fighter)
    Stage 4 ↘
    Stage 5 → train_regression()    Matchup features + regression training
    Stage 6 → predict()             Prediction with confidence intervals

Typical usage:
    predictor = MMAPredictor()
    predictor.load_data(Path("data/"))
    predictor.build_elo()
    predictor.train_regression()
    result = predictor.predict("fighter_a_id", "fighter_b_id", WeightClass.LIGHTWEIGHT, date.today())
"""
import dataclasses
import pickle
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import Config, ELOConfig
from .data.loader import (
    filter_tier1_post_era, load_fighter_profiles, load_major_promotion_fights,
    load_sherdog_fights, load_ufcstats_fights, sort_fights_chronologically,
)
from .data.schema import (
    FightRecord, FighterProfile, MatchupFeatures,
    PredictionResult, StyleAxes, WeightClass,
)
from .elo.elo import ELOModel
from .features.construction import apply_cold_start_prior, compute_style_axes
from .matchup.interactions import (
    build_matchup_features,
    features_to_array,
    FEATURE_GROUPS,
    FEATURE_NAMES,
)
from .model.regression import (
    CLASS_LABELS,
    MultinomialLogisticModel,
    N_CLASSES,
    encode_outcome,
    format_coefficient_importance_report,
)
from .confidence.intervals import (
    compute_prediction_ci, effective_sample_size, fit_bootstrap_coefficients,
    format_prediction_table,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_profile(fighter_id: str) -> FighterProfile:
    """Minimal profile for fighters missing from the profiles CSV."""
    return FighterProfile(fighter_id=fighter_id, name=fighter_id)


def _migrate_model_config_elo_mc_fields(model: Optional[object]) -> None:
    """Pickle migration: ``ModelConfig`` gains Cauchy ELO MC γ hyperparameters."""
    if model is None:
        return
    defaults = {
        "elo_mc_n_draws": 200,
        "elo_mc_gamma_min": 5.0,
        "elo_mc_gamma_slope_sqrt_year": 25.0,
        "elo_mc_gamma_max": 120.0,
    }
    for key, val in defaults.items():
        if not hasattr(model, key):
            object.__setattr__(model, key, val)


def _migrate_model_config_lbfgs_fields(model: Optional[object]) -> None:
    """Pickle migration: ``ModelConfig`` gains L-BFGS-B knobs."""
    if model is None:
        return
    defaults = {
        "lbfgs_max_iter": 10_000,
        "lbfgs_ftol": 1e-12,
        "lbfgs_gtol": 1e-7,
    }
    for key, val in defaults.items():
        if not hasattr(model, key):
            object.__setattr__(model, key, val)


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class MMAPredictor:
    """
    End-to-end pre-fight MMA prediction pipeline.

    State after each stage:
        after load_data()        → self.fights, self.profiles populated
        after build_elo()        → self.elo_model ready
        after train_regression() → self.regression ready, training arrays stored
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.fights: List[FightRecord] = []
        self.profiles: Dict[str, FighterProfile] = {}
        self.elo_model: Optional[ELOModel] = None
        self.regression: Optional[MultinomialLogisticModel] = None
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._train_weights: Optional[np.ndarray] = None
        #: Structured summary of coefficient norms (from last ``train_regression`` with fit).
        self.training_regression_audit: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Stage 1: Data loading
    # ------------------------------------------------------------------

    def load_data(self, data_dir: Path) -> "MMAPredictor":
        """
        Load all CSV data sources from data_dir.

        Expected files (missing files are silently skipped):
            ufcstats_fights.csv (or legacy tier1_ufcstats.csv)
            tier2_bellator.csv  tier2_one.csv  tier2_pfl.csv  tier2_rizin.csv
            tier3_sherdog.csv
            fighter_profiles.csv
        """
        all_fights: List[FightRecord] = []

        for fname in ("ufcstats_fights.csv", "tier1_ufcstats.csv"):
            p = data_dir / fname
            if p.exists():
                print(f"  [data] loading {fname} ...", flush=True)
                chunk = load_ufcstats_fights(p)
                all_fights.extend(chunk)
                print(f"  [data]   +{len(chunk):,} fights (running total {len(all_fights):,})", flush=True)
                break
        else:
            print("  [data] no ufcstats_fights.csv or tier1_ufcstats.csv found", flush=True)

        for promo in ("bellator", "one", "pfl", "rizin"):
            p = data_dir / f"tier2_{promo}.csv"
            if p.exists():
                print(f"  [data] loading tier2_{promo}.csv ...", flush=True)
                chunk = load_major_promotion_fights(p, promo.upper())
                all_fights.extend(chunk)
                print(f"  [data]   +{len(chunk):,} fights (total {len(all_fights):,})", flush=True)

        p = data_dir / "tier3_sherdog.csv"
        if p.exists():
            print("  [data] loading tier3_sherdog.csv ...", flush=True)
            chunk = load_sherdog_fights(p)
            all_fights.extend(chunk)
            print(f"  [data]   +{len(chunk):,} fights (total {len(all_fights):,})", flush=True)

        p = data_dir / "fighter_profiles.csv"
        if p.exists():
            print("  [data] loading fighter_profiles.csv ...", flush=True)
            self.profiles = load_fighter_profiles(p)
            print(f"  [data]   {len(self.profiles):,} profiles", flush=True)
        else:
            print("  [data] no fighter_profiles.csv (profiles empty)", flush=True)

        print("  [data] sorting fights chronologically ...", flush=True)
        self.fights = sort_fights_chronologically(all_fights)
        return self

    def load_fights_direct(
        self,
        fights: List[FightRecord],
        profiles: Optional[Dict[str, FighterProfile]] = None,
    ) -> "MMAPredictor":
        """Load data directly from Python objects (testing / programmatic use)."""
        self.fights = sort_fights_chronologically(fights)
        if profiles:
            self.profiles = profiles
        return self

    # ------------------------------------------------------------------
    # Stage 2: ELO construction
    # ------------------------------------------------------------------

    def build_elo(
        self,
        *,
        elo_progress_every: int = 1000,
        record_trajectories: bool = False,
    ) -> "MMAPredictor":
        """
        Build ELO ratings from all available fight history.

        If ``record_trajectories`` is True, each bout appends a post-fight ELO point
        per fighter in that weight class as ``(date, elo, opponent_fighter_id)``;
        query via ``elo_model.get_trajectory(fid, wc)``.
        """
        self.elo_model = ELOModel(self.config.elo)
        self.elo_model.process_fights(
            self.fights,
            self.profiles if self.profiles else None,
            progress_every=elo_progress_every,
            record_trajectories=record_trajectories,
        )
        return self

    _ELO_CACHE_VERSION = 1

    @staticmethod
    def _elo_config_signature(elo: ELOConfig) -> dict:
        """JSON-serializable dict for cache validation (ELOConfig only)."""
        d = dataclasses.asdict(elo)
        # Stabilize tier_discount key order for comparison
        d["tier_discount"] = {int(k): float(v) for k, v in sorted(d["tier_discount"].items())}
        return d

    def try_load_elo_from_cache(self, path: Path) -> bool:
        """
        Load a serialized ELO model from *path* if it exists, matches
        ``len(self.fights)``, and matches current ``config.elo``.

        Returns True if the cache was applied; False if missing or stale.
        """
        path = Path(path)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if not isinstance(blob, dict) or blob.get("_elo_cache_v") != self._ELO_CACHE_VERSION:
            return False
        if blob.get("n_fights") != len(self.fights):
            return False
        cur_sig = self._elo_config_signature(self.config.elo)
        if blob.get("elo_config_sig") != cur_sig:
            return False
        self.elo_model = blob["elo_model"]
        return True

    def save_elo_cache(self, path: Path) -> None:
        """Write ELO state for ``try_load_elo_from_cache`` (after ``build_elo``)."""
        if self.elo_model is None:
            raise RuntimeError("build_elo() before save_elo_cache()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "_elo_cache_v": self._ELO_CACHE_VERSION,
                    "n_fights": len(self.fights),
                    "elo_config_sig": self._elo_config_signature(self.config.elo),
                    "elo_model": self.elo_model,
                },
                f,
            )

    # ------------------------------------------------------------------
    # Stage 3: Style axis query (per fighter, per date)
    # ------------------------------------------------------------------

    def _fighter_fights(self, fighter_id: str, wc: WeightClass) -> List[FightRecord]:
        return [
            f for f in self.fights
            if (f.fighter_a_id == fighter_id or f.fighter_b_id == fighter_id)
            and f.weight_class == wc
        ]

    def _n_prior_bouts_in_wc(
        self, fighter_id: str, wc: WeightClass, fight_date: date
    ) -> int:
        """
        Count bouts in ``wc`` strictly before ``fight_date`` (any tier in loaded data).

        Used to detect a **weight-class debut** for Cauchy CIs: fewer than one
        such bout means no prior in-division history in our corpus before this card.
        """
        return sum(
            1
            for f in self._fighter_fights(fighter_id, wc)
            if f.fight_date < fight_date
        )

    def _force_cauchy_weight_class_debut(
        self,
        fighter_a_id: str,
        fighter_b_id: str,
        wc: WeightClass,
        fight_date: date,
    ) -> bool:
        """True if either corner has zero prior bouts in ``wc`` before ``fight_date``."""
        return (
            self._n_prior_bouts_in_wc(fighter_a_id, wc, fight_date) < 1
            or self._n_prior_bouts_in_wc(fighter_b_id, wc, fight_date) < 1
        )

    def get_style_axes(
        self,
        fighter_id: str,
        wc: WeightClass,
        as_of_date: date,
    ) -> StyleAxes:
        """
        Return style axes for fighter_id in wc strictly before as_of_date.
        Cold-start pedigree blending is applied automatically.
        """
        if self.elo_model is None:
            raise RuntimeError("Call build_elo() before get_style_axes().")

        history = self._fighter_fights(fighter_id, wc)
        axes = compute_style_axes(
            fighter_id=fighter_id,
            wc=wc,
            fight_history=history,
            elo_model=self.elo_model,
            reference_date=as_of_date,
            config=self.config.features,
        )
        profile = self.profiles.get(fighter_id)
        if profile is not None:
            axes = apply_cold_start_prior(axes, profile, self.config.features)
        return axes

    # ------------------------------------------------------------------
    # Stages 4 + 5: Training data construction and regression fit
    # ------------------------------------------------------------------

    def train_regression(
        self,
        *,
        matrix_progress_every: int = 500,
        fit_model: bool = True,
    ) -> "MMAPredictor":
        """
        Build the training feature matrix from Tier 1 post-era fights and fit
        the multinomial logistic regression.

        Sample weights are recency-based: fights closer to today are weighted
        higher, reflecting that more recent data comes from a more similar
        distribution to future fights.

        If *fit_model* is False, only the training matrix and weights are built;
        no L-BFGS or bootstrap. Used by pilot / diagnostics to reuse *X* / *y*.
        """
        if self.elo_model is None:
            raise RuntimeError("Call build_elo() before train_regression().")

        training_fights = filter_tier1_post_era(
            self.fights, self.config.master_start_year
        )
        print(
            f"  [train] Tier-1 candidate fights (year >= {self.config.master_start_year}): "
            f"{len(training_fights):,}",
            flush=True,
        )
        hsd = self.config.holdout_start_date
        if hsd is not None:
            n_pre = len(training_fights)
            training_fights = [f for f in training_fights if f.fight_date < hsd]
            print(
                f"  [train] holdout: excluding fight_date >= {hsd} "
                f"({n_pre - len(training_fights):,} excluded; {len(training_fights):,} train rows)",
                flush=True,
            )
        print(
            "  [train] building feature rows (style axes + ELO snapshot per fight date) ...",
            flush=True,
        )

        n_train_candidates = len(training_fights)
        X_rows, y_rows, w_rows = [], [], []
        today = date.today()

        for fi, fight in enumerate(training_fights):
            a_id, b_id = fight.fighter_a_id, fight.fighter_b_id
            wc, fdate = fight.weight_class, fight.fight_date

            label = encode_outcome(fight, a_id)
            if label is None:
                continue

            axes_a = self.get_style_axes(a_id, wc, fdate)
            axes_b = self.get_style_axes(b_id, wc, fdate)
            elo_a = self.elo_model.get_state(a_id, wc, fdate)
            elo_b = self.elo_model.get_state(b_id, wc, fdate)
            prof_a = self.profiles.get(a_id, _empty_profile(a_id))
            prof_b = self.profiles.get(b_id, _empty_profile(b_id))

            features = build_matchup_features(elo_a, elo_b, axes_a, axes_b, prof_a, prof_b, fdate)
            x = features_to_array(features)

            X_rows.append(x)
            y_rows.append(label)
            days_old = max(0, (today - fdate).days)
            w_rows.append(1.0 / (1.0 + days_old / 365.0))

            n_done = len(X_rows)
            if matrix_progress_every > 0 and (
                n_done == 1 or n_done % matrix_progress_every == 0
            ):
                print(
                    f"  [train] {n_done:,} regression rows "
                    f"(scanned {fi + 1:,} / {n_train_candidates:,} train-candidate fights)",
                    flush=True,
                )

        if not X_rows:
            raise RuntimeError(
                "No valid training fights found. Check data_dir, Config.master_start_year, "
                "and holdout_start_date (holdout may have removed all rows)."
            )

        self._X_train = np.array(X_rows)
        self._y_train = np.array(y_rows, dtype=int)
        self._train_weights = np.array(w_rows)

        if not fit_model:
            print(
                f"  [train] matrix shape {self._X_train.shape}  (fit_model=False: skip L-BFGS, skip bootstrap)\n",
                flush=True,
            )
            return self

        print(
            f"  [train] matrix shape {self._X_train.shape}, fitting multinomial regression ...",
            flush=True,
        )
        self.regression = MultinomialLogisticModel(
            n_features=self._X_train.shape[1],
            delta=self.config.model.huber_delta,
            l2_lambda=self.config.model.l2_lambda,
        )
        m = self.config.model
        self.regression.fit(
            self._X_train,
            self._y_train,
            verbose=True,
            max_iter=m.lbfgs_max_iter,
            ftol=m.lbfgs_ftol,
            gtol=m.lbfgs_gtol,
        )

        if self.regression.W is not None:
            rep_text, audit = format_coefficient_importance_report(
                self.regression.W,
                list(FEATURE_NAMES),
                FEATURE_GROUPS,
                self._X_train,
            )
            print(rep_text, flush=True)
            self.training_regression_audit = audit

        eff_n = effective_sample_size(self._train_weights)
        cauchy_th = self.config.model.cauchy_fallback_threshold
        n_bootstrap = self.config.model.n_bootstrap
        self._bootstrap_W = None

        if eff_n < cauchy_th:
            print(
                f"  [train] after point fit: no weighted bootstrap  "
                f"(ESS {eff_n:.1f} < cauchy_fallback_threshold {cauchy_th:g}; "
                "CIs at predict use Cauchy ELO / point path).",
                flush=True,
            )
        elif n_bootstrap <= 0:
            print(
                f"  [train] after point fit: no bootstrap resamples  (n_bootstrap={n_bootstrap}).",
                flush=True,
            )
        else:
            print(
                f"  [train] after point fit: running {n_bootstrap} bootstrap resamples  "
                f"(ESS {eff_n:.1f} >= {cauchy_th:g}, store coefficient draws for CIs) ...",
                flush=True,
            )
            W_stack, n_valid = fit_bootstrap_coefficients(
                self._X_train,
                self._y_train,
                self._train_weights,
                self.config.model,
                progress_every=40,
            )
            if n_valid >= 10:
                self._bootstrap_W = W_stack
                print(
                    f"  [train] stored {n_valid} bootstrap coefficient matrices "
                    f"(fast CIs at predict; no per-fight refit).",
                    flush=True,
                )
            else:
                print(
                    f"  [train] only {n_valid} valid bootstrap fits (< 10); "
                    "prediction CIs will use Cauchy or legacy bootstrap.",
                    flush=True,
                )

        return self

    # ------------------------------------------------------------------
    # Stage 6: Prediction with confidence intervals
    # ------------------------------------------------------------------

    def predict_proba_point_only(
        self,
        fighter_a_id: str,
        fighter_b_id: str,
        wc: WeightClass,
        fight_date: date,
    ) -> np.ndarray:
        """
        Return the (6,) softmax probability vector for fighter A vs B — no bootstrap / Cauchy CIs.

        Use for fast checks (symmetry, unit tests). ``predict()`` is preferred interactively.
        """
        if self.regression is None:
            raise RuntimeError("Call train_regression() before predict_proba_point_only().")

        axes_a = self.get_style_axes(fighter_a_id, wc, fight_date)
        axes_b = self.get_style_axes(fighter_b_id, wc, fight_date)
        elo_a = self.elo_model.get_state(fighter_a_id, wc, fight_date)
        elo_b = self.elo_model.get_state(fighter_b_id, wc, fight_date)
        prof_a = self.profiles.get(fighter_a_id, _empty_profile(fighter_a_id))
        prof_b = self.profiles.get(fighter_b_id, _empty_profile(fighter_b_id))

        features = build_matchup_features(elo_a, elo_b, axes_a, axes_b, prof_a, prof_b, fight_date)
        x = features_to_array(features)
        return self.regression.predict_proba(x)

    def predict(
        self,
        fighter_a_id: str,
        fighter_b_id: str,
        wc: WeightClass,
        fight_date: date,
        verbose: bool = True,
    ) -> PredictionResult:
        """
        Produce a calibrated 6-class prediction with confidence intervals.

        verbose=True prints the formatted output table to stdout.
        """
        if self.regression is None:
            raise RuntimeError("Call train_regression() before predict().")

        axes_a = self.get_style_axes(fighter_a_id, wc, fight_date)
        axes_b = self.get_style_axes(fighter_b_id, wc, fight_date)
        elo_a = self.elo_model.get_state(fighter_a_id, wc, fight_date)
        elo_b = self.elo_model.get_state(fighter_b_id, wc, fight_date)
        prof_a = self.profiles.get(fighter_a_id, _empty_profile(fighter_a_id))
        prof_b = self.profiles.get(fighter_b_id, _empty_profile(fighter_b_id))

        features = build_matchup_features(elo_a, elo_b, axes_a, axes_b, prof_a, prof_b, fight_date)
        x = features_to_array(features)
        point_est = self.regression.predict_proba(x)

        eff_n = effective_sample_size(self._train_weights)
        bootstrap_W = getattr(self, "_bootstrap_W", None)
        force_cauchy_wc_debut = self._force_cauchy_weight_class_debut(
            fighter_a_id, fighter_b_id, wc, fight_date
        )
        use_stored = (
            bootstrap_W is not None
            and bootstrap_W.size > 0
            and bootstrap_W.shape[0] >= 10
        )
        bootstrap_progress_every = 0 if use_stored else (40 if verbose else 0)
        use_elo_mc = (
            self.config.model.elo_mc_n_draws > 0
            and not force_cauchy_wc_debut
        )
        days_idle_a = self.elo_model.days_since_last_fight_global(fighter_a_id, fight_date)
        days_idle_b = self.elo_model.days_since_last_fight_global(fighter_b_id, fight_date)
        gamma_a = self.config.model.elo_mc_gamma_for_days_idle(days_idle_a)
        gamma_b = self.config.model.elo_mc_gamma_for_days_idle(days_idle_b)
        W_point = self.regression.W

        if verbose:
            if force_cauchy_wc_debut:
                print(
                    "  [predict] Cauchy CIs (weight-class debut; bootstrap skipped) ...",
                    flush=True,
                )
            elif use_stored and use_elo_mc:
                print(
                    f"  [predict] CIs: {bootstrap_W.shape[0]} bootstrap W × "
                    f"{self.config.model.elo_mc_n_draws} Cauchy ELO draws (γ) ...",
                    flush=True,
                )
            elif use_stored:
                print(
                    f"  [predict] CIs from {bootstrap_W.shape[0]} stored bootstrap draws (no refit) ...",
                    flush=True,
                )
            elif (
                eff_n >= self.config.model.cauchy_fallback_threshold
                and self._X_train is not None
            ):
                prog = (
                    f", status every {bootstrap_progress_every} resamples"
                    if bootstrap_progress_every
                    else ""
                )
                _elo_suffix = " + Cauchy ELO (γ) per resample" if use_elo_mc else ""
                print(
                    f"  [predict] bootstrap CIs ({self.config.model.n_bootstrap} resamples{prog}) "
                    f"(legacy path; retrain to cache draws){_elo_suffix} ...",
                    flush=True,
                )
            elif (
                use_elo_mc
                and eff_n < self.config.model.cauchy_fallback_threshold
            ):
                print(
                    f"  [predict] CIs from {self.config.model.elo_mc_n_draws} Cauchy ELO draws "
                    "(point coefficients; sparse ESS) ...",
                    flush=True,
                )
        elo_ga = gamma_a if use_elo_mc else None
        elo_gb = gamma_b if use_elo_mc else None
        lower, upper, ci_method = compute_prediction_ci(
            x=x,
            point_estimate=point_est,
            X_train=self._X_train,
            y_train=self._y_train,
            train_weights=self._train_weights,
            effective_n=eff_n,
            config=self.config.model,
            bootstrap_W=bootstrap_W,
            bootstrap_progress_every=bootstrap_progress_every,
            force_cauchy_wc_debut=force_cauchy_wc_debut,
            elo_mc_gamma_a=elo_ga,
            elo_mc_gamma_b=elo_gb,
            W_point=W_point,
        )

        if verbose:
            print(format_prediction_table(
                point_est, lower, upper, ci_method, eff_n,
                pct_post_era=1.0,
                master_start_year=self.config.master_start_year,
                alpha=self.config.model.ci_alpha,
            ))

        return PredictionResult(
            fighter_a_id=fighter_a_id,
            fighter_b_id=fighter_b_id,
            weight_class=wc,
            fight_date=fight_date,
            p_win_ko_tko=float(point_est[0]),
            p_win_submission=float(point_est[1]),
            p_win_decision=float(point_est[2]),
            p_lose_decision=float(point_est[3]),
            p_lose_ko_tko=float(point_est[4]),
            p_lose_submission=float(point_est[5]),
            ci_win_ko_tko=(float(lower[0]), float(upper[0])),
            ci_win_submission=(float(lower[1]), float(upper[1])),
            ci_win_decision=(float(lower[2]), float(upper[2])),
            ci_lose_decision=(float(lower[3]), float(upper[3])),
            ci_lose_ko_tko=(float(lower[4]), float(upper[4])),
            ci_lose_submission=(float(lower[5]), float(upper[5])),
            ci_method=ci_method,
            effective_n=eff_n,
            pct_post_era=1.0,
            features=features,
        )

    def explain(
        self,
        fighter_a_id: str,
        fighter_b_id: str,
        wc: WeightClass,
        fight_date: date,
    ) -> None:
        """
        Print the exact additive decomposition of the prediction log-odds.

        Shows the top-5 contributing features per class — no approximation,
        since the model is fully additive.
        """
        if self.regression is None:
            raise RuntimeError("Call train_regression() before explain().")

        axes_a = self.get_style_axes(fighter_a_id, wc, fight_date)
        axes_b = self.get_style_axes(fighter_b_id, wc, fight_date)
        elo_a = self.elo_model.get_state(fighter_a_id, wc, fight_date)
        elo_b = self.elo_model.get_state(fighter_b_id, wc, fight_date)
        prof_a = self.profiles.get(fighter_a_id, _empty_profile(fighter_a_id))
        prof_b = self.profiles.get(fighter_b_id, _empty_profile(fighter_b_id))

        features = build_matchup_features(elo_a, elo_b, axes_a, axes_b, prof_a, prof_b, fight_date)
        x = features_to_array(features)
        probs, contributions = self.regression.predict_with_decomposition(x, FEATURE_NAMES)

        print(f"\n=== Prediction Decomposition: {fighter_a_id} vs {fighter_b_id} ===")
        print(f"Weight class: {wc.value}  |  Fight date: {fight_date}\n")
        for k, label in enumerate(CLASS_LABELS):
            print(f"  {label}  (p={probs[k]:.3f})")
            ranked = sorted(
                ((fname, contributions[fname][label]) for fname in FEATURE_NAMES),
                key=lambda t: abs(t[1]),
                reverse=True,
            )
            for fname, val in ranked[:5]:
                print(f"    {fname:<35s} {val:+.4f}")
        print()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise the fitted predictor to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        """Pickle migration: older ``model.pkl`` files lack ``_bootstrap_W`` / ``master_start_year``."""
        self.__dict__.update(state)
        if "_bootstrap_W" not in self.__dict__:
            self._bootstrap_W = None
        if "training_regression_audit" not in self.__dict__:
            self.training_regression_audit = None
        cfg = self.__dict__.get("config")
        if cfg is not None:
            if not hasattr(cfg, "master_start_year"):
                feat = getattr(cfg, "features", None)
                legacy = getattr(feat, "era_cutoff_year", None) if feat is not None else None
                object.__setattr__(
                    cfg,
                    "master_start_year",
                    int(legacy) if legacy is not None else 2005,
                )
            if not hasattr(cfg, "holdout_start_date"):
                from .config import DEFAULT_HOLDOUT_START_DATE

                object.__setattr__(cfg, "holdout_start_date", DEFAULT_HOLDOUT_START_DATE)
            _migrate_model_config_elo_mc_fields(getattr(cfg, "model", None))
            _migrate_model_config_lbfgs_fields(getattr(cfg, "model", None))

    @classmethod
    def load(cls, path: Path) -> "MMAPredictor":
        """Load a previously saved predictor from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)
