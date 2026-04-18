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
import pickle
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import Config
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
    build_matchup_features, features_to_array, FEATURE_NAMES,
)
from .model.regression import (
    CLASS_LABELS, MultinomialLogisticModel, N_CLASSES, encode_outcome,
)
from .confidence.intervals import (
    compute_prediction_ci, effective_sample_size, format_prediction_table,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_profile(fighter_id: str) -> FighterProfile:
    """Minimal profile for fighters missing from the profiles CSV."""
    return FighterProfile(fighter_id=fighter_id, name=fighter_id)


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
                all_fights.extend(load_ufcstats_fights(p))
                break

        for promo in ("bellator", "one", "pfl", "rizin"):
            p = data_dir / f"tier2_{promo}.csv"
            if p.exists():
                all_fights.extend(load_major_promotion_fights(p, promo.upper()))

        p = data_dir / "tier3_sherdog.csv"
        if p.exists():
            all_fights.extend(load_sherdog_fights(p))

        p = data_dir / "fighter_profiles.csv"
        if p.exists():
            self.profiles = load_fighter_profiles(p)

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

    def build_elo(self) -> "MMAPredictor":
        """Build ELO ratings from all available fight history."""
        self.elo_model = ELOModel(self.config.elo)
        self.elo_model.process_fights(self.fights, self.profiles if self.profiles else None)
        return self

    # ------------------------------------------------------------------
    # Stage 3: Style axis query (per fighter, per date)
    # ------------------------------------------------------------------

    def _fighter_fights(self, fighter_id: str, wc: WeightClass) -> List[FightRecord]:
        return [
            f for f in self.fights
            if (f.fighter_a_id == fighter_id or f.fighter_b_id == fighter_id)
            and f.weight_class == wc
        ]

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

    def train_regression(self) -> "MMAPredictor":
        """
        Build the training feature matrix from Tier 1 post-era fights and fit
        the multinomial logistic regression.

        Sample weights are recency-based: fights closer to today are weighted
        higher, reflecting that more recent data comes from a more similar
        distribution to future fights.
        """
        if self.elo_model is None:
            raise RuntimeError("Call build_elo() before train_regression().")

        training_fights = filter_tier1_post_era(
            self.fights, self.config.features.era_cutoff_year
        )

        X_rows, y_rows, w_rows = [], [], []
        today = date.today()

        for fight in training_fights:
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

        if not X_rows:
            raise RuntimeError("No valid training fights found. Check data_dir and era_cutoff_year.")

        self._X_train = np.array(X_rows)
        self._y_train = np.array(y_rows, dtype=int)
        self._train_weights = np.array(w_rows)

        self.regression = MultinomialLogisticModel(
            n_features=self._X_train.shape[1],
            delta=self.config.model.huber_delta,
            l2_lambda=self.config.model.l2_lambda,
        )
        self.regression.fit(self._X_train, self._y_train)
        return self

    # ------------------------------------------------------------------
    # Stage 6: Prediction with confidence intervals
    # ------------------------------------------------------------------

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
        lower, upper, ci_method = compute_prediction_ci(
            x=x,
            point_estimate=point_est,
            X_train=self._X_train,
            y_train=self._y_train,
            train_weights=self._train_weights,
            effective_n=eff_n,
            config=self.config.model,
        )

        if verbose:
            print(format_prediction_table(
                point_est, lower, upper, ci_method, eff_n,
                pct_post_era=1.0,
                era_cutoff_year=self.config.features.era_cutoff_year,
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

    @classmethod
    def load(cls, path: Path) -> "MMAPredictor":
        """Load a previously saved predictor from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)
