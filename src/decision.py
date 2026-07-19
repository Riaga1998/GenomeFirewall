"""
Module 03 — The Decision Report (core logic)
============================================
Combines everything into one honest verdict per drug. The order matters:

  1. GATE      deterministic biology (target absent / intrinsic marker) -> can force FAIL
  2. OOD       genome unlike training set -> NO-CALL (novelty, not weak evidence)
  3. MODEL     calibrated p(resistant)
  4. CONFLICT  known R gene present but model says susceptible -> NO-CALL (disagreement)
  5. BAND      weak/ambiguous calibrated probability -> NO-CALL (weak evidence)

Every verdict carries:
  label         : "likely to fail" | "likely to work" | "no-call"
  confidence    : calibrated probability backing the call
  evidence_tier : "known_determinant" | "statistical_only" | "no_signal"
  reason        : plain-language why (incl. why it abstained)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

from target_gate import apply_gate, GateResult


LIKELY_FAIL = "likely to fail"
LIKELY_WORK = "likely to work"
NO_CALL = "no-call"


@dataclass
class Verdict:
    antibiotic: str
    label: str
    confidence: float
    evidence_tier: str
    reason: str
    supporting_features: list[str]

    def to_dict(self):
        return asdict(self)


def _evidence_tier(present_curated: list[str], model_leans_on: list[str]) -> str:
    if present_curated:
        return "known_determinant"          # tier 1: curated AMR hit present
    if model_leans_on:
        return "statistical_only"           # tier 2: model weight, no curated hit
    return "no_signal"


def decide(
    antibiotic: str,
    p_resistant: float | None,
    present_features: set[str],
    curated_cols: set[str],
    gate: dict,
    containment_to_train: float,
    cfg: dict,
    top_model_features: list[str] | None = None,
) -> Verdict:
    dec = cfg["decision"]
    top_model_features = top_model_features or []

    # curated AMR determinants actually detected in this genome
    present_curated = sorted(present_features & curated_cols)

    # ---- 1. Deterministic gate -------------------------------------------
    g: GateResult = apply_gate(antibiotic, present_features, gate)
    if g.forced_label == "R":
        return Verdict(antibiotic, LIKELY_FAIL, 1.0, "known_determinant",
                       g.reason, present_curated)

    # ---- 2. OOD novelty no-call ------------------------------------------
    if containment_to_train < cfg["ood"]["min_containment_to_train"]:
        return Verdict(
            antibiotic, NO_CALL, float(p_resistant or 0.0), "no_signal",
            (f"Genome is unlike the training data "
             f"(containment {containment_to_train:.2f} < "
             f"{cfg['ood']['min_containment_to_train']}). Abstaining."),
            present_curated,
        )

    if p_resistant is None:
        return Verdict(antibiotic, NO_CALL, 0.0, "no_signal",
                       "No trained model available for this drug.", present_curated)

    tier = _evidence_tier(present_curated, top_model_features)

    # ---- 4. Conflict: curated R gene present but model leans susceptible --
    if present_curated and p_resistant < dec["susceptible_threshold"] and \
            p_resistant <= (1 - dec["resistant_threshold"]):
        return Verdict(
            antibiotic, NO_CALL, float(p_resistant), "known_determinant",
            (f"Conflicting evidence: known determinant(s) {present_curated} "
             f"detected, but model estimates low resistance "
             f"({p_resistant:.2f}). Abstaining."),
            present_curated,
        )

    # ---- 5. Confidence band ----------------------------------------------
    if p_resistant >= dec["resistant_threshold"]:
        reason = (f"Resistance likely (calibrated p={p_resistant:.2f})."
                  + (f" Detected: {present_curated}." if present_curated else
                     " Driven by statistical association, not a curated determinant."))
        return Verdict(antibiotic, LIKELY_FAIL, float(p_resistant), tier,
                       reason, present_curated)

    if p_resistant <= (1 - dec["susceptible_threshold"]) and g.gate_open:
        return Verdict(
            antibiotic, LIKELY_WORK, float(1 - p_resistant), tier,
            (f"Susceptible: target present and no resistance signal "
             f"(calibrated p_resistant={p_resistant:.2f})."),
            present_curated,
        )

    # weak / ambiguous
    return Verdict(
        antibiotic, NO_CALL, float(p_resistant), tier,
        (f"Weak or ambiguous evidence (calibrated p_resistant={p_resistant:.2f}); "
         f"between decision thresholds. Abstaining."),
        present_curated,
    )
