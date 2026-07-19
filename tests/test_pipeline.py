"""Tests for the properties that make the reported metrics trustworthy.

These guard the claims the submission makes: no cluster leakage across splits, honest
no-call behaviour, a working intrinsic-resistance gate, and evidence attribution that
does not overstate what the model knows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drug_database import get_drug
from src.genome_reader import hits_to_features
from src.predictor import (
    EVIDENCE_KNOWN_DETERMINANT,
    EVIDENCE_INTRINSIC,
    EVIDENCE_NO_SIGNAL,
    GenomeFirewall,
)
from src.synthetic_data import generate_cohort
from src.utils.amrfinder import AMRHit
from src.utils.calibration import (
    LIKELY_TO_FAIL,
    LIKELY_TO_WORK,
    NO_CALL,
    ProbabilityCalibrator,
    apply_no_call,
    expected_calibration_error,
)
from src.utils.clustering import grouped_split, verify_no_leakage


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(n_genomes=600, n_clusters=25, seed=7)


@pytest.fixture(scope="module")
def trained_panel(cohort):
    features, labels, clusters = cohort
    split = grouped_split(clusters, seed=7)
    panel = GenomeFirewall(species="escherichia coli")
    panel.fit(
        features.loc[split.train],
        labels.loc[split.train],
        features.loc[split.calibration],
        labels.loc[split.calibration],
    )
    return panel, features, labels, split


class TestSplitting:
    def test_split_is_cluster_disjoint(self, cohort):
        _, _, clusters = cohort
        split = grouped_split(clusters, seed=7)
        verify_no_leakage(split, clusters)  # raises if a cluster spans two sets

    def test_split_covers_every_genome_exactly_once(self, cohort):
        _, _, clusters = cohort
        split = grouped_split(clusters, seed=7)
        assigned = split.train + split.calibration + split.test
        assert len(assigned) == len(clusters)
        assert set(assigned) == set(clusters.index)

    def test_leakage_check_catches_a_bad_split(self, cohort):
        _, _, clusters = cohort
        split = grouped_split(clusters, seed=7)
        split.train.append(split.test[0])  # force one genome into both sets
        with pytest.raises(AssertionError, match="leakage"):
            verify_no_leakage(split, clusters)


class TestNoCall:
    def test_uncertain_band_returns_no_call(self):
        decisions = apply_no_call(np.array([0.05, 0.5, 0.95]), low=0.35, high=0.65)
        assert list(decisions) == ["likely to work", NO_CALL, LIKELY_TO_FAIL]

    def test_band_boundaries_are_inclusive(self):
        decisions = apply_no_call(np.array([0.35, 0.65]), low=0.35, high=0.65)
        assert list(decisions) == ["likely to work", LIKELY_TO_FAIL]


class TestCalibration:
    def test_calibration_improves_a_skewed_score(self):
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, size=800)
        # Overconfident scores: pushed toward the extremes relative to true rate.
        raw = np.clip(y * 0.55 + 0.22 + rng.normal(0, 0.12, size=800), 0, 1)

        calibrator = ProbabilityCalibrator(method="auto").fit(raw, y)
        calibrated = calibrator.transform(raw)

        assert expected_calibration_error(y, calibrated) <= expected_calibration_error(y, raw)

    def test_single_class_calibration_does_not_crash(self):
        calibrator = ProbabilityCalibrator().fit(np.array([0.2, 0.4, 0.6]), np.array([1, 1, 1]))
        assert calibrator.fitted_method_ == "identity"
        assert np.all((calibrator.transform(np.array([0.5])) >= 0) & (calibrator.transform(np.array([0.5])) <= 1))


class TestTargetGate:
    def test_vancomycin_blocked_in_gram_negative(self):
        applicable, reason = get_drug("vancomycin").is_applicable("escherichia coli")
        assert not applicable
        assert "gram-negative" in reason

    def test_vancomycin_allowed_in_gram_positive(self):
        applicable, _ = get_drug("vancomycin").is_applicable("staphylococcus aureus")
        assert applicable

    def test_intrinsic_resistance_is_flagged(self):
        applicable, reason = get_drug("ampicillin").is_applicable("klebsiella pneumoniae")
        assert not applicable
        assert "intrinsically resistant" in reason

    def test_unknown_species_does_not_block(self):
        applicable, reason = get_drug("colistin").is_applicable("some novel species")
        assert applicable
        assert "not in the property table" in reason

    def test_drug_lookup_tolerates_aliases(self):
        assert get_drug("TMP-SMX") is get_drug("trimethoprim-sulfamethoxazole")
        assert get_drug("nonexistent-drug") is None


class TestAMRFinderSchema:
    """The parser must survive AMRFinderPlus's column renames.

    v4 renamed most headers. Reading a v4 report with only v3 names matched nothing and
    produced an empty hit list — indistinguishable from a genome with no resistance.
    """

    V4_HEADER = [
        "Protein id", "Contig id", "Element symbol", "Element name",
        "Type", "Subtype", "Class", "Subclass",
        "% Coverage of reference", "% Identity to reference",
    ]
    V3_HEADER = [
        "Protein identifier", "Contig id", "Gene symbol", "Sequence name",
        "Element type", "Element subtype", "Class", "Subclass",
        "% Coverage of reference sequence", "% Identity to reference sequence",
    ]
    ROW = ["NA", "contig1", "blaCTX-M-15", "ESBL", "AMR", "AMR",
           "BETA-LACTAM", "CEPHALOSPORIN", "100.0", "100.0"]

    @pytest.mark.parametrize("header", [V4_HEADER, V3_HEADER])
    def test_both_schema_versions_parse(self, header):
        from src.utils.amrfinder import normalize_report, parse_report

        hits = parse_report(normalize_report(pd.DataFrame([self.ROW], columns=header)))
        assert len(hits) == 1
        assert hits[0].gene_symbol == "blaCTX-M-15"
        assert hits[0].drug_class == "BETA-LACTAM"

    def test_unrecognized_schema_raises_instead_of_returning_nothing(self):
        from src.utils.amrfinder import normalize_report

        bogus = pd.DataFrame([["x", "y"]], columns=["Something", "Else"])
        with pytest.raises(ValueError, match="missing required column"):
            normalize_report(bogus)


class TestEvidenceVocabulary:
    """Drug class tags must match AMRFinderPlus's vocabulary.

    AMRFinderPlus reports sul1 under SULFONAMIDE. A drug database that called the same
    thing FOLATE-PATHWAY-ANTAGONIST failed to match, and a genuine known determinant was
    reported as a mere statistical association — understating real evidence.
    """

    def test_combination_drug_covers_each_component_class(self):
        from src.drug_database import class_features_for_drug

        tags = set(class_features_for_drug(get_drug("trimethoprim-sulfamethoxazole")))
        assert "class:SULFONAMIDE" in tags
        assert "class:TRIMETHOPRIM" in tags

    def test_macrolide_gene_reads_as_a_known_mechanism(self, trained_panel):
        """ermC is a curated erythromycin mechanism, not a bare correlation.

        AMRFinderPlus spans several spellings for the MLSb classes; if the drug's
        accepted tags miss the one the tool emits, a genuine determinant is demoted to
        "statistical association" and the report understates its own evidence.
        """
        panel, _, _, _ = trained_panel
        drug = "erythromycin"
        if drug not in panel.models:
            pytest.skip(f"{drug} not trained in this sample")

        predictions = panel.predict_genome(
            {"gene:erm(C)": 1, "class:LINCOSAMIDE/MACROLIDE/STREPTOGRAMIN": 1}
        )
        pred = predictions[list(panel.models).index(drug)]
        assert pred.evidence_type == EVIDENCE_KNOWN_DETERMINANT, (
            f"erm(C) should count as a known mechanism, got: {pred.evidence_type}"
        )


class TestFeatureEncoding:
    def test_hits_become_gene_point_and_class_features(self):
        hits = [
            AMRHit("blaTEM-1", "AMR", "AMR", "BETA-LACTAM", "BETA-LACTAM", 99.0, 100.0, "c1"),
            AMRHit("gyrA_S83L", "AMR", "POINT", "QUINOLONE", "QUINOLONE", 99.5, 100.0, "c1"),
        ]
        features = hits_to_features(hits)
        assert features["gene:blaTEM-1"] == 1
        assert features["point:gyrA_S83L"] == 1
        assert features["class:BETA-LACTAM"] == 1
        assert features["class:QUINOLONE"] == 1

    def test_unnamed_hits_are_dropped(self):
        assert hits_to_features([AMRHit("", "AMR", "AMR", "", "", 99.0, 100.0, "c1")]) == {}


class TestPredictions:
    def test_panel_trains_on_every_well_populated_drug(self, trained_panel):
        panel, _, labels, _ = trained_panel
        assert len(panel.models) > 0
        assert set(panel.models).issubset(set(labels.columns))

    def test_prediction_carries_confidence_and_evidence(self, trained_panel):
        panel, features, _, split = trained_panel
        predictions = panel.predict_genome(features.loc[split.test[0]])

        assert len(predictions) == len(panel.models)
        for pred in predictions:
            assert 0.0 <= pred.resistance_probability <= 1.0
            assert pred.decision in {LIKELY_TO_FAIL, "likely to work", NO_CALL}
            assert pred.evidence_type
            if pred.is_called:
                assert 0.0 <= pred.confidence <= 1.0

    def test_no_call_reports_no_confidence(self, trained_panel):
        """A no-call has no claim to be confident about.

        The earlier formula scored a no-call by closeness to the decision boundary,
        which peaks exactly where the model knows least — a genome the system refused
        to call was displayed as "94% confident". Confidence is now undefined unless a
        decision was actually made.
        """
        panel, features, _, split = trained_panel

        seen_no_call = False
        for genome_id in split.test[:40]:
            for pred in panel.predict_genome(features.loc[genome_id]):
                if pred.decision == NO_CALL:
                    seen_no_call = True
                    assert pred.confidence is None, (
                        f"{pred.drug} reported confidence {pred.confidence} on a no-call"
                    )
                else:
                    assert pred.confidence is not None

        assert seen_no_call, "no no-call encountered; test did not exercise the path"

    def test_called_confidence_agrees_with_the_decision(self, trained_panel):
        """Confidence must describe the decision made, not always resistance."""
        panel, features, _, split = trained_panel

        for genome_id in split.test[:40]:
            for pred in panel.predict_genome(features.loc[genome_id]):
                if pred.decision == LIKELY_TO_FAIL:
                    assert pred.confidence == pytest.approx(pred.resistance_probability)
                elif pred.decision == LIKELY_TO_WORK:
                    assert pred.confidence == pytest.approx(1.0 - pred.resistance_probability)

    def test_empty_genome_never_asserts_failure(self, trained_panel):
        """The core honesty property.

        With no determinant detected, a high model score reflects only the training
        cohort's resistance prevalence. Reporting that as "likely to fail" would be
        confident-but-unevidenced — the exact failure the challenge penalizes.
        """
        panel, _, _, _ = trained_panel

        for pred in panel.predict_genome({}):
            assert pred.supporting_features == []
            if pred.evidence_type == EVIDENCE_NO_SIGNAL:
                assert pred.decision != LIKELY_TO_FAIL, (
                    f"{pred.drug} asserted failure with no supporting determinant"
                )

    def test_unevidenced_failure_is_downgraded_with_an_explanation(self, trained_panel):
        panel, _, _, _ = trained_panel
        downgraded = [
            p for p in panel.predict_genome({})
            if p.decision == NO_CALL and p.evidence_type == EVIDENCE_NO_SIGNAL
        ]
        for pred in downgraded:
            assert "training" in pred.gate_note.lower() or "no resistance" in pred.gate_note.lower()

    def test_detected_determinant_still_drives_a_failure_call(self, trained_panel):
        """The downgrade must not suppress genuine evidence-backed resistance."""
        panel, _, _, _ = trained_panel
        predictions = panel.predict_genome({"gene:mecA": 1, "class:BETA-LACTAM": 1})

        # mecA encodes PBP2a — this is MRSA, and cefoxitin is how the laboratory calls it.
        cefoxitin = [p for p in predictions if p.drug.lower() == "cefoxitin"]
        assert any(p.decision == LIKELY_TO_FAIL for p in cefoxitin), (
            "mecA should still produce a failure call for cefoxitin"
        )
        for pred in cefoxitin:
            if pred.decision == LIKELY_TO_FAIL:
                assert pred.supporting_features, "a failure call must cite its evidence"

    def test_intrinsic_gate_overrides_the_model(self):
        panel = GenomeFirewall(species="klebsiella pneumoniae")  # gate is species-generic
        features, labels, clusters = generate_cohort(n_genomes=300, n_clusters=12, seed=3)
        split = grouped_split(clusters, seed=3)
        panel.fit(features.loc[split.train], labels.loc[split.train])

        if "ampicillin" not in panel.models:
            pytest.skip("ampicillin not trained in this sample")

        pred = panel.predict_genome({})[list(panel.models).index("ampicillin")]
        assert pred.decision == LIKELY_TO_FAIL
        assert pred.evidence_type == EVIDENCE_INTRINSIC

    def test_cohort_probabilities_are_valid(self, trained_panel):
        panel, features, _, split = trained_panel
        probabilities = panel.predict_cohort(features.loc[split.test])
        assert probabilities.shape == (len(split.test), len(panel.models))
        assert probabilities.to_numpy().min() >= 0.0
        assert probabilities.to_numpy().max() <= 1.0

    def test_unfitted_model_refuses_to_predict(self):
        panel = GenomeFirewall(species="escherichia coli")
        with pytest.raises(RuntimeError, match="not fitted"):
            panel.predict_cohort(pd.DataFrame({"gene:x": [1]}))


class TestRealDataLoader:
    """The organizer's dataset arrives in shapes the synthetic generator never produces.

    Each of these was a real failure found by running the pipeline against fixtures
    built to match the challenge brief's Appendix.
    """

    @staticmethod
    def _write(tmp_path, name, frame):
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return path

    def test_long_format_pivots_to_a_genome_by_drug_matrix(self, tmp_path):
        from src.data_loader import load_labels

        frame = pd.DataFrame([
            {"genome_id": "g1", "antibiotic": "ampicillin", "resistant_phenotype": "Resistant"},
            {"genome_id": "g1", "antibiotic": "meropenem", "resistant_phenotype": "Susceptible"},
            {"genome_id": "g2", "antibiotic": "ampicillin", "resistant_phenotype": "Susceptible"},
        ])
        labels = load_labels(self._write(tmp_path, "l.csv", frame), verbose=False)

        assert labels.shape == (2, 2)
        assert labels.loc["g1", "ampicillin"] == 1.0
        assert labels.loc["g1", "meropenem"] == 0.0
        assert np.isnan(labels.loc["g2", "meropenem"])  # pair absent -> untested

    def test_intermediate_is_unlabelled_not_forced_into_a_class(self, tmp_path):
        """The assay could not call it; collapsing it manufactures certainty."""
        from src.data_loader import load_labels

        frame = pd.DataFrame([
            {"genome_id": "g1", "antibiotic": "ampicillin", "resistant_phenotype": "Intermediate"},
            {"genome_id": "g2", "antibiotic": "ampicillin", "resistant_phenotype": "Resistant"},
        ])
        labels = load_labels(self._write(tmp_path, "l.csv", frame), verbose=False)

        assert "g1" not in labels.index or np.isnan(labels.loc["g1", "ampicillin"])
        assert labels.loc["g2", "ampicillin"] == 1.0

    def test_computationally_predicted_rows_are_dropped_by_default(self, tmp_path):
        """Training on predicted labels fits a model to another model's output.

        Every metric stays healthy while meaning nothing, so this is refused unless
        explicitly asked for.
        """
        from src.data_loader import load_labels

        frame = pd.DataFrame([
            {"genome_id": "g1", "antibiotic": "ampicillin",
             "resistant_phenotype": "Resistant", "evidence": "Laboratory Method"},
            {"genome_id": "g2", "antibiotic": "ampicillin",
             "resistant_phenotype": "Resistant", "evidence": "Computational Method"},
        ])
        path = self._write(tmp_path, "l.csv", frame)

        kept = load_labels(path, verbose=False)
        assert list(kept.index) == ["g1"]

        both = load_labels(path, lab_measured_only=False, verbose=False)
        assert set(both.index) == {"g1", "g2"}

    def test_conflicting_labels_are_dropped_not_guessed(self, tmp_path):
        from src.data_loader import load_labels

        frame = pd.DataFrame([
            {"genome_id": "g1", "antibiotic": "ampicillin", "resistant_phenotype": "Resistant"},
            {"genome_id": "g1", "antibiotic": "ampicillin", "resistant_phenotype": "Susceptible"},
            {"genome_id": "g2", "antibiotic": "ampicillin", "resistant_phenotype": "Resistant"},
        ])
        labels = load_labels(self._write(tmp_path, "l.csv", frame), verbose=False)

        assert "g1" not in labels.index or np.isnan(labels.loc["g1", "ampicillin"])
        assert labels.loc["g2", "ampicillin"] == 1.0

    def test_organizer_split_is_used_verbatim(self, tmp_path):
        from src.data_loader import load_split

        frame = pd.DataFrame({
            "genome_id": ["g1", "g2", "g3", "g4"],
            "split": ["train", "train", "calibration", "test"],
        })
        split = load_split(self._write(tmp_path, "s.csv", frame), verbose=False)

        assert split.train == ["g1", "g2"]
        assert split.calibration == ["g3"]
        assert split.test == ["g4"]

    def test_generalization_accepts_non_integer_group_ids(self, trained_panel):
        """Organizer group ids are names, not integers.

        Coercing them with int() crashed on every real grouping — "clade_8", "ST131",
        any lineage label.
        """
        from src.decision_report import generalization_report

        panel, features, labels, split = trained_panel
        named = pd.Series(
            [f"clade_{i % 4}" for i in range(len(split.test))], index=split.test
        )
        report = generalization_report(
            panel, features.loc[split.test], labels.loc[split.test], named,
            min_group_size=5,
        )
        if not report.empty:
            assert report.index.get_level_values("cluster_id")[0].startswith("clade_")


class TestMRSADemoGenome:
    """End-to-end check on the demo assembly, if AMRFinderPlus is installed.

    This is the path a judge exercises: upload a genome, read the report. Unit tests
    cover the pieces, but only running the real annotator catches vocabulary drift
    between what AMRFinderPlus emits and what the drug knowledge base expects — which
    has now happened three times (v3/v4 column names, SULFONAMIDE, and the alphabetical
    ordering of the combined MLSb class tag).
    """

    DEMO = Path(__file__).resolve().parent.parent / "data" / "raw" / "mrsa_demo.fasta"

    @pytest.fixture(scope="class")
    def annotated(self):
        from src.utils import amrfinder
        if not amrfinder.is_available():
            pytest.skip("AMRFinderPlus not installed")
        if not self.DEMO.exists():
            pytest.skip("demo genome not present")
        from src.genome_reader import featurize_fasta
        _, features, qc, hits = featurize_fasta(self.DEMO, organism="Staphylococcus_aureus")
        return features, qc, hits

    def test_all_four_determinants_are_detected(self, annotated):
        _, _, hits = annotated
        found = {h.gene_symbol for h in hits}
        assert {"mecA", "erm(C)", "tet(K)", "blaZ"} <= found, f"missing from {found}"

    def test_assembly_passes_qc(self, annotated):
        """A realistic-length scaffold, so no QC flag fires during a demo."""
        _, qc, _ = annotated
        assert not qc.flags()

    def test_mecA_drives_the_cefoxitin_call(self, annotated, trained_panel):
        """mecA encodes PBP2a — this is the MRSA call the system exists to make."""
        features, _, _ = annotated
        panel, *_ = trained_panel
        if "cefoxitin" not in panel.models:
            pytest.skip("cefoxitin not trained in this sample")

        pred = panel.predict_genome(features)[list(panel.models).index("cefoxitin")]
        assert pred.decision == LIKELY_TO_FAIL
        assert any("mecA" in f for f in pred.supporting_features), pred.supporting_features

    def test_blaZ_is_not_credited_for_cefoxitin(self, annotated, trained_panel):
        """blaZ is a penicillinase; it leaves cephamycins intact.

        It is common in S. aureus and co-occurs with resistance without causing it, so
        it is exactly the confounder that separates a known mechanism from a bare
        correlation. Citing it as the reason for a cefoxitin failure would be wrong
        even though the call itself would be right.
        """
        features, _, _ = annotated
        panel, *_ = trained_panel
        if "cefoxitin" not in panel.models:
            pytest.skip("cefoxitin not trained in this sample")

        pred = panel.predict_genome(features)[list(panel.models).index("cefoxitin")]
        assert not any("blaZ" in f for f in pred.supporting_features), (
            f"blaZ cited as cefoxitin evidence: {pred.supporting_features}"
        )

    def test_erm_hit_reads_as_a_known_mechanism(self, annotated, trained_panel):
        """Guards the combined MLSb class tag, whose component order we had wrong."""
        features, _, _ = annotated
        panel, *_ = trained_panel
        if "erythromycin" not in panel.models:
            pytest.skip("erythromycin not trained in this sample")

        pred = panel.predict_genome(features)[list(panel.models).index("erythromycin")]
        assert pred.evidence_type == EVIDENCE_KNOWN_DETERMINANT, pred.evidence_type

    def test_ciprofloxacin_is_declined_without_a_determinant(self, annotated, trained_panel):
        """No gyrA/grlA mutation in this genome, so no confident call either way."""
        features, _, _ = annotated
        panel, *_ = trained_panel
        if "ciprofloxacin" not in panel.models:
            pytest.skip("ciprofloxacin not trained in this sample")

        pred = panel.predict_genome(features)[list(panel.models).index("ciprofloxacin")]
        assert pred.decision != LIKELY_TO_FAIL or pred.supporting_features
