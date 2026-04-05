import sys
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.guard_consistency import (
    compute_car,
    compute_severity_variance,
    compute_eos,
    compute_consistency_score,
)
from src.utils.output_parser import ThreatAssessment


def make_assessment(classification, severity, confidence, techniques):
    """Helper to create mock ThreatAssessment objects."""
    return ThreatAssessment(
        threat_classification=classification,
        severity_level=severity,
        confidence=confidence,
        mitre_attack_techniques=[
            {"technique_id": t, "technique_name": "", "tactic": "", "relevance": ""}
            for t in techniques
        ],
        detected_indicators=["indicator1"],
        reasoning_chain="test reasoning",
        recommended_actions=[{"action": "test", "priority": "immediate", "rationale": "test"}],
        parse_success=True,
    )


# ============================================
# Test CAR (Classification Agreement Rate)
# ============================================

def test_car_perfect_agreement():
    """All 5 passes agree on same label."""
    assessments = [
        make_assessment("BEC Payment Fraud", 5, 0.9, ["T1566.002"]),
        make_assessment("BEC Payment Fraud", 5, 0.85, ["T1566.002"]),
        make_assessment("BEC Payment Fraud", 4, 0.88, ["T1566.002"]),
        make_assessment("BEC Payment Fraud", 5, 0.92, ["T1566.002"]),
        make_assessment("BEC Payment Fraud", 5, 0.87, ["T1566.002"]),
    ]
    car = compute_car(assessments)
    assert car == 1.0, f"Expected 1.0, got {car}"
    print("✅ test_car_perfect_agreement PASSED (CAR=1.0)")


def test_car_majority_agreement():
    """3 out of 5 passes agree."""
    assessments = [
        make_assessment("BEC Payment Fraud", 5, 0.9, ["T1566.002"]),
        make_assessment("BEC Payment Fraud", 5, 0.85, ["T1566.002"]),
        make_assessment("Phishing", 4, 0.7, ["T1566"]),
        make_assessment("BEC Payment Fraud", 5, 0.88, ["T1566.002"]),
        make_assessment("Invoice Fraud", 3, 0.6, ["T1565"]),
    ]
    car = compute_car(assessments)
    assert car == 0.6, f"Expected 0.6, got {car}"
    print("✅ test_car_majority_agreement PASSED (CAR=0.6)")


def test_car_no_agreement():
    """All 5 passes give different labels."""
    assessments = [
        make_assessment("BEC Payment Fraud", 5, 0.9, ["T1566"]),
        make_assessment("Phishing", 4, 0.7, ["T1566.002"]),
        make_assessment("Invoice Fraud", 3, 0.6, ["T1565"]),
        make_assessment("Data Tampering", 4, 0.75, ["T1565.001"]),
        make_assessment("Insider Threat", 3, 0.65, ["T1078"]),
    ]
    car = compute_car(assessments)
    assert car == 0.2, f"Expected 0.2, got {car}"
    print("✅ test_car_no_agreement PASSED (CAR=0.2)")


# ============================================
# Test Severity Variance
# ============================================

def test_sv_perfect_consistency():
    """All passes give same severity."""
    assessments = [
        make_assessment("BEC Payment Fraud", 5, 0.9, []),
        make_assessment("BEC Payment Fraud", 5, 0.9, []),
        make_assessment("BEC Payment Fraud", 5, 0.9, []),
        make_assessment("BEC Payment Fraud", 5, 0.9, []),
        make_assessment("BEC Payment Fraud", 5, 0.9, []),
    ]
    sv = compute_severity_variance(assessments)
    assert sv == 0.0, f"Expected 0.0, got {sv}"
    print("✅ test_sv_perfect_consistency PASSED (SV=0.0)")


def test_sv_some_variance():
    """Passes give different severities."""
    assessments = [
        make_assessment("BEC Payment Fraud", 5, 0.9, []),
        make_assessment("BEC Payment Fraud", 4, 0.85, []),
        make_assessment("BEC Payment Fraud", 5, 0.88, []),
        make_assessment("BEC Payment Fraud", 4, 0.82, []),
        make_assessment("BEC Payment Fraud", 5, 0.9, []),
    ]
    sv = compute_severity_variance(assessments)
    assert 0 < sv < 0.5, f"Expected moderate variance, got {sv}"
    print(f"✅ test_sv_some_variance PASSED (SV={sv})")


def test_sv_high_variance():
    """Passes give wildly different severities."""
    assessments = [
        make_assessment("BEC Payment Fraud", 1, 0.5, []),
        make_assessment("BEC Payment Fraud", 5, 0.9, []),
        make_assessment("BEC Payment Fraud", 1, 0.4, []),
        make_assessment("BEC Payment Fraud", 5, 0.85, []),
        make_assessment("BEC Payment Fraud", 1, 0.3, []),
    ]
    sv = compute_severity_variance(assessments)
    assert sv > 0.5, f"Expected high variance, got {sv}"
    print(f"✅ test_sv_high_variance PASSED (SV={sv})")


# ============================================
# Test EOS (Evidence Overlap Score)
# ============================================

def test_eos_perfect_overlap():
    """All passes cite same techniques."""
    assessments = [
        make_assessment("BEC", 5, 0.9, ["T1566.002", "T1534"]),
        make_assessment("BEC", 5, 0.9, ["T1566.002", "T1534"]),
        make_assessment("BEC", 5, 0.9, ["T1566.002", "T1534"]),
        make_assessment("BEC", 5, 0.9, ["T1566.002", "T1534"]),
        make_assessment("BEC", 5, 0.9, ["T1566.002", "T1534"]),
    ]
    eos = compute_eos(assessments)
    assert eos == 1.0, f"Expected 1.0, got {eos}"
    print("✅ test_eos_perfect_overlap PASSED (EOS=1.0)")


def test_eos_partial_overlap():
    """Passes cite some common and some different techniques."""
    assessments = [
        make_assessment("BEC", 5, 0.9, ["T1566.002", "T1534"]),
        make_assessment("BEC", 5, 0.9, ["T1566.002", "T1078"]),
        make_assessment("BEC", 5, 0.9, ["T1566.002", "T1534"]),
    ]
    eos = compute_eos(assessments)
    assert 0 < eos < 1.0, f"Expected partial overlap, got {eos}"
    print(f"✅ test_eos_partial_overlap PASSED (EOS={eos})")


def test_eos_no_overlap():
    """No techniques shared across any passes."""
    assessments = [
        make_assessment("BEC", 5, 0.9, ["T1566"]),
        make_assessment("BEC", 5, 0.9, ["T1078"]),
        make_assessment("BEC", 5, 0.9, ["T1486"]),
        make_assessment("BEC", 5, 0.9, ["T1110"]),
        make_assessment("BEC", 5, 0.9, ["T1498"]),
    ]
    eos = compute_eos(assessments)
    assert eos == 0.0, f"Expected 0.0, got {eos}"
    print("✅ test_eos_no_overlap PASSED (EOS=0.0)")


def test_eos_empty_techniques():
    """All passes cite no techniques (both empty = agreement)."""
    assessments = [
        make_assessment("Benign/Normal", 1, 0.9, []),
        make_assessment("Benign/Normal", 1, 0.85, []),
        make_assessment("Benign/Normal", 1, 0.88, []),
    ]
    eos = compute_eos(assessments)
    assert eos == 1.0, f"Expected 1.0 (all empty = agreement), got {eos}"
    print("✅ test_eos_empty_techniques PASSED (EOS=1.0)")


# ============================================
# Test Full C(O) Score
# ============================================

def test_consistency_score_perfect():
    """Perfect agreement on everything → C(O) close to 1.0."""
    assessments = [
        make_assessment("BEC Payment Fraud", 5, 0.9, ["T1566.002", "T1534"]),
        make_assessment("BEC Payment Fraud", 5, 0.92, ["T1566.002", "T1534"]),
        make_assessment("BEC Payment Fraud", 5, 0.88, ["T1566.002", "T1534"]),
        make_assessment("BEC Payment Fraud", 5, 0.91, ["T1566.002", "T1534"]),
        make_assessment("BEC Payment Fraud", 5, 0.89, ["T1566.002", "T1534"]),
    ]
    result = compute_consistency_score(assessments)
    c_score = result["consistency_score"]
    assert c_score >= 0.95, f"Expected >= 0.95, got {c_score}"
    assert result["car"] == 1.0
    assert result["severity_variance"] == 0.0
    assert result["eos"] == 1.0
    assert result["details"]["majority_label"] == "BEC Payment Fraud"
    print(f"✅ test_consistency_score_perfect PASSED (C(O)={c_score})")


def test_consistency_score_poor():
    """Complete disagreement → C(O) low."""
    assessments = [
        make_assessment("BEC Payment Fraud", 1, 0.5, ["T1566"]),
        make_assessment("Phishing", 5, 0.9, ["T1078"]),
        make_assessment("Data Tampering", 3, 0.6, ["T1486"]),
        make_assessment("Insider Threat", 2, 0.4, ["T1110"]),
        make_assessment("Ransomware", 4, 0.7, ["T1498"]),
    ]
    result = compute_consistency_score(assessments)
    c_score = result["consistency_score"]
    assert c_score < 0.4, f"Expected < 0.4, got {c_score}"
    assert result["car"] == 0.2
    assert result["eos"] == 0.0
    print(f"✅ test_consistency_score_poor PASSED (C(O)={c_score})")


def test_consistency_score_moderate():
    """Partial agreement → moderate C(O)."""
    assessments = [
        make_assessment("BEC Payment Fraud", 5, 0.9, ["T1566.002", "T1534"]),
        make_assessment("BEC Payment Fraud", 4, 0.85, ["T1566.002", "T1078"]),
        make_assessment("BEC Payment Fraud", 5, 0.88, ["T1566.002", "T1534"]),
        make_assessment("Phishing", 4, 0.7, ["T1566.002"]),
        make_assessment("BEC Payment Fraud", 5, 0.82, ["T1566.002", "T1534", "T1078"]),
    ]
    result = compute_consistency_score(assessments)
    c_score = result["consistency_score"]
    assert 0.5 < c_score < 0.9, f"Expected moderate, got {c_score}"
    assert result["details"]["majority_label"] == "BEC Payment Fraud"
    print(f"✅ test_consistency_score_moderate PASSED (C(O)={c_score})")


def test_consistency_score_weights():
    """Verify custom weights work correctly."""
    assessments = [
        make_assessment("BEC Payment Fraud", 5, 0.9, ["T1566.002"]),
        make_assessment("BEC Payment Fraud", 5, 0.9, ["T1566.002"]),
        make_assessment("BEC Payment Fraud", 5, 0.9, ["T1566.002"]),
    ]
    # Default weights
    r1 = compute_consistency_score(assessments, w1=0.4, w2=0.3, w3=0.3)
    # All weight on CAR
    r2 = compute_consistency_score(assessments, w1=1.0, w2=0.0, w3=0.0)
    # All weight on EOS
    r3 = compute_consistency_score(assessments, w1=0.0, w2=0.0, w3=1.0)

    assert r2["consistency_score"] == 1.0  # CAR = 1.0, weight = 1.0
    assert r3["consistency_score"] == 1.0  # EOS = 1.0, weight = 1.0
    print(f"✅ test_consistency_score_weights PASSED")


def test_consistency_output_structure():
    """Verify the output dict has all required fields."""
    assessments = [
        make_assessment("BEC Payment Fraud", 5, 0.9, ["T1566.002"]),
        make_assessment("BEC Payment Fraud", 4, 0.85, ["T1566"]),
    ]
    result = compute_consistency_score(assessments)

    required_keys = ["consistency_score", "car", "severity_variance", "eos", "weights", "details"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    detail_keys = [
        "num_passes", "labels_per_pass", "label_distribution",
        "majority_label", "severities_per_pass", "mean_severity",
        "mean_confidence", "techniques_per_pass", "techniques_union",
        "techniques_intersection", "technique_count_per_pass"
    ]
    for key in detail_keys:
        assert key in result["details"], f"Missing detail key: {key}"

    print("✅ test_consistency_output_structure PASSED")


# ============================================
# Run all tests
# ============================================
if __name__ == "__main__":
    print("\n🧪 Running Week 3 Tests...\n")

    # CAR tests
    test_car_perfect_agreement()
    test_car_majority_agreement()
    test_car_no_agreement()

    # SV tests
    test_sv_perfect_consistency()
    test_sv_some_variance()
    test_sv_high_variance()

    # EOS tests
    test_eos_perfect_overlap()
    test_eos_partial_overlap()
    test_eos_no_overlap()
    test_eos_empty_techniques()

    # Full C(O) tests
    test_consistency_score_perfect()
    test_consistency_score_poor()
    test_consistency_score_moderate()
    test_consistency_score_weights()
    test_consistency_output_structure()

    print(f"\n✅ All 15 Week 3 tests passed!\n")
