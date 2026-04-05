import json
import time
import argparse
import sys
import math
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    LLM_MODEL, OLLAMA_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT, RESULTS_DIR, SCENARIOS_DIR,
    GUARD_NUM_PASSES, RAG_TOP_K, EMBEDDING_MODEL
)
from src.utils.prompt_templates import MULTI_PASS_PROMPT_VARIATIONS, SYSTEM_PROMPT
from src.utils.output_parser import parse_llm_output, extract_technique_ids
from src.agents.rag_knowledge_base import MitreKnowledgeBase

# ============================================
# LLM Setup
# ============================================
try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARNING] ollama not installed. Run: pip install ollama")


def run_single_pass(scenario: dict, prompt_template: str, temperature: float,
                    kb: Optional[MitreKnowledgeBase] = None) -> dict:
    """
    Run one inference pass with a specific prompt variation and temperature.
    Optionally includes RAG context if knowledge base is provided.
    """
    # Build RAG context if available
    retrieved_context = ""
    if kb is not None:
        results = kb.query(scenario["event_description"], top_k=RAG_TOP_K)
        retrieved_context = kb.format_context_for_llm(results)

    # Format prompt
    prompt = prompt_template.format(
        event_description=scenario["event_description"],
        timestamp=scenario["timestamp"],
        source_org=scenario["source_org"],
        event_type=scenario["event_type"],
        data_source=scenario["data_source"],
    )

    # Append explicit JSON format (Llama 3 needs this to produce parseable output)
    json_format = """

Respond with ONLY this exact JSON structure, no other text:
{
    "threat_classification": "<BEC Payment Fraud | Invoice Fraud | Data Tampering | Network Intrusion | Insider Threat | Ransomware | Phishing | DDoS | Brute Force | Benign/Normal>",
    "severity_level": <1-5>,
    "confidence": <0.0-1.0>,
    "mitre_attack_techniques": [{"technique_id": "<e.g. T1566.002>", "technique_name": "<name>", "tactic": "<tactic>", "relevance": "<why>"}],
    "detected_indicators": ["<indicator1>", "<indicator2>"],
    "reasoning_chain": "<step by step reasoning>",
    "recommended_actions": [{"action": "<action>", "priority": "<immediate|short-term|long-term>", "rationale": "<why>"}],
    "false_positive_assessment": "<assessment>"
}"""
    prompt = prompt + json_format

    # Prepend RAG context if available
    if retrieved_context:
        prompt = f"## Relevant MITRE ATT&CK Intelligence\n{retrieved_context}\n\n{prompt}"

    # Run LLM inference
    start_time = time.time()
    try:
        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": LLM_MAX_TOKENS,
            },
        )
        raw_output = response["message"]["content"]
    except Exception as e:
        raw_output = f"LLM_ERROR: {str(e)}"

    inference_time = time.time() - start_time

    # Parse output
    assessment = parse_llm_output(raw_output)
    technique_ids = extract_technique_ids(assessment)

    return {
        "raw_output": raw_output,
        "threat_classification": assessment.threat_classification,
        "severity_level": assessment.severity_level,
        "confidence": assessment.confidence,
        "technique_ids": technique_ids,
        "parse_success": assessment.parse_success,
        "inference_time": round(inference_time, 3),
    }


# ============================================
# CONSISTENCY SUB-METRICS
# ============================================

def compute_car(passes: list) -> float:
    """
    Classification Agreement Rate (CAR).
    Measures what fraction of passes agree on the most common threat label.
    
    CAR = (count of most frequent label) / (total passes)
    
    CAR = 1.0 -> all passes agree (high consistency)
    CAR = 0.2 -> all passes disagree (low consistency, k=5)
    """
    classifications = [p["threat_classification"] for p in passes if p["parse_success"]]
    if not classifications:
        return 0.0

    counter = Counter(classifications)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(classifications)


def compute_sv(passes: list) -> float:
    """
    Severity Variance (SV).
    Standard deviation of severity levels across passes.
    
    Normalized to [0, 1] where:
      SV_norm = std_dev / max_possible_std_dev
      Max std dev for 1-5 scale = 2.0 (all 1s and 5s)
    
    Returns 1 - SV_norm so higher = more consistent.
    """
    severities = [p["severity_level"] for p in passes if p["parse_success"] and p["severity_level"] > 0]
    if len(severities) < 2:
        return 1.0

    mean = sum(severities) / len(severities)
    variance = sum((s - mean) ** 2 for s in severities) / len(severities)
    std_dev = math.sqrt(variance)

    max_std_dev = 2.0
    sv_normalized = min(std_dev / max_std_dev, 1.0)

    return round(1.0 - sv_normalized, 4)


def compute_eos(passes: list) -> float:
    """
    Evidence Overlap Score (EOS).
    Average pairwise Jaccard similarity of cited ATT&CK technique sets.
    
    EOS = 1.0 -> all passes cite exactly the same techniques
    EOS = 0.0 -> no overlap in cited techniques
    """
    technique_sets = [
        set(p["technique_ids"]) for p in passes
        if p["parse_success"] and len(p["technique_ids"]) > 0
    ]

    if len(technique_sets) < 2:
        empty_count = sum(1 for p in passes if p["parse_success"] and len(p["technique_ids"]) == 0)
        if empty_count == len([p for p in passes if p["parse_success"]]):
            return 1.0
        return 0.0

    total_jaccard = 0.0
    num_pairs = 0

    for i in range(len(technique_sets)):
        for j in range(i + 1, len(technique_sets)):
            set_a = technique_sets[i]
            set_b = technique_sets[j]
            union = set_a | set_b
            intersection = set_a & set_b

            if len(union) > 0:
                jaccard = len(intersection) / len(union)
            else:
                jaccard = 1.0

            total_jaccard += jaccard
            num_pairs += 1

    return round(total_jaccard / num_pairs, 4) if num_pairs > 0 else 0.0


def compute_consistency_score(passes: list, weights: tuple = (0.4, 0.3, 0.3)) -> dict:
    """
    Compute the full Consistency Score C(O).
    
    C(O) = w1*CAR + w2*SV_score + w3*EOS
    
    Default weights: CAR=0.4, SV=0.3, EOS=0.3
    (CAR weighted higher because classification is most critical)
    """
    w1, w2, w3 = weights

    car = compute_car(passes)
    sv_score = compute_sv(passes)
    eos = compute_eos(passes)

    consistency_score = w1 * car + w2 * sv_score + w3 * eos

    # Analysis
    classifications = [p["threat_classification"] for p in passes if p["parse_success"]]
    severities = [p["severity_level"] for p in passes if p["parse_success"] and p["severity_level"] > 0]
    confidences = [p["confidence"] for p in passes if p["parse_success"] and p["confidence"] > 0]

    # Majority vote classification
    if classifications:
        counter = Counter(classifications)
        majority_classification = counter.most_common(1)[0][0]
        classification_distribution = dict(counter)
    else:
        majority_classification = "PARSE_ERROR"
        classification_distribution = {}

    median_severity = sorted(severities)[len(severities) // 2] if severities else 0
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Techniques appearing in majority of passes
    all_techniques = set()
    technique_counts = Counter()
    valid_passes = [p for p in passes if p["parse_success"]]
    for p in valid_passes:
        all_techniques.update(p["technique_ids"])
        for tid in p["technique_ids"]:
            technique_counts[tid] += 1

    majority_threshold = len(valid_passes) / 2
    consensus_techniques = [
        tid for tid, count in technique_counts.items()
        if count > majority_threshold
    ]

    return {
        "consistency_score": round(consistency_score, 4),
        "sub_metrics": {
            "car": round(car, 4),
            "sv_score": round(sv_score, 4),
            "eos": round(eos, 4),
        },
        "weights": {"w1_car": w1, "w2_sv": w2, "w3_eos": w3},
        "consensus": {
            "majority_classification": majority_classification,
            "classification_distribution": classification_distribution,
            "median_severity": median_severity,
            "mean_confidence": round(mean_confidence, 4),
            "consensus_techniques": consensus_techniques,
            "all_techniques_cited": list(all_techniques),
        },
        "pass_details": {
            "total_passes": len(passes),
            "successful_passes": len(valid_passes),
            "classifications": classifications,
            "severities": severities,
        },
    }


# ============================================
# MULTI-PASS RUNNER
# ============================================

def run_multi_pass_analysis(scenario: dict, num_passes: int = GUARD_NUM_PASSES,
                            kb: Optional[MitreKnowledgeBase] = None) -> dict:
    """
    Run k independent reasoning passes on a single scenario.
    Each pass uses a different prompt variation and stochastic sampling.
    """
    passes = []
    prompts = MULTI_PASS_PROMPT_VARIATIONS

    for i in range(num_passes):
        prompt_idx = i % len(prompts)
        prompt_template = prompts[prompt_idx]
        temperature = LLM_TEMPERATURE

        print(f"      Pass {i+1}/{num_passes} (prompt_var={prompt_idx+1}, temp={temperature})...",
              end=" ", flush=True)

        result = run_single_pass(scenario, prompt_template, temperature, kb)
        passes.append(result)

        status = "ok" if result["parse_success"] else "FAIL"
        print(f"{status} -> {result['threat_classification']} "
              f"(sev={result['severity_level']}, techs={len(result['technique_ids'])}, "
              f"{result['inference_time']:.1f}s)")

    # Compute consistency score
    consistency = compute_consistency_score(passes)

    # Compare with ground truth
    ground_truth = scenario.get("ground_truth", {})
    gt_classification = ground_truth.get("classification", "")
    gt_is_attack = ground_truth.get("is_attack", None)
    gt_techniques = ground_truth.get("attack_techniques", [])

    consensus_correct = consistency["consensus"]["majority_classification"] == gt_classification
    consensus_is_attack = consistency["consensus"]["majority_classification"] != "Benign/Normal"
    consensus_attack_correct = consensus_is_attack == gt_is_attack if gt_is_attack is not None else None

    consensus_techs = set(consistency["consensus"]["consensus_techniques"])
    gt_tech_set = set(gt_techniques)
    consensus_technique_overlap = (
        len(consensus_techs & gt_tech_set) / len(consensus_techs | gt_tech_set)
        if (consensus_techs | gt_tech_set) else 1.0 if not gt_tech_set else 0.0
    )

    return {
        "scenario_id": scenario["id"],
        "num_passes": num_passes,
        "consistency": consistency,
        "individual_passes": passes,
        "evaluation": {
            "consensus_classification_correct": consensus_correct,
            "consensus_classification": consistency["consensus"]["majority_classification"],
            "ground_truth_classification": gt_classification,
            "consensus_attack_correct": consensus_attack_correct,
            "consensus_is_attack": consensus_is_attack,
            "ground_truth_is_attack": gt_is_attack,
            "consensus_severity": consistency["consensus"]["median_severity"],
            "ground_truth_severity": ground_truth.get("severity", 0),
            "consensus_technique_overlap": round(consensus_technique_overlap, 4),
            "consensus_techniques": list(consensus_techs),
            "ground_truth_techniques": gt_techniques,
        },
        "timing": {
            "total_time": round(sum(p["inference_time"] for p in passes), 3),
            "avg_pass_time": round(sum(p["inference_time"] for p in passes) / len(passes), 3),
        },
    }


def compute_summary_metrics(results: list) -> dict:
    """Compute aggregate metrics across all scenarios."""
    total = len(results)
    if total == 0:
        return {}

    c_scores = [r["consistency"]["consistency_score"] for r in results]
    car_scores = [r["consistency"]["sub_metrics"]["car"] for r in results]
    sv_scores = [r["consistency"]["sub_metrics"]["sv_score"] for r in results]
    eos_scores = [r["consistency"]["sub_metrics"]["eos"] for r in results]

    correct = sum(1 for r in results if r["evaluation"]["consensus_classification_correct"])

    attack_results = [r for r in results if r["evaluation"]["ground_truth_is_attack"] is not None]
    if attack_results:
        tp = sum(1 for r in attack_results if r["evaluation"]["consensus_is_attack"] and r["evaluation"]["ground_truth_is_attack"])
        tn = sum(1 for r in attack_results if not r["evaluation"]["consensus_is_attack"] and not r["evaluation"]["ground_truth_is_attack"])
        fp = sum(1 for r in attack_results if r["evaluation"]["consensus_is_attack"] and not r["evaluation"]["ground_truth_is_attack"])
        fn = sum(1 for r in attack_results if not r["evaluation"]["consensus_is_attack"] and r["evaluation"]["ground_truth_is_attack"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        tp = tn = fp = fn = 0
        precision = recall = f1 = 0

    benign = [r for r in results if not r["evaluation"]["ground_truth_is_attack"]]
    false_esc = sum(1 for r in benign if r["evaluation"]["consensus_is_attack"])

    correct_c = [r["consistency"]["consistency_score"] for r in results if r["evaluation"]["consensus_classification_correct"]]
    incorrect_c = [r["consistency"]["consistency_score"] for r in results if not r["evaluation"]["consensus_classification_correct"]]
    attack_c = [r["consistency"]["consistency_score"] for r in results if r["evaluation"]["ground_truth_is_attack"]]
    benign_c = [r["consistency"]["consistency_score"] for r in results if not r["evaluation"]["ground_truth_is_attack"]]

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0

    return {
        "total_scenarios": total,
        "consistency_distribution": {
            "mean_C": avg(c_scores), "min_C": round(min(c_scores), 4) if c_scores else 0,
            "max_C": round(max(c_scores), 4) if c_scores else 0,
            "mean_CAR": avg(car_scores), "mean_SV": avg(sv_scores), "mean_EOS": avg(eos_scores),
            "all_C_scores": [round(c, 4) for c in c_scores],
        },
        "consistency_by_correctness": {
            "avg_C_when_correct": avg(correct_c),
            "avg_C_when_incorrect": avg(incorrect_c),
            "c_score_gap": round(avg(correct_c) - avg(incorrect_c), 4),
        },
        "consistency_by_type": {
            "avg_C_attack_scenarios": avg(attack_c),
            "avg_C_benign_scenarios": avg(benign_c),
        },
        "consensus_accuracy": {
            "classification_accuracy": round(correct / total, 4),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        },
        "false_escalation_rate": round(false_esc / len(benign), 4) if benign else 0,
        "timing": {
            "avg_total_time_per_scenario": avg([r["timing"]["total_time"] for r in results]),
            "avg_per_pass_time": avg([r["timing"]["avg_pass_time"] for r in results]),
        },
    }


def print_summary(metrics: dict):
    """Print formatted summary."""
    print("\n" + "=" * 65)
    print("  WEEK 3 RESULTS: Guard Agent Consistency Score C(O)")
    print("=" * 65)

    cd = metrics["consistency_distribution"]
    print(f"\n  --- Consistency Score Distribution ---")
    print(f"  Mean C(O):  {cd['mean_C']:.4f}")
    print(f"  Min C(O):   {cd['min_C']:.4f}")
    print(f"  Max C(O):   {cd['max_C']:.4f}")
    print(f"  Mean CAR:   {cd['mean_CAR']:.4f}  (classification agreement)")
    print(f"  Mean SV:    {cd['mean_SV']:.4f}  (severity stability)")
    print(f"  Mean EOS:   {cd['mean_EOS']:.4f}  (evidence overlap)")

    cc = metrics["consistency_by_correctness"]
    print(f"\n  --- C(O) vs Correctness (KEY INSIGHT) ---")
    print(f"  Avg C(O) when CORRECT:   {cc['avg_C_when_correct']:.4f}")
    print(f"  Avg C(O) when INCORRECT: {cc['avg_C_when_incorrect']:.4f}")
    print(f"  Gap: {cc['c_score_gap']:+.4f}")
    if cc['c_score_gap'] > 0:
        print(f"  >> Higher consistency correlates with correctness!")
    else:
        print(f"  !! Consistency does not correlate with correctness")

    ct = metrics["consistency_by_type"]
    print(f"\n  --- C(O) by Scenario Type ---")
    print(f"  Avg C(O) for attacks: {ct['avg_C_attack_scenarios']:.4f}")
    print(f"  Avg C(O) for benign:  {ct['avg_C_benign_scenarios']:.4f}")

    ca = metrics["consensus_accuracy"]
    print(f"\n  --- Consensus Classification (Majority Vote) ---")
    print(f"  Accuracy:  {ca['classification_accuracy']:.1%}")
    print(f"  Precision: {ca['precision']:.1%}")
    print(f"  Recall:    {ca['recall']:.1%}")
    print(f"  F1 Score:  {ca['f1_score']:.1%}")
    print(f"  TP={ca['tp']} TN={ca['tn']} FP={ca['fp']} FN={ca['fn']}")

    print(f"\n  False Escalation Rate: {metrics['false_escalation_rate']:.1%}")

    tm = metrics["timing"]
    print(f"\n  --- Timing ---")
    print(f"  Avg total per scenario: {tm['avg_total_time_per_scenario']:.1f}s")
    print(f"  Avg per pass: {tm['avg_per_pass_time']:.1f}s")

    print("\n" + "=" * 65)
    print("  This C(O) becomes part of: RTS(O) = a*C(O) + b*V(O) + g*S(O)")
    print("  Week 4: V(O) -- ATT&CK Validation Score")
    print("  Week 5: S(O) -- Semantic Stability Score")
    print("=" * 65 + "\n")


def load_scenarios(path: Optional[str] = None) -> list:
    """Load threat scenarios."""
    if path is None:
        path = SCENARIOS_DIR / "threat_scenarios.json"
    else:
        path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)
    return data["scenarios"]


def main():
    parser = argparse.ArgumentParser(description="Guard Agent - Week 3: Consistency Score C(O)")
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--single", type=str, default=None, help="Run single scenario by ID")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--passes", type=int, default=GUARD_NUM_PASSES, help="Number of passes (default 5)")
    parser.add_argument("--no-rag", action="store_true", help="Run without RAG")
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")
    args = parser.parse_args()

    print("\n===  Guard Agent - Week 3: Consistency Score C(O)  ===")
    print(f"   Model: {LLM_MODEL}")
    print(f"   Temperature: {LLM_TEMPERATURE} (stochastic sampling)")
    print(f"   Passes per scenario: {args.passes}")
    print(f"   RAG enabled: {not args.no_rag}")
    print(f"   Timestamp: {datetime.now().isoformat()}")

    scenarios = load_scenarios(args.scenarios)
    print(f"   Loaded {len(scenarios)} scenarios")

    if args.single:
        scenarios = [s for s in scenarios if s["id"] == args.single]
        if not scenarios:
            print(f"   ERROR: Scenario {args.single} not found")
            return
        print(f"   Running single scenario: {args.single}")

    if args.dry_run:
        print(f"\n   [DRY RUN] Would run {len(scenarios)} scenarios x {args.passes} passes = {len(scenarios) * args.passes} LLM calls")
        print(f"   Estimated time: ~{len(scenarios) * args.passes * 55 / 60:.0f} minutes")
        print(f"   Prompt variations: {len(MULTI_PASS_PROMPT_VARIATIONS)}")
        for i, p in enumerate(MULTI_PASS_PROMPT_VARIATIONS):
            print(f"   Variation {i+1}: {p[:80]}...")
        return

    # Initialize RAG
    kb = None
    if not args.no_rag:
        print("\n   Initializing RAG Knowledge Base...")
        try:
            kb = MitreKnowledgeBase()
            kb.load()
        except Exception as e:
            print(f"   [WARNING] RAG unavailable: {e}")
            kb = None

    if not OLLAMA_AVAILABLE:
        print("   ERROR: ollama package not installed. Run: pip install ollama")
        return

    print(f"\n   Starting multi-pass analysis ({len(scenarios)} scenarios x {args.passes} passes)...\n")

    results = []
    for i, scenario in enumerate(scenarios):
        print(f"   [{i+1}/{len(scenarios)}] {scenario['id']} -- {scenario['event_type']}")

        result = run_multi_pass_analysis(scenario, num_passes=args.passes, kb=kb)
        results.append(result)

        c = result["consistency"]
        ev = result["evaluation"]
        status = "CORRECT" if ev["consensus_classification_correct"] else "WRONG"
        print(f"      {status} C(O)={c['consistency_score']:.3f} "
              f"(CAR={c['sub_metrics']['car']:.2f} SV={c['sub_metrics']['sv_score']:.2f} "
              f"EOS={c['sub_metrics']['eos']:.2f}) "
              f"-> {c['consensus']['majority_classification']}\n")

    metrics = compute_summary_metrics(results)
    print_summary(metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(RESULTS_DIR / "guard_consistency_results.json")

    output_data = {
        "experiment": "Week 3 - Guard Agent Consistency Score C(O)",
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "num_passes": args.passes,
        "rag_enabled": not args.no_rag,
        "timestamp": datetime.now().isoformat(),
        "summary_metrics": metrics,
        "individual_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"   Results saved to: {output_path}")


if __name__ == "__main__":
    main()