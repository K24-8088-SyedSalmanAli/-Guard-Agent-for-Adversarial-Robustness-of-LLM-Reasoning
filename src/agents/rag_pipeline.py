import json
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    LLM_MODEL, OLLAMA_BASE_URL, LLM_TEMPERATURE_BASELINE,
    LLM_MAX_TOKENS, LLM_REQUEST_TIMEOUT, RESULTS_DIR,
    SCENARIOS_DIR, RAG_TOP_K, EMBEDDING_MODEL
)
from src.utils.prompt_templates import RAG_THREAT_ANALYSIS_PROMPT, SYSTEM_PROMPT
from src.utils.output_parser import parse_llm_output, compute_output_completeness, extract_technique_ids
from src.agents.rag_knowledge_base import MitreKnowledgeBase

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def get_llm():
    """Initialize the LLM client."""
    if LANGCHAIN_AVAILABLE:
        return ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE_BASELINE,
            num_predict=LLM_MAX_TOKENS,
            timeout=LLM_REQUEST_TIMEOUT,
        )
    elif OLLAMA_AVAILABLE:
        return "ollama_direct"
    else:
        raise RuntimeError("Neither langchain_ollama nor ollama package available.")


def run_rag_analysis(llm, kb: MitreKnowledgeBase, scenario: dict, top_k: int = RAG_TOP_K) -> dict:
    """
    Run a single threat scenario through the LLM+RAG pipeline.
    
    Pipeline:
    1. Query ChromaDB with event description → get relevant ATT&CK techniques
    2. Format retrieved context into prompt
    3. Send enriched prompt to LLM
    4. Parse and evaluate output
    """
    # === Step 1: RAG Retrieval ===
    retrieval_start = time.time()
    retrieved_results = kb.query(scenario["event_description"], top_k=top_k)
    retrieval_time = time.time() - retrieval_start

    # Format context for LLM
    retrieved_context = kb.format_context_for_llm(retrieved_results)

    # Track what was retrieved (for analysis)
    retrieved_technique_ids = [r["technique_id"] for r in retrieved_results]
    retrieved_similarities = [r["similarity"] for r in retrieved_results]

    # === Step 2: Build RAG-enhanced prompt ===
    prompt = RAG_THREAT_ANALYSIS_PROMPT.format(
        retrieved_context=retrieved_context,
        event_description=scenario["event_description"],
        timestamp=scenario["timestamp"],
        source_org=scenario["source_org"],
        event_type=scenario["event_type"],
        data_source=scenario["data_source"],
    )

    # === Step 3: LLM Inference ===
    inference_start = time.time()

    if LANGCHAIN_AVAILABLE and not isinstance(llm, str):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        try:
            response = llm.invoke(messages)
            raw_output = response.content
        except Exception as e:
            raw_output = f"LLM_ERROR: {str(e)}"
    elif OLLAMA_AVAILABLE:
        try:
            response = ollama_client.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": LLM_TEMPERATURE_BASELINE,
                    "num_predict": LLM_MAX_TOKENS,
                },
            )
            raw_output = response["message"]["content"]
        except Exception as e:
            raw_output = f"LLM_ERROR: {str(e)}"
    else:
        raw_output = "LLM_ERROR: No LLM client available"

    inference_time = time.time() - inference_start
    total_time = retrieval_time + inference_time

    # === Step 4: Parse output ===
    assessment = parse_llm_output(raw_output)
    completeness = compute_output_completeness(assessment)
    technique_ids = extract_technique_ids(assessment)

    # === Step 5: Evaluate against ground truth ===
    ground_truth = scenario.get("ground_truth", {})
    gt_classification = ground_truth.get("classification", "")
    gt_techniques = ground_truth.get("attack_techniques", [])
    gt_is_attack = ground_truth.get("is_attack", None)

    classification_correct = assessment.threat_classification == gt_classification
    is_attack_predicted = assessment.threat_classification != "Benign/Normal"
    is_attack_correct = is_attack_predicted == gt_is_attack if gt_is_attack is not None else None

    # Technique overlap
    predicted_set = set(technique_ids)
    gt_set = set(gt_techniques)
    technique_overlap = (
        len(predicted_set & gt_set) / len(predicted_set | gt_set)
        if (predicted_set | gt_set)
        else 1.0 if not gt_set else 0.0
    )

    hallucinated_techniques = list(predicted_set - gt_set) if gt_set else []

    # === RAG-specific metrics ===
    # Did the retrieval include the ground truth techniques?
    retrieved_set = set(retrieved_technique_ids)
    gt_techniques_retrieved = list(gt_set & retrieved_set)
    retrieval_recall = (
        len(gt_techniques_retrieved) / len(gt_set) if gt_set else 1.0
    )

    # Did the LLM use what was retrieved (vs hallucinating new ones)?
    predicted_from_retrieved = list(predicted_set & retrieved_set)
    retrieval_utilization = (
        len(predicted_from_retrieved) / len(predicted_set)
        if predicted_set else 1.0
    )

    # Validate all predicted technique IDs against MITRE database
    validation_results = {}
    for tid in technique_ids:
        validation_results[tid] = kb.validate_technique_id(tid)
    
    invalid_technique_ids = [
        tid for tid, v in validation_results.items() if not v["exists"]
    ]

    return {
        "scenario_id": scenario["id"],
        "retrieval_time_seconds": round(retrieval_time, 3),
        "inference_time_seconds": round(inference_time, 3),
        "total_time_seconds": round(total_time, 3),
        "raw_output": raw_output,
        "parsed_assessment": assessment.to_dict(),
        "output_completeness": round(completeness, 3),
        "rag_details": {
            "top_k": top_k,
            "retrieved_technique_ids": retrieved_technique_ids,
            "retrieved_similarities": [round(s, 4) for s in retrieved_similarities],
            "avg_retrieval_similarity": round(
                sum(retrieved_similarities) / len(retrieved_similarities), 4
            ) if retrieved_similarities else 0,
            "gt_techniques_in_retrieval": gt_techniques_retrieved,
            "retrieval_recall": round(retrieval_recall, 4),
            "retrieval_utilization": round(retrieval_utilization, 4),
        },
        "evaluation": {
            "classification_correct": classification_correct,
            "predicted_classification": assessment.threat_classification,
            "ground_truth_classification": gt_classification,
            "is_attack_correct": is_attack_correct,
            "predicted_is_attack": is_attack_predicted,
            "ground_truth_is_attack": gt_is_attack,
            "severity_predicted": assessment.severity_level,
            "severity_ground_truth": ground_truth.get("severity", 0),
            "severity_error": abs(assessment.severity_level - ground_truth.get("severity", 0)),
            "technique_overlap_jaccard": round(technique_overlap, 4),
            "predicted_techniques": technique_ids,
            "ground_truth_techniques": gt_techniques,
            "hallucinated_techniques": hallucinated_techniques,
            "invalid_technique_ids": invalid_technique_ids,
            "confidence_score": assessment.confidence,
        },
    }


def compute_summary_metrics(results: list) -> dict:
    """Compute aggregate metrics — same as Week 1 + RAG-specific metrics."""
    total = len(results)
    if total == 0:
        return {}

    # === Same metrics as Week 1 (for direct comparison) ===
    correct_classifications = sum(
        1 for r in results if r["evaluation"]["classification_correct"]
    )

    attack_results = [r for r in results if r["evaluation"]["ground_truth_is_attack"] is not None]
    if attack_results:
        tp = sum(1 for r in attack_results if r["evaluation"]["predicted_is_attack"] and r["evaluation"]["ground_truth_is_attack"])
        tn = sum(1 for r in attack_results if not r["evaluation"]["predicted_is_attack"] and not r["evaluation"]["ground_truth_is_attack"])
        fp = sum(1 for r in attack_results if r["evaluation"]["predicted_is_attack"] and not r["evaluation"]["ground_truth_is_attack"])
        fn = sum(1 for r in attack_results if not r["evaluation"]["predicted_is_attack"] and r["evaluation"]["ground_truth_is_attack"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(attack_results)
    else:
        tp = tn = fp = fn = 0
        precision = recall = f1 = accuracy = 0

    severity_errors = [r["evaluation"]["severity_error"] for r in results]
    technique_overlaps = [r["evaluation"]["technique_overlap_jaccard"] for r in results]

    total_hallucinated = sum(len(r["evaluation"]["hallucinated_techniques"]) for r in results)
    scenarios_with_hallucinations = sum(1 for r in results if len(r["evaluation"]["hallucinated_techniques"]) > 0)

    # Invalid technique IDs (hallucinated IDs that don't exist in MITRE at all)
    total_invalid_ids = sum(len(r["evaluation"]["invalid_technique_ids"]) for r in results)
    scenarios_with_invalid_ids = sum(1 for r in results if len(r["evaluation"]["invalid_technique_ids"]) > 0)

    parse_successes = sum(1 for r in results if r["parsed_assessment"]["parse_success"])

    benign_scenarios = [r for r in results if not r["evaluation"]["ground_truth_is_attack"]]
    false_escalations = sum(1 for r in benign_scenarios if r["evaluation"]["predicted_is_attack"])
    false_escalation_rate = false_escalations / len(benign_scenarios) if benign_scenarios else 0

    avg_confidence = sum(r["evaluation"]["confidence_score"] for r in results) / total
    avg_completeness = sum(r["output_completeness"] for r in results) / total

    # === RAG-specific metrics ===
    avg_retrieval_time = sum(r["retrieval_time_seconds"] for r in results) / total
    avg_inference_time = sum(r["inference_time_seconds"] for r in results) / total
    avg_total_time = sum(r["total_time_seconds"] for r in results) / total

    avg_retrieval_recall = sum(r["rag_details"]["retrieval_recall"] for r in results) / total
    avg_retrieval_utilization = sum(r["rag_details"]["retrieval_utilization"] for r in results) / total
    avg_retrieval_similarity = sum(r["rag_details"]["avg_retrieval_similarity"] for r in results) / total

    return {
        "total_scenarios": total,
        "classification_accuracy": round(correct_classifications / total, 4),
        "correct_classifications": correct_classifications,
        "binary_detection": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
        },
        "severity_analysis": {
            "mean_absolute_error": round(sum(severity_errors) / len(severity_errors), 3),
            "max_error": max(severity_errors),
        },
        "technique_mapping": {
            "avg_jaccard_overlap": round(sum(technique_overlaps) / len(technique_overlaps), 3),
            "total_hallucinated_techniques": total_hallucinated,
            "scenarios_with_hallucinations": scenarios_with_hallucinations,
            "hallucination_rate": round(scenarios_with_hallucinations / total, 4),
            "total_invalid_technique_ids": total_invalid_ids,
            "scenarios_with_invalid_ids": scenarios_with_invalid_ids,
            "invalid_id_rate": round(scenarios_with_invalid_ids / total, 4),
        },
        "false_escalation_rate": round(false_escalation_rate, 4),
        "parse_success_rate": round(parse_successes / total, 4),
        "avg_inference_time_seconds": round(avg_inference_time, 3),
        "avg_confidence": round(avg_confidence, 3),
        "avg_output_completeness": round(avg_completeness, 3),
        "rag_metrics": {
            "avg_retrieval_time_seconds": round(avg_retrieval_time, 3),
            "avg_total_time_seconds": round(avg_total_time, 3),
            "avg_retrieval_recall": round(avg_retrieval_recall, 4),
            "avg_retrieval_utilization": round(avg_retrieval_utilization, 4),
            "avg_retrieval_similarity": round(avg_retrieval_similarity, 4),
            "top_k": RAG_TOP_K,
        },
    }


def print_summary(metrics: dict):
    """Print formatted summary of RAG pipeline results."""
    print("\n" + "=" * 65)
    print("  WEEK 2 RESULTS: LLM + RAG (MITRE ATT&CK Retrieval)")
    print("=" * 65)

    print(f"\n  Total Scenarios: {metrics['total_scenarios']}")
    print(f"  Classification Accuracy: {metrics['classification_accuracy']:.1%}")
    print(f"  Parse Success Rate: {metrics['parse_success_rate']:.1%}")

    bd = metrics["binary_detection"]
    print(f"\n  --- Binary Attack Detection ---")
    print(f"  Accuracy:  {bd['accuracy']:.1%}")
    print(f"  Precision: {bd['precision']:.1%}")
    print(f"  Recall:    {bd['recall']:.1%}")
    print(f"  F1 Score:  {bd['f1_score']:.1%}")
    print(f"  TP={bd['true_positives']} TN={bd['true_negatives']} FP={bd['false_positives']} FN={bd['false_negatives']}")

    tm = metrics["technique_mapping"]
    print(f"\n  --- MITRE ATT&CK Technique Mapping ---")
    print(f"  Avg Jaccard Overlap:    {tm['avg_jaccard_overlap']:.1%}")
    print(f"  Hallucination Rate:     {tm['hallucination_rate']:.1%}")
    print(f"  Invalid Technique IDs:  {tm['total_invalid_technique_ids']} (in {tm['scenarios_with_invalid_ids']} scenarios)")

    rm = metrics["rag_metrics"]
    print(f"\n  --- RAG Performance ---")
    print(f"  Retrieval Recall:       {rm['avg_retrieval_recall']:.1%} (GT techniques found in top-{rm['top_k']})")
    print(f"  Retrieval Utilization:  {rm['avg_retrieval_utilization']:.1%} (LLM used retrieved techniques)")
    print(f"  Avg Retrieval Similarity: {rm['avg_retrieval_similarity']:.4f}")
    print(f"  Avg Retrieval Time:     {rm['avg_retrieval_time_seconds']:.3f}s")
    print(f"  Avg LLM Inference Time: {metrics['avg_inference_time_seconds']:.3f}s")
    print(f"  Avg Total Time:         {rm['avg_total_time_seconds']:.3f}s")

    print(f"\n  --- Key Comparison Metrics (vs Week 1 LLM-only) ---")
    print(f"  False Escalation Rate: {metrics['false_escalation_rate']:.1%}")
    print(f"  Technique Hallucination Rate: {tm['hallucination_rate']:.1%}")
    print(f"  Invalid ID Rate: {tm['invalid_id_rate']:.1%}  ← RAG should reduce this!")
    print(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")

    print("\n" + "=" * 65)
    print("  Compare with Week 1:")
    print("  python -m src.utils.evaluation --compare \\")
    print("    data/results/baseline_llm_results.json \\")
    print("    data/results/rag_pipeline_results.json")
    print("=" * 65 + "\n")


def print_comparison(rag_metrics: dict, baseline_path: str):
    """Side-by-side comparison with Week 1 baseline."""
    try:
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
        baseline = baseline_data.get("summary_metrics", {})
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"   [!] Could not load baseline from: {baseline_path}")
        return

    print("\n" + "=" * 70)
    print("  WEEK 1 vs WEEK 2 COMPARISON")
    print("=" * 70)

    metrics_to_compare = [
        ("Classification Accuracy", "classification_accuracy"),
        ("F1 Score", ("binary_detection", "f1_score")),
        ("Precision", ("binary_detection", "precision")),
        ("Recall", ("binary_detection", "recall")),
        ("False Escalation Rate", "false_escalation_rate"),
        ("Hallucination Rate", ("technique_mapping", "hallucination_rate")),
        ("Avg Confidence", "avg_confidence"),
        ("Parse Success Rate", "parse_success_rate"),
    ]

    print(f"\n  {'Metric':<30} {'LLM-Only':>12} {'LLM+RAG':>12} {'Change':>12}")
    print(f"  {'─'*66}")

    for label, key in metrics_to_compare:
        if isinstance(key, tuple):
            v1 = baseline.get(key[0], {}).get(key[1], "N/A")
            v2 = rag_metrics.get(key[0], {}).get(key[1], "N/A")
        else:
            v1 = baseline.get(key, "N/A")
            v2 = rag_metrics.get(key, "N/A")

        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            change = v2 - v1
            direction = "↑" if change > 0 else "↓" if change < 0 else "="
            # For hallucination/false escalation, lower is better
            if "hallucination" in label.lower() or "false" in label.lower():
                direction = "✅" if change < 0 else "⚠️" if change > 0 else "="
            else:
                direction = "✅" if change > 0 else "⚠️" if change < 0 else "="
            print(f"  {label:<30} {v1:>12.4f} {v2:>12.4f} {direction} {change:>+.4f}")
        else:
            print(f"  {label:<30} {str(v1):>12} {str(v2):>12}")

    print(f"\n  RAG-Specific Metrics (new in Week 2):")
    rm = rag_metrics.get("rag_metrics", {})
    print(f"  Retrieval Recall:       {rm.get('avg_retrieval_recall', 0):.1%}")
    print(f"  Retrieval Utilization:  {rm.get('avg_retrieval_utilization', 0):.1%}")
    print(f"  Avg Retrieval Time:     {rm.get('avg_retrieval_time_seconds', 0):.3f}s")
    print("=" * 70 + "\n")


def load_scenarios(path: Optional[str] = None) -> list:
    """Load threat scenarios from JSON file."""
    if path is None:
        path = SCENARIOS_DIR / "threat_scenarios.json"
    else:
        path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)
    return data["scenarios"]


def main():
    parser = argparse.ArgumentParser(description="Guard Agent - Week 2: RAG Pipeline")
    parser.add_argument("--scenarios", type=str, default=None, help="Path to scenarios JSON")
    parser.add_argument("--single", type=str, default=None, help="Run single scenario by ID")
    parser.add_argument("--output", type=str, default=None, help="Output results path")
    parser.add_argument("--compare", type=str, default=None, help="Path to Week 1 baseline results for comparison")
    parser.add_argument("--top-k", type=int, default=RAG_TOP_K, help="Number of RAG results")
    parser.add_argument("--dry-run", action="store_true", help="Show retrieval results without LLM")
    args = parser.parse_args()

    print("\n🔬 Guard Agent - Week 2: LLM + RAG Pipeline")
    print(f"   Model: {LLM_MODEL}")
    print(f"   Temperature: {LLM_TEMPERATURE_BASELINE}")
    print(f"   RAG Top-K: {args.top_k}")
    print(f"   Timestamp: {datetime.now().isoformat()}")

    # Load scenarios
    scenarios = load_scenarios(args.scenarios)
    print(f"   Loaded {len(scenarios)} scenarios")

    if args.single:
        scenarios = [s for s in scenarios if s["id"] == args.single]
        if not scenarios:
            print(f"   ERROR: Scenario {args.single} not found")
            return

    # Initialize RAG Knowledge Base
    print("\n   Initializing RAG Knowledge Base...")
    kb = MitreKnowledgeBase()
    kb.load()
    stats = kb.get_stats()
    print(f"   Techniques in KB: {stats.get('total_techniques_in_lookup', 0)}")
    print(f"   Documents in ChromaDB: {stats.get('collection_document_count', 0)}")

    # Dry run — show retrieval only
    if args.dry_run:
        for scenario in scenarios[:5]:
            print(f"\n   --- {scenario['id']}: {scenario['event_type']} ---")
            results = kb.query(scenario["event_description"], top_k=args.top_k)
            gt_techniques = scenario["ground_truth"]["attack_techniques"]
            retrieved_ids = [r["technique_id"] for r in results]
            
            print(f"   Ground Truth: {gt_techniques}")
            print(f"   Retrieved:    {retrieved_ids}")
            
            overlap = set(gt_techniques) & set(retrieved_ids)
            print(f"   Overlap:      {list(overlap) if overlap else 'NONE'}")
            
            for r in results:
                print(f"     {r['technique_id']} - {r['name']} (sim={r['similarity']:.3f})")
        print("\n[DRY RUN] No LLM calls made.")
        return

    # Initialize LLM
    print("\n   Initializing LLM...")
    try:
        llm = get_llm()
        print("   LLM ready.\n")
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    # Run all scenarios
    results = []
    for i, scenario in enumerate(scenarios):
        print(f"   [{i+1}/{len(scenarios)}] Processing {scenario['id']}...", end=" ", flush=True)

        result = run_rag_analysis(llm, kb, scenario, top_k=args.top_k)
        results.append(result)

        ev = result["evaluation"]
        rag = result["rag_details"]
        status = "✅" if ev["classification_correct"] else "❌"
        print(
            f"{status} Pred: {ev['predicted_classification']:<20} "
            f"GT: {ev['ground_truth_classification']:<20} "
            f"Retr: {rag['retrieval_recall']:.0%} "
            f"Time: {result['total_time_seconds']:.1f}s"
        )

    # Compute summary
    metrics = compute_summary_metrics(results)
    print_summary(metrics)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(RESULTS_DIR / "rag_pipeline_results.json")

    output_data = {
        "experiment": "Week 2 - LLM + RAG (MITRE ATT&CK)",
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE_BASELINE,
        "rag_top_k": args.top_k,
        "embedding_model": EMBEDDING_MODEL,
        "timestamp": datetime.now().isoformat(),
        "summary_metrics": metrics,
        "individual_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"   Results saved to: {output_path}")

    # Auto-compare if baseline path provided
    if args.compare:
        print_comparison(metrics, args.compare)
    else:
        # Try auto-detect baseline
        baseline_path = RESULTS_DIR / "baseline_llm_results.json"
        if baseline_path.exists():
            print(f"\n   [Auto] Found Week 1 baseline, showing comparison:")
            print_comparison(metrics, str(baseline_path))


if __name__ == "__main__":
    main()
