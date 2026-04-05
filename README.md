# Project Progress -- Aspect 1: Guard Agent for Adversarial Robustness of LLM-Based Reasoning

## Author

Syed Salman Ali

## Project Title

Guard Agent for Adversarial Robustness of LLM-Based Reasoning

## Problem Statement

LLMs are vulnerable to:

- Prompt injection
- Data poisoning via RAG
- False-positive escalation

## Proposed Contribution

Design an **AgentShield Guard Agent** that:

- Detects conflicting threat narratives
- Performs cross-validation against MITRE ATT&CK graph
- Rejects low-consistency reasoning outputs

## Experiment Design

- Inject adversarial threat descriptions
- Compare: LLM-only reasoning vs LLM + Guard Agent

## Metrics

- Reasoning consistency score
- False escalation rate
- Hallucination rate

## Progress Summary

- Repository created
- Initial documentation added
- Change request process tested

## Current Status

🟢 On Track

## Next Steps

- Implement AgentShield Guard Agent prototype
- Set up adversarial test scenarios
- Integrate MITRE ATT&CK graph validation

# Guard Agent for Adversarial Robustness of LLM-Based Reasoning

## Blockchain-Integrated Multi-Agent Framework — Aspect 01

### Research: Reasoning Trust Score (RTS) Model

**RTS(O) = α·C(O) + β·V(O) + γ·S(O)**

---

## Project Structure

```
guard-agent-project/
├── config/
│   └── settings.py              # All configuration constants
├── data/
│   ├── scenarios/
│   │   └── threat_scenarios.json # 30 threat scenarios (Week 1)
│   └── results/                  # Experiment outputs
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── baseline_llm.py      # Week 1: LLM-only baseline
│   │   ├── rag_pipeline.py      # Week 2: RAG + MITRE ATT&CK
│   │   ├── guard_consistency.py # Week 3: C(O) module
│   │   ├── guard_validation.py  # Week 4: V(O) module
│   │   ├── guard_stability.py   # Week 5: S(O) module
│   │   └── guard_agent.py       # Week 5: Full RTS integration
│   └── utils/
│       ├── __init__.py
│       ├── prompt_templates.py   # Threat analysis prompts
│       ├── output_parser.py      # Structured output extraction
│       └── evaluation.py         # Metrics & logging
├── tests/
│   └── test_baseline.py
├── notebooks/
│   └── week1_analysis.ipynb
├── requirements.txt
├── setup_environment.sh
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

### 2. Install Ollama + Pull Llama 3

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3:8b
```

### 3. Run LLM-Only Baseline (Week 1)

```bash
python -m src.agents.baseline_llm
```

### 4. Evaluate Results

```bash
python -m src.utils.evaluation --results data/results/baseline_llm_results.json
```
