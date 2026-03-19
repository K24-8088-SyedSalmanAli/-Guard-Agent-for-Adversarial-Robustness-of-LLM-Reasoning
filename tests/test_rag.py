"""
============================================
Guard Agent - Week 2 Tests
============================================
Tests that work WITHOUT Ollama or ChromaDB running.
Validates MITRE parsing, document generation, and
knowledge base validation logic.

Run: python tests/test_rag.py
"""

import json
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mitre_attack_loader import (
    parse_technique,
    parse_tactic,
    parse_mitigation,
    build_rag_documents,
)


# ============================================
# Test MITRE ATT&CK Parsing
# ============================================

def test_parse_technique_valid():
    """Test parsing a valid STIX attack-pattern."""
    stix_obj = {
        "id": "attack-pattern--12345",
        "type": "attack-pattern",
        "name": "Spearphishing Link",
        "description": "Adversaries may send spearphishing emails with a malicious link.",
        "external_references": [
            {"source_name": "mitre-attack", "external_id": "T1566.002", "url": "https://attack.mitre.org/techniques/T1566/002"}
        ],
        "kill_chain_phases": [
            {"kill_chain_name": "mitre-attack", "phase_name": "initial-access"}
        ],
        "x_mitre_platforms": ["Windows", "Linux", "macOS"],
        "x_mitre_data_sources": ["Application Log: Application Log Content"],
        "x_mitre_detection": "Monitor for suspicious email links.",
    }

    result = parse_technique(stix_obj)
    assert result is not None
    assert result["technique_id"] == "T1566.002"
    assert result["name"] == "Spearphishing Link"
    assert result["is_subtechnique"] is True
    assert "initial-access" in result["tactic_shortnames"]
    assert "Windows" in result["platforms"]
    print("✅ test_parse_technique_valid PASSED")


def test_parse_technique_no_id():
    """Test that techniques without MITRE ID are skipped."""
    stix_obj = {
        "id": "attack-pattern--99999",
        "type": "attack-pattern",
        "name": "Unknown Technique",
        "external_references": [
            {"source_name": "other-source", "external_id": "X123"}
        ],
    }
    result = parse_technique(stix_obj)
    assert result is None
    print("✅ test_parse_technique_no_id PASSED")


def test_parse_technique_parent():
    """Test parsing a parent technique (no dot in ID)."""
    stix_obj = {
        "id": "attack-pattern--11111",
        "type": "attack-pattern",
        "name": "Phishing",
        "description": "General phishing technique.",
        "external_references": [
            {"source_name": "mitre-attack", "external_id": "T1566", "url": "https://attack.mitre.org/techniques/T1566"}
        ],
        "kill_chain_phases": [
            {"kill_chain_name": "mitre-attack", "phase_name": "initial-access"}
        ],
        "x_mitre_platforms": ["Windows"],
        "x_mitre_data_sources": [],
    }
    result = parse_technique(stix_obj)
    assert result["is_subtechnique"] is False
    assert result["technique_id"] == "T1566"
    print("✅ test_parse_technique_parent PASSED")


def test_parse_tactic():
    """Test parsing a STIX tactic."""
    stix_obj = {
        "type": "x-mitre-tactic",
        "name": "Initial Access",
        "description": "The adversary gains initial access to the network.",
        "x_mitre_shortname": "initial-access",
        "external_references": [
            {"source_name": "mitre-attack", "external_id": "TA0001"}
        ],
    }
    result = parse_tactic(stix_obj)
    assert result["name"] == "Initial Access"
    assert result["short_name"] == "initial-access"
    assert result["tactic_id"] == "TA0001"
    print("✅ test_parse_tactic PASSED")


def test_parse_mitigation():
    """Test parsing a STIX mitigation."""
    stix_obj = {
        "type": "course-of-action",
        "name": "User Training",
        "description": "Train users to identify social engineering techniques.",
        "external_references": [
            {"source_name": "mitre-attack", "external_id": "M1017"}
        ],
    }
    result = parse_mitigation(stix_obj)
    assert result["name"] == "User Training"
    assert result["mitigation_id"] == "M1017"
    print("✅ test_parse_mitigation PASSED")


def test_build_rag_documents():
    """Test RAG document generation from parsed data."""
    parsed_data = {
        "techniques": [
            {
                "technique_id": "T1566.002",
                "name": "Spearphishing Link",
                "description": "Send malicious links via email.",
                "tactics": ["Initial Access"],
                "tactic_shortnames": ["initial-access"],
                "platforms": ["Windows", "Linux"],
                "data_sources": ["Network Traffic"],
                "detection": "Monitor email links.",
                "is_subtechnique": True,
                "url": "https://attack.mitre.org/techniques/T1566/002",
                "stix_id": "attack-pattern--12345",
                "mitigations": ["User Training"],
            },
            {
                "technique_id": "T1486",
                "name": "Data Encrypted for Impact",
                "description": "Encrypt data for ransom.",
                "tactics": ["Impact"],
                "tactic_shortnames": ["impact"],
                "platforms": ["Windows"],
                "data_sources": ["File Monitoring"],
                "detection": "Monitor file encryption.",
                "is_subtechnique": False,
                "url": "https://attack.mitre.org/techniques/T1486",
                "stix_id": "attack-pattern--67890",
                "mitigations": ["Data Backup"],
            },
        ],
    }

    documents = build_rag_documents(parsed_data)
    assert len(documents) == 2

    # Check first document
    doc1 = documents[0]
    assert doc1["id"] == "T1566.002"
    assert "Spearphishing Link" in doc1["text"]
    assert "Initial Access" in doc1["text"]
    assert "User Training" in doc1["text"]
    assert doc1["metadata"]["technique_id"] == "T1566.002"
    assert doc1["metadata"]["is_subtechnique"] == True

    # Check second document
    doc2 = documents[1]
    assert doc2["id"] == "T1486"
    assert "Data Encrypted for Impact" in doc2["text"]

    print("✅ test_build_rag_documents PASSED")


def test_rag_document_format():
    """Test that RAG documents contain all required sections."""
    parsed_data = {
        "techniques": [
            {
                "technique_id": "T1078",
                "name": "Valid Accounts",
                "description": "Adversaries use legitimate credentials.",
                "tactics": ["Defense Evasion", "Persistence", "Privilege Escalation", "Initial Access"],
                "tactic_shortnames": ["defense-evasion", "persistence"],
                "platforms": ["Windows", "Linux", "macOS"],
                "data_sources": ["Logon Session"],
                "detection": "Monitor for unusual account activity.",
                "is_subtechnique": False,
                "url": "https://attack.mitre.org/techniques/T1078",
                "stix_id": "attack-pattern--00000",
                "mitigations": ["Password Policies", "Multi-factor Authentication"],
            },
        ],
    }

    documents = build_rag_documents(parsed_data)
    doc = documents[0]
    text = doc["text"]

    # Verify all required sections are present
    assert "MITRE ATT&CK Technique: T1078" in text
    assert "Tactics:" in text
    assert "Platforms:" in text
    assert "Description:" in text
    assert "Detection Methods:" in text
    assert "Recommended Mitigations:" in text
    assert "Data Sources for Detection:" in text

    # Verify content is populated
    assert "Valid Accounts" in text
    assert "Password Policies" in text
    assert "Multi-factor Authentication" in text

    print("✅ test_rag_document_format PASSED")


def test_technique_validation_logic():
    """Test technique ID validation (used by Guard Agent Week 4)."""
    # Simulate technique lookup
    technique_lookup = {
        "T1566": {"name": "Phishing", "tactics": ["Initial Access"], "is_subtechnique": False, "platforms": ["Windows"]},
        "T1566.001": {"name": "Spearphishing Attachment", "tactics": ["Initial Access"], "is_subtechnique": True, "platforms": ["Windows"]},
        "T1566.002": {"name": "Spearphishing Link", "tactics": ["Initial Access"], "is_subtechnique": True, "platforms": ["Windows"]},
        "T1486": {"name": "Data Encrypted for Impact", "tactics": ["Impact"], "is_subtechnique": False, "platforms": ["Windows"]},
    }

    # Valid technique
    assert "T1566.002" in technique_lookup
    assert technique_lookup["T1566.002"]["name"] == "Spearphishing Link"

    # Invalid technique (hallucinated by LLM)
    assert "T9999" not in technique_lookup
    assert "T1566.999" not in technique_lookup

    # Test format validation
    valid_pattern = r'^T\d{4}(\.\d{3})?$'
    assert re.match(valid_pattern, "T1566")
    assert re.match(valid_pattern, "T1566.002")
    assert not re.match(valid_pattern, "FAKE123")
    assert not re.match(valid_pattern, "T123")

    print("✅ test_technique_validation_logic PASSED")


def test_description_cleaning():
    """Test that markdown links and citations are cleaned from descriptions."""
    # Simulate STIX description with markdown
    raw_description = "Adversaries may use [Valid Accounts](https://attack.mitre.org/techniques/T1078) to access systems.(Citation: CISA 2023)"

    # Clean it
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', raw_description)
    cleaned = re.sub(r'\(Citation:[^\)]+\)', '', cleaned)

    assert "[" not in cleaned
    assert "](http" not in cleaned
    assert "(Citation:" not in cleaned
    assert "Valid Accounts" in cleaned

    print("✅ test_description_cleaning PASSED")


# ============================================
# Run all tests
# ============================================
if __name__ == "__main__":
    print("\n🧪 Running Week 2 Tests...\n")

    test_parse_technique_valid()
    test_parse_technique_no_id()
    test_parse_technique_parent()
    test_parse_tactic()
    test_parse_mitigation()
    test_build_rag_documents()
    test_rag_document_format()
    test_technique_validation_logic()
    test_description_cleaning()

    print("\n✅ All Week 2 tests passed!\n")
