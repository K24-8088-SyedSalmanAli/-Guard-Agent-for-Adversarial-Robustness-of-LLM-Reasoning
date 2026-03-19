import json
import os
import sys
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MITRE_DATA_DIR

# ============================================
# STIX Data URL (Official MITRE GitHub)
# ============================================
MITRE_STIX_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
LOCAL_STIX_PATH = MITRE_DATA_DIR / "enterprise-attack.json"


def download_mitre_data(force: bool = False) -> Path:
    """
    Download MITRE ATT&CK Enterprise STIX data from GitHub.
    Saves to data/mitre_attack/enterprise-attack.json
    """
    import urllib.request

    MITRE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_STIX_PATH.exists() and not force:
        print(f"   [✓] MITRE data already exists: {LOCAL_STIX_PATH}")
        return LOCAL_STIX_PATH

    print(f"   [↓] Downloading MITRE ATT&CK Enterprise data...")
    print(f"       Source: {MITRE_STIX_URL}")

    try:
        urllib.request.urlretrieve(MITRE_STIX_URL, str(LOCAL_STIX_PATH))
        size_mb = LOCAL_STIX_PATH.stat().st_size / (1024 * 1024)
        print(f"   [✓] Downloaded: {size_mb:.1f} MB")
        return LOCAL_STIX_PATH
    except Exception as e:
        print(f"   [✗] Download failed: {e}")
        print(f"   [!] Manual download: {MITRE_STIX_URL}")
        print(f"   [!] Save to: {LOCAL_STIX_PATH}")
        raise


def parse_stix_data(stix_path: Optional[Path] = None) -> dict:
    """
    Parse MITRE ATT&CK STIX bundle into structured data.
    
    Returns dict with:
        - techniques: list of attack techniques with full details
        - tactics: list of tactics (categories)
        - mitigations: list of mitigations
        - relationships: technique-to-tactic mappings
        - stats: count summary
    """
    if stix_path is None:
        stix_path = LOCAL_STIX_PATH

    print(f"   [⚙] Parsing STIX data from: {stix_path}")

    with open(stix_path, "r", encoding="utf-8") as f:
        stix_bundle = json.load(f)

    objects = stix_bundle.get("objects", [])
    print(f"   [⚙] Total STIX objects: {len(objects)}")

    # Categorize STIX objects
    techniques = []
    tactics = []
    mitigations = []
    relationships = []
    id_to_name = {}  # For resolving relationships

    for obj in objects:
        obj_type = obj.get("type", "")
        
        # Skip revoked or deprecated
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        if obj_type == "attack-pattern":
            technique = parse_technique(obj)
            if technique:
                techniques.append(technique)
                id_to_name[obj["id"]] = technique["technique_id"]

        elif obj_type == "x-mitre-tactic":
            tactic = parse_tactic(obj)
            if tactic:
                tactics.append(tactic)
                id_to_name[obj["id"]] = tactic["short_name"]

        elif obj_type == "course-of-action":
            mitigation = parse_mitigation(obj)
            if mitigation:
                mitigations.append(mitigation)
                id_to_name[obj["id"]] = mitigation["name"]

        elif obj_type == "relationship":
            rel = parse_relationship(obj)
            if rel:
                relationships.append(rel)

    # Resolve tactic names for techniques using kill_chain_phases
    tactic_shortname_to_name = {t["short_name"]: t["name"] for t in tactics}

    for tech in techniques:
        resolved_tactics = []
        for short_name in tech.get("tactic_shortnames", []):
            full_name = tactic_shortname_to_name.get(short_name, short_name)
            resolved_tactics.append(full_name)
        tech["tactics"] = resolved_tactics

    # Resolve mitigation relationships
    technique_mitigations = {}
    for rel in relationships:
        if rel["relationship_type"] == "mitigates":
            source_id = rel["source_ref"]
            target_id = rel["target_ref"]
            source_name = id_to_name.get(source_id, source_id)
            target_name = id_to_name.get(target_id, target_id)
            if target_name not in technique_mitigations:
                technique_mitigations[target_name] = []
            technique_mitigations[target_name].append(source_name)

    # Attach mitigations to techniques
    for tech in techniques:
        tech_id = tech["technique_id"]
        tech["mitigations"] = technique_mitigations.get(tech_id, [])

    stats = {
        "total_techniques": len(techniques),
        "total_tactics": len(tactics),
        "total_mitigations": len(mitigations),
        "total_relationships": len(relationships),
        "sub_techniques": sum(1 for t in techniques if t["is_subtechnique"]),
        "parent_techniques": sum(1 for t in techniques if not t["is_subtechnique"]),
    }

    print(f"   [✓] Parsed: {stats['parent_techniques']} techniques, "
          f"{stats['sub_techniques']} sub-techniques, "
          f"{stats['total_tactics']} tactics, "
          f"{stats['total_mitigations']} mitigations")

    return {
        "techniques": techniques,
        "tactics": tactics,
        "mitigations": mitigations,
        "relationships": relationships,
        "stats": stats,
    }


def parse_technique(obj: dict) -> Optional[dict]:
    """Parse a STIX attack-pattern into a structured technique dict."""
    external_refs = obj.get("external_references", [])
    technique_id = ""
    url = ""
    for ref in external_refs:
        if ref.get("source_name") == "mitre-attack":
            technique_id = ref.get("external_id", "")
            url = ref.get("url", "")
            break

    if not technique_id:
        return None

    # Extract tactic short names from kill chain phases
    tactic_shortnames = []
    for phase in obj.get("kill_chain_phases", []):
        if phase.get("kill_chain_name") == "mitre-attack":
            tactic_shortnames.append(phase["phase_name"])

    # Determine if sub-technique
    is_subtechnique = "." in technique_id

    # Clean description - remove markdown links and references
    description = obj.get("description", "")
    description = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', description)  # Remove markdown links
    description = re.sub(r'\(Citation:[^\)]+\)', '', description)  # Remove citations

    # Extract data sources
    data_sources = []
    for ds in obj.get("x_mitre_data_sources", []):
        data_sources.append(ds)

    # Extract platforms
    platforms = obj.get("x_mitre_platforms", [])

    # Detection info
    detection = obj.get("x_mitre_detection", "")
    detection = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', detection)
    detection = re.sub(r'\(Citation:[^\)]+\)', '', detection)

    return {
        "technique_id": technique_id,
        "name": obj.get("name", ""),
        "description": description.strip(),
        "tactic_shortnames": tactic_shortnames,
        "tactics": [],  # Resolved later
        "platforms": platforms,
        "data_sources": data_sources,
        "detection": detection.strip(),
        "is_subtechnique": is_subtechnique,
        "url": url,
        "stix_id": obj.get("id", ""),
        "mitigations": [],  # Attached later
    }


def parse_tactic(obj: dict) -> Optional[dict]:
    """Parse a STIX x-mitre-tactic into a structured tactic dict."""
    external_refs = obj.get("external_references", [])
    tactic_id = ""
    for ref in external_refs:
        if ref.get("source_name") == "mitre-attack":
            tactic_id = ref.get("external_id", "")
            break

    short_name = obj.get("x_mitre_shortname", "")
    description = obj.get("description", "")
    description = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', description)

    return {
        "tactic_id": tactic_id,
        "name": obj.get("name", ""),
        "short_name": short_name,
        "description": description.strip(),
    }


def parse_mitigation(obj: dict) -> Optional[dict]:
    """Parse a STIX course-of-action into a structured mitigation dict."""
    external_refs = obj.get("external_references", [])
    mitigation_id = ""
    for ref in external_refs:
        if ref.get("source_name") == "mitre-attack":
            mitigation_id = ref.get("external_id", "")
            break

    description = obj.get("description", "")
    description = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', description)

    return {
        "mitigation_id": mitigation_id,
        "name": obj.get("name", ""),
        "description": description.strip(),
    }


def parse_relationship(obj: dict) -> Optional[dict]:
    """Parse a STIX relationship."""
    return {
        "relationship_type": obj.get("relationship_type", ""),
        "source_ref": obj.get("source_ref", ""),
        "target_ref": obj.get("target_ref", ""),
    }


def build_rag_documents(parsed_data: dict) -> list[dict]:
    """
    Convert parsed MITRE data into RAG-ready documents.
    Each document = one technique with full context for embedding.
    
    This is the KEY function — the document format determines
    what the RAG retrieves and how useful it is for the LLM.
    """
    documents = []

    for tech in parsed_data["techniques"]:
        # Build rich text document for embedding
        tactics_str = ", ".join(tech["tactics"]) if tech["tactics"] else "Unknown"
        platforms_str = ", ".join(tech["platforms"]) if tech["platforms"] else "All"
        mitigations_str = ", ".join(tech["mitigations"]) if tech["mitigations"] else "None documented"
        data_sources_str = ", ".join(tech["data_sources"]) if tech["data_sources"] else "Not specified"

        # Full document text for embedding
        doc_text = f"""MITRE ATT&CK Technique: {tech['technique_id']} - {tech['name']}
Tactics: {tactics_str}
Platforms: {platforms_str}

Description:
{tech['description']}

Detection Methods:
{tech['detection'] if tech['detection'] else 'No specific detection guidance available.'}

Recommended Mitigations:
{mitigations_str}

Data Sources for Detection:
{data_sources_str}"""

        # Metadata for filtering and validation
        metadata = {
            "technique_id": tech["technique_id"],
            "name": tech["name"],
            "tactics": tactics_str,
            "platforms": platforms_str,
            "is_subtechnique": tech["is_subtechnique"],
            "url": tech["url"],
            "has_detection": bool(tech["detection"]),
            "has_mitigations": len(tech["mitigations"]) > 0,
            "document_type": "technique",
        }

        documents.append({
            "id": tech["technique_id"],
            "text": doc_text,
            "metadata": metadata,
        })

    print(f"   [✓] Built {len(documents)} RAG documents from MITRE ATT&CK")
    return documents


def save_parsed_data(parsed_data: dict, rag_documents: list):
    """Save parsed data and RAG documents for inspection and reuse."""
    MITRE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save full parsed data
    parsed_path = MITRE_DATA_DIR / "parsed_techniques.json"
    with open(parsed_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2, default=str)
    print(f"   [✓] Saved parsed data: {parsed_path}")

    # Save RAG documents
    rag_path = MITRE_DATA_DIR / "rag_documents.json"
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(rag_documents, f, indent=2)
    print(f"   [✓] Saved RAG documents: {rag_path}")

    # Save technique ID lookup (for Guard Agent validation in Week 4)
    technique_lookup = {}
    for tech in parsed_data["techniques"]:
        technique_lookup[tech["technique_id"]] = {
            "name": tech["name"],
            "tactics": tech["tactics"],
            "is_subtechnique": tech["is_subtechnique"],
            "platforms": tech["platforms"],
        }

    lookup_path = MITRE_DATA_DIR / "technique_lookup.json"
    with open(lookup_path, "w", encoding="utf-8") as f:
        json.dump(technique_lookup, f, indent=2)
    print(f"   [✓] Saved technique lookup ({len(technique_lookup)} entries): {lookup_path}")

    # Save tactic list
    tactic_list = {t["short_name"]: t["name"] for t in parsed_data["tactics"]}
    tactic_path = MITRE_DATA_DIR / "tactic_lookup.json"
    with open(tactic_path, "w", encoding="utf-8") as f:
        json.dump(tactic_list, f, indent=2)
    print(f"   [✓] Saved tactic lookup ({len(tactic_list)} entries): {tactic_path}")


def main():
    print("\n🔬 Guard Agent - Week 2: MITRE ATT&CK Data Loader")
    print(f"   Timestamp: {datetime.now().isoformat()}\n")

    # Step 1: Download STIX data
    stix_path = download_mitre_data()

    # Step 2: Parse STIX into structured format
    parsed_data = parse_stix_data(stix_path)

    # Step 3: Build RAG documents
    rag_documents = build_rag_documents(parsed_data)

    # Step 4: Save everything
    save_parsed_data(parsed_data, rag_documents)

    # Print summary
    stats = parsed_data["stats"]
    print(f"\n{'='*50}")
    print(f"  MITRE ATT&CK Data Summary")
    print(f"{'='*50}")
    print(f"  Parent Techniques: {stats['parent_techniques']}")
    print(f"  Sub-Techniques:    {stats['sub_techniques']}")
    print(f"  Total Techniques:  {stats['total_techniques']}")
    print(f"  Tactics:           {stats['total_tactics']}")
    print(f"  Mitigations:       {stats['total_mitigations']}")
    print(f"  RAG Documents:     {len(rag_documents)}")
    print(f"{'='*50}")

    # Print sample
    print(f"\n  Sample RAG Document (first technique):")
    print(f"  {'─'*40}")
    if rag_documents:
        sample = rag_documents[0]
        print(f"  ID: {sample['id']}")
        print(f"  Text (first 300 chars):")
        print(f"  {sample['text'][:300]}...")
    print()


if __name__ == "__main__":
    main()
