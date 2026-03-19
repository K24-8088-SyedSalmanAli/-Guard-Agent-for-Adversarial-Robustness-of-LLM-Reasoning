import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    MITRE_DATA_DIR, CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR, RAG_TOP_K, EMBEDDING_MODEL
)

# ============================================
# ChromaDB + Embeddings Setup
# ============================================
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[WARNING] chromadb not installed. Install: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[WARNING] sentence-transformers not installed. Install: pip install sentence-transformers")


class MitreKnowledgeBase:
    """
    ChromaDB-backed vector store for MITRE ATT&CK techniques.
    Provides semantic search over the full ATT&CK knowledge base.
    """

    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.technique_lookup = {}

        # Load technique lookup for validation (used by Guard Agent Week 4)
        lookup_path = MITRE_DATA_DIR / "technique_lookup.json"
        if lookup_path.exists():
            with open(lookup_path, "r") as f:
                self.technique_lookup = json.load(f)

    def _init_chromadb(self):
        """Initialize ChromaDB client."""
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb")

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        print(f"   [✓] ChromaDB initialized at: {self.persist_dir}")

    def _init_embeddings(self):
        """Initialize the sentence transformer embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")

        print(f"   [⚙] Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"   [✓] Embedding model loaded (dim={self.embedding_model.get_sentence_embedding_dimension()})")

    def build(self, force: bool = False):
        """
        Build the ChromaDB collection from RAG documents.
        
        Steps:
        1. Load RAG documents (from mitre_attack_loader output)
        2. Generate embeddings using sentence-transformers
        3. Store in ChromaDB with metadata
        """
        self._init_chromadb()
        self._init_embeddings()

        # Check if collection already exists
        existing_collections = [c.name for c in self.client.list_collections()]
        if self.collection_name in existing_collections:
            if force:
                print(f"   [⚙] Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                self.collection = self.client.get_collection(self.collection_name)
                count = self.collection.count()
                print(f"   [✓] Collection already exists with {count} documents")
                print(f"   [!] Use --force to rebuild")
                return

        # Load RAG documents
        rag_docs_path = MITRE_DATA_DIR / "rag_documents.json"
        if not rag_docs_path.exists():
            raise FileNotFoundError(
                f"RAG documents not found at {rag_docs_path}. "
                "Run mitre_attack_loader first: python -m src.agents.mitre_attack_loader"
            )

        with open(rag_docs_path, "r") as f:
            rag_documents = json.load(f)

        print(f"   [⚙] Building ChromaDB collection from {len(rag_documents)} documents...")

        # Create collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "MITRE ATT&CK Enterprise techniques for RAG"}
        )

        # Process in batches (ChromaDB has batch limits)
        batch_size = 100
        total_added = 0

        for i in range(0, len(rag_documents), batch_size):
            batch = rag_documents[i:i + batch_size]

            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]

            # Convert metadata booleans to strings (ChromaDB requirement)
            for meta in metadatas:
                for key, value in meta.items():
                    if isinstance(value, bool):
                        meta[key] = str(value)

            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            total_added += len(batch)
            print(f"   [⚙] Added {total_added}/{len(rag_documents)} documents...", end="\r")

        print(f"\n   [✓] ChromaDB collection built: {total_added} documents indexed")
        print(f"   [✓] Collection: {self.collection_name}")

    def load(self):
        """Load existing ChromaDB collection for querying."""
        if self.collection is not None:
            return

        self._init_chromadb()
        self._init_embeddings()

        try:
            self.collection = self.client.get_collection(self.collection_name)
            count = self.collection.count()
            print(f"   [✓] Loaded collection: {self.collection_name} ({count} documents)")
        except Exception as e:
            raise RuntimeError(
                f"Collection '{self.collection_name}' not found. "
                "Build it first: python -m src.agents.rag_knowledge_base --build"
            ) from e

    # ============================================
    # KEYWORD MAPPING: Business Language → ATT&CK Language
    # This bridges the semantic gap between supply chain
    # event descriptions and MITRE ATT&CK terminology.
    # ============================================
    THREAT_KEYWORD_MAP = {
        # BEC / Email threats
        "bank account change": "phishing spearphishing email spoofing financial theft",
        "payment redirect": "phishing spearphishing financial theft wire transfer fraud",
        "invoice fraud": "data manipulation stored data manipulation financial theft",
        "duplicate invoice": "data manipulation stored data manipulation",
        "email impersonation": "phishing spearphishing email spoofing internal spearphishing",
        "ceo fraud": "phishing spearphishing email spoofing business email compromise",
        "wire transfer": "financial theft phishing spearphishing",
        "urgency language": "phishing social engineering spearphishing",
        "domain spoofing": "email spoofing phishing spearphishing link",
        "lookalike domain": "email spoofing phishing spearphishing link",
        # Data tampering
        "data tampering": "data manipulation stored data manipulation",
        "record modification": "data manipulation stored data manipulation",
        "unauthorized edit": "data manipulation valid accounts stored data manipulation",
        "purity change": "data manipulation stored data manipulation",
        "post-approval modification": "data manipulation stored data manipulation",
        "audit log": "indicator removal data manipulation",
        "quality certification": "data manipulation stored data manipulation",
        # Network intrusion
        "port scan": "network service discovery active scanning reconnaissance",
        "scada": "remote services external remote services industrial control systems",
        "modbus": "remote services industrial control systems",
        "brute force": "brute force password guessing credential access",
        "ssh login": "brute force remote services valid accounts",
        "vpn compromise": "external remote services valid accounts",
        "data exfiltration": "exfiltration over alternative protocol automated exfiltration",
        "tor exit node": "proxy multi-hop proxy exfiltration",
        "lateral movement": "remote services lateral movement pass the hash",
        "credential compromise": "valid accounts credential access",
        "reconnaissance": "network service discovery active scanning",
        "firmware update": "firmware corruption system firmware",
        # Insider threat
        "insider threat": "valid accounts data from local system insider",
        "usb transfer": "exfiltration over physical medium removable media",
        "resignation": "data from local system exfiltration valid accounts insider",
        "unauthorized access": "valid accounts abuse elevation control mechanism",
        "off-hours access": "valid accounts data from local system",
        "privilege escalation": "abuse elevation control mechanism valid accounts",
        "shell company": "financial theft data manipulation",
        # Ransomware
        "ransomware": "data encrypted for impact inhibit system recovery",
        "encrypted files": "data encrypted for impact",
        "lockbit": "data encrypted for impact ransomware",
        "ransom note": "data encrypted for impact",
        # Phishing
        "phishing": "phishing spearphishing attachment spearphishing link",
        "credential harvest": "phishing credential access input capture",
        "fake login": "phishing credential access input capture",
        # DDoS
        "ddos": "network denial of service endpoint denial of service",
        "traffic spike": "network denial of service endpoint denial of service",
        "botnet": "network denial of service endpoint denial of service",
        # Supply chain
        "supply chain": "supply chain compromise trusted relationship",
        "vendor compromise": "trusted relationship supply chain compromise",
        "multi-vector": "external remote services data manipulation network denial of service",
        "coordinated attack": "external remote services data manipulation valid accounts",
        # General
        "new supplier": "trusted relationship supply chain compromise",
        "api access token": "valid accounts application access token",
        "service account": "valid accounts",
        "remote desktop": "remote desktop protocol remote services",
    }

    def _extract_security_keywords(self, text: str) -> list[str]:
        """
        Extract cybersecurity-relevant keywords from business-level text.
        Maps supply chain language to MITRE ATT&CK terminology.
        """
        text_lower = text.lower()
        extracted_terms = []

        for business_term, attack_terms in self.THREAT_KEYWORD_MAP.items():
            if business_term in text_lower:
                extracted_terms.append(attack_terms)

        return extracted_terms

    def _build_enriched_queries(self, query_text: str) -> list[str]:
        """
        Build multiple query variations for better retrieval.
        
        Strategy:
        1. Original text (truncated) — catches direct semantic matches
        2. Extracted security keywords — bridges business→ATT&CK gap
        3. Combined focused query — best of both
        """
        queries = []

        # Query 1: Truncated original (first 500 chars — too long hurts embedding quality)
        truncated = query_text[:500]
        queries.append(truncated)

        # Query 2: Extracted cybersecurity keywords
        keyword_terms = self._extract_security_keywords(query_text)
        if keyword_terms:
            keyword_query = " ".join(keyword_terms)
            queries.append(keyword_query)

        # Query 3: Focused combination — first sentence + keywords
        first_sentence = query_text.split(".")[0] if "." in query_text else query_text[:200]
        if keyword_terms:
            focused_query = f"{first_sentence}. Attack techniques: {' '.join(keyword_terms[:3])}"
            queries.append(focused_query)

        return queries

    def _single_query(self, query_text: str, top_k: int) -> list[dict]:
        """Execute a single semantic search query against ChromaDB."""
        query_embedding = self.embedding_model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            technique_id = results["ids"][0][i]
            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            similarity = 1.0 / (1.0 + distance)

            formatted.append({
                "technique_id": technique_id,
                "name": metadata.get("name", ""),
                "tactics": metadata.get("tactics", ""),
                "document": document,
                "similarity": round(similarity, 4),
                "distance": round(distance, 4),
                "metadata": metadata,
            })

        return formatted

    def query(self, query_text: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Multi-query semantic search for relevant MITRE ATT&CK techniques.
        
        Uses enriched queries (original + keyword extraction + focused)
        to bridge the gap between business-level event descriptions
        and technical ATT&CK terminology.
        
        Args:
            query_text: Natural language description of the threat/event
            top_k: Number of final results to return (default from config)
            
        Returns:
            List of dicts with technique info, relevance score, and full text
        """
        if self.collection is None:
            self.load()

        k = top_k or RAG_TOP_K

        # Build enriched queries
        queries = self._build_enriched_queries(query_text)

        # Run all queries and collect results
        all_results = {}  # technique_id → best result
        for q in queries:
            results = self._single_query(q, top_k=k)
            for r in results:
                tid = r["technique_id"]
                # Keep the result with highest similarity
                if tid not in all_results or r["similarity"] > all_results[tid]["similarity"]:
                    all_results[tid] = r

        # Sort by similarity and return top-k
        sorted_results = sorted(all_results.values(), key=lambda x: x["similarity"], reverse=True)
        return sorted_results[:k]

    def format_context_for_llm(self, results: list[dict]) -> str:
        """
        Format retrieved techniques into context string for the LLM prompt.
        This is injected into the RAG_THREAT_ANALYSIS_PROMPT.
        """
        if not results:
            return "No relevant MITRE ATT&CK techniques found."

        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"--- Retrieved Technique {i} (Relevance: {r['similarity']:.2f}) ---\n"
                f"{r['document']}\n"
            )

        return "\n".join(context_parts)

    def validate_technique_id(self, technique_id: str) -> dict:
        """
        Validate whether a technique ID exists in MITRE ATT&CK.
        Used by Guard Agent (Week 4) for V(O) validation score.
        
        Returns:
            {
                "exists": bool,
                "name": str or None,
                "tactics": list or None,
            }
        """
        if technique_id in self.technique_lookup:
            info = self.technique_lookup[technique_id]
            return {
                "exists": True,
                "name": info["name"],
                "tactics": info["tactics"],
            }
        return {"exists": False, "name": None, "tactics": None}

    def get_all_technique_ids(self) -> set:
        """Return set of all valid MITRE ATT&CK technique IDs."""
        return set(self.technique_lookup.keys())

    def get_stats(self) -> dict:
        """Return knowledge base statistics."""
        stats = {
            "total_techniques_in_lookup": len(self.technique_lookup),
            "chromadb_persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
            "embedding_model": EMBEDDING_MODEL,
        }
        if self.collection:
            stats["collection_document_count"] = self.collection.count()
        return stats


def main():
    parser = argparse.ArgumentParser(description="Guard Agent - RAG Knowledge Base")
    parser.add_argument("--build", action="store_true", help="Build ChromaDB collection")
    parser.add_argument("--force", action="store_true", help="Force rebuild (delete existing)")
    parser.add_argument("--query", type=str, help="Test query against knowledge base")
    parser.add_argument("--top-k", type=int, default=RAG_TOP_K, help="Number of results")
    parser.add_argument("--stats", action="store_true", help="Show knowledge base stats")
    args = parser.parse_args()

    print("\n🔬 Guard Agent - Week 2: RAG Knowledge Base")
    print(f"   Timestamp: {datetime.now().isoformat()}\n")

    kb = MitreKnowledgeBase()

    if args.build:
        kb.build(force=args.force)

    elif args.query:
        kb.load()
        print(f"\n   Query: '{args.query}'")
        print(f"   Top-K: {args.top_k}")
        print(f"   {'─'*50}")

        results = kb.query(args.query, top_k=args.top_k)

        for i, r in enumerate(results, 1):
            print(f"\n   Result {i}: {r['technique_id']} - {r['name']}")
            print(f"   Tactics: {r['tactics']}")
            print(f"   Similarity: {r['similarity']:.4f} (distance: {r['distance']:.4f})")
            print(f"   {'─'*50}")

        # Also show formatted context
        context = kb.format_context_for_llm(results)
        print(f"\n   === FORMATTED CONTEXT FOR LLM ===")
        print(f"   (This gets injected into the RAG prompt)")
        print(f"   {'─'*50}")
        print(context[:1500] + "..." if len(context) > 1500 else context)

    elif args.stats:
        kb.load()
        stats = kb.get_stats()
        print("   Knowledge Base Stats:")
        for k, v in stats.items():
            print(f"   {k}: {v}")

    else:
        print("   Usage:")
        print("   python -m src.agents.rag_knowledge_base --build")
        print("   python -m src.agents.rag_knowledge_base --build --force")
        print("   python -m src.agents.rag_knowledge_base --query 'phishing email attack'")
        print("   python -m src.agents.rag_knowledge_base --stats")


if __name__ == "__main__":
    main()