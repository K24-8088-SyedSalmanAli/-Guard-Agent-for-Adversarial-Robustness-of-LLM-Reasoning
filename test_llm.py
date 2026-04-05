import ollama

r = ollama.chat(
    model='llama3:8b',
    messages=[
        {"role": "system", "content": "You are a cybersecurity analyst. Always respond with ONLY valid JSON."},
        {"role": "user", "content": """Analyze this: Email from techparts-inc.com requesting bank account change for payment.

Respond with ONLY this JSON:
{"threat_classification": "BEC Payment Fraud", "severity_level": 5, "confidence": 0.9, "mitre_attack_techniques": [], "detected_indicators": [], "reasoning_chain": "test", "recommended_actions": [], "false_positive_assessment": "test"}"""}
    ],
    options={"temperature": 0.7}
)

print("=== RAW OUTPUT ===")
print(r['message']['content'][:1500])