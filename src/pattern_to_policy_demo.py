"""
pattern_to_policy_demo.py — Shows how the pattern-detection side of the
project (waste-generation pattern analysis) hands off to the RAG system to
produce a policy-grounded recommendation, instead of RAG being a standalone
chatbot bolted onto the side.

Flow:
  1. Synthetic ward-level waste audit data (stand-in for your real pattern
     detection / anomaly model output).
  2. A simple rule flags wards below the SOP's segregation compliance
     threshold.
  3. For each flagged ward, the RAG retriever is queried for the relevant
     escalation clause, and a combined, cited recommendation is generated.
"""

import pandas as pd
from rag import WastePolicyRAG

# Stand-in for real audit data (would come from your pattern-detection
# model / sensor-bin data / municipal dashboard export in the real project)
ward_audit_data = pd.DataFrame(
    [
        {"ward": "Ward 12 - Indiranagar", "segregation_compliance_pct": 58, "audit_cycles_below_threshold": 2},
        {"ward": "Ward 45 - Whitefield", "segregation_compliance_pct": 82, "audit_cycles_below_threshold": 0},
        {"ward": "Ward 7 - Jayanagar", "segregation_compliance_pct": 69, "audit_cycles_below_threshold": 2},
        {"ward": "Ward 88 - Yelahanka", "segregation_compliance_pct": 91, "audit_cycles_below_threshold": 0},
    ]
)

COMPLIANCE_THRESHOLD = 75  # from wet_waste_segregation_sop.md, Section B.2


def flag_wards(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["segregation_compliance_pct"] < COMPLIANCE_THRESHOLD)
        & (df["audit_cycles_below_threshold"] >= 2)
    ]


def main():
    rag = WastePolicyRAG(backend="extractive")
    flagged = flag_wards(ward_audit_data)

    print(f"Pattern detection flagged {len(flagged)} of {len(ward_audit_data)} wards:\n")

    for _, row in flagged.iterrows():
        query = "What enforcement action applies when a ward's segregation compliance is below threshold for two consecutive audit cycles?"
        result = rag.answer(query, k=2)

        print(f"⚠️  {row['ward']} — {row['segregation_compliance_pct']}% compliance "
              f"({row['audit_cycles_below_threshold']} consecutive low cycles)")
        print("Policy-grounded recommendation:")
        print(result["answer"])
        print("Cited sections:", [f"{s['doc']} / {s['section']}" for s in result["sources"]])
        print("=" * 90)


if __name__ == "__main__":
    main()
