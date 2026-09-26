"""
Step 2: Normalize Names and Addresses
CLI wrapper and testing script for name & address normalization.
"""

import sys
from pathlib import Path

# Adjust path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import normalize_record, normalize_business_name, normalize_business_address


if __name__ == "__main__":
    test_cases = [
        {"entity_id": "S1-1", "business_name": "ABC Technologies Pvt. Ltd.", "business_address": "105 Elm St, Apt 4B, Morganton, NC 28655", "country": "US"},
        {"entity_id": "S2-1", "business_name": "abc technologies private limited", "business_address": "105 Elm Street, Morganton", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Maure Williams Colombier Inc", "business_address": "85 Wayne Avenue, Ticonderoga, NY", "country": "US"},
        {"entity_id": "S3-2", "business_name": "maurewilliamscolombier.com", "business_address": "Wayne Ave, Ticonderoga Townshiip, New York", "country": "US"},
        {"entity_id": "S1-3", "business_name": "Marina Ecole France Sarl", "business_address": "63 R. DE DIEPPE, LILLE, Hauts-de-France", "country": "France"},
        {"entity_id": "S1-4", "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड", "business_address": "KH NO. -570/13, NEW DELHI, Delhi 110001", "country": "India"}
    ]

    print("--- Testing Normalization Module (Step 2) ---")
    for tc in test_cases:
        norm = normalize_record(tc)
        print(f"\nID: {norm['entity_id']} | Country: {norm['country']}")
        print(f"  Name Norm: '{norm['name_normalized']}'")
        print(f"  No Suffix: '{norm['name_without_legal_suffix']}'")
        print(f"  Sorted Tok: '{norm['name_sorted_tokens']}'")
        print(f"  Addr Norm: '{norm['address_normalized']}'")
        print(f"  Postal: {norm['postal_tokens']} | Building: {norm['building_numbers']}")
