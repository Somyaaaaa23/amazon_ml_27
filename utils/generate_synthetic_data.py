#!/usr/bin/env python3
"""
Generate realistic synthetic data for Entity Resolution Challenge.
Generates:
- dataset/train/train_source1.tsv
- dataset/train/train_source2.tsv
- dataset/train/train_source3.tsv
- dataset/train/train_ground_truth.tsv
- dataset/test/test_source1.tsv
- dataset/test/test_source2.tsv
- dataset/test/test_source3.tsv

Simulates real-world noise:
1. Legal suffixes & abbreviations (Pvt, Ltd, Inc, Corp, LLC, SARL, SA)
2. Character typos and accent variations (é, è, à, ç, etc.)
3. Word swaps and token drop/add
4. Address abbreviations (Rd, St, Ave, Blvd, Marg, Nagar)
5. Landmark injection ("Near Metro Station", "Opposite City Mall")
6. Singletons (entities with 0 matches)
7. Many-to-one (multiple matches in S2/S3 for single S1)
8. Multi-country support (US, India, and unseen France in test)
"""

import os
import random
import unicodedata
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# Sample base components to generate hundreds of UNIQUE entities per country
US_PREFIXES = ["Apex", "Blue Ridge", "Vanguard", "Horizon", "Green Valley", "Summit", "Cascade", "Metro", "Pinnacle", "Quantum", "Starlight", "Silverline", "Evergreen", "Nexus", "Golden Gate", "Beacon", "Crestview", "Ironclad", "Paramount", "Velocity"]
US_SECTORS = ["Logistics", "Coffee Roasters", "Robotics", "Financial Services", "Organic Farms", "Outdoor Gear", "BioHealth Technologies", "Media & Advertising", "Construction & Engineering", "Computing Labs", "Retail", "Automotive", "Timber", "Telecommunications", "Hospitality"]
US_SUFFIXES = ["Corp", "LLC", "Inc", "Co", "Group", "Enterprises"]
US_CITIES = [("Chicago", "IL", "60607"), ("Asheville", "NC", "28801"), ("Boston", "MA", "02110"), ("New York", "NY", "10005"), ("Sacramento", "CA", "95814"), ("Denver", "CO", "80202"), ("Seattle", "WA", "98101"), ("Austin", "TX", "78701"), ("Atlanta", "GA", "30303"), ("San Jose", "CA", "95110")]

IN_PREFIXES = ["Shree Balaji", "Mahalaxmi", "TechVeda", "Sai Krupa", "Ananda", "Royal Heritage", "Maruti", "Krishna", "Venkateshwara", "Southern Star", "Ganesh", "Tata Chem", "Deepak", "Kaveri", "Himalayan", "Navkar", "Aditya", "Shiva", "Surya", "Om"]
IN_SECTORS = ["Enterprises", "Textiles", "Software Solutions", "Logistics", "Dairy & Agro", "Sweets", "Pharma", "Medical Diagnostics", "Spices & Exports", "Security Services", "Trading Co", "Electricals", "Silk Sarees", "Herbal Wellness", "Industries"]
IN_SUFFIXES = ["Pvt Ltd", "Limited", "and Sons", "Trading Co", "Agency", "Brothers"]
IN_CITIES = [("New Delhi", "Delhi", "110020"), ("Surat", "Gujarat", "395002"), ("Bengaluru", "Karnataka", "560100"), ("Thane", "Maharashtra", "421302"), ("Bareilly", "Uttar Pradesh", "243001"), ("Pune", "Maharashtra", "411001"), ("Guntur", "Andhra Pradesh", "522004"), ("Chennai", "Tamil Nadu", "600017"), ("Jaipur", "Rajasthan", "302001"), ("Mumbai", "Maharashtra", "400093")]

FR_PREFIXES = ["Saint-Honore", "Atelier Dupont", "Vignobles de Bordeaux", "Lumiere", "Provence", "Mode Parisienne", "Nouvelle Aeronautique", "Brasserie Alsacienne", "Comptoir Mediterraneen", "Alps", "Chateau", "Lafayette", "Hexagone", "Belleville", "Montmartre"]
FR_SECTORS = ["Boulangerie", "Architecture", "Vins & Spiritueux", "Electronique", "Parfums", "Haute Couture", "Ingenierie", "Gastronomie", "Logistique", "Informatique", "Artisanat", "Optique", "Transport", "Cosmetiques"]
FR_SUFFIXES = ["SARL", "SA", "SAS", "EURL", "SNC"]
FR_CITIES = [("Paris", "75008"), ("Paris", "75005"), ("Bordeaux", "33000"), ("Lyon", "69007"), ("Marseille", "13002"), ("Lille", "59000"), ("Toulouse", "31000"), ("Strasbourg", "67000"), ("Grenoble", "38000"), ("Nantes", "44000")]

def generate_unique_pool(n_samples: int, country: str):
    pool = []
    for i in range(n_samples):
        if country == "United States":
            p = random.choice(US_PREFIXES)
            s = random.choice(US_SECTORS)
            suf = random.choice(US_SUFFIXES)
            city, state, zipc = random.choice(US_CITIES)
            name = f"{p} {s} {i+1} {suf}"
            addr = f"{random.randint(100, 9999)} Commerce Way, {city}, {state} {zipc}"
            pool.append((name, addr, "United States"))
        elif country == "India":
            p = random.choice(IN_PREFIXES)
            s = random.choice(IN_SECTORS)
            suf = random.choice(IN_SUFFIXES)
            city, state, pin = random.choice(IN_CITIES)
            name = f"{p} {s} {i+1} {suf}"
            addr = f"Plot {random.randint(1, 99)}, Industrial Area, {city}, {state} {pin}"
            pool.append((name, addr, "India"))
        else: # France
            p = random.choice(FR_PREFIXES)
            s = random.choice(FR_SECTORS)
            suf = random.choice(FR_SUFFIXES)
            city, zipc = random.choice(FR_CITIES)
            name = f"{s} {p} {i+1} {suf}"
            addr = f"{random.randint(1, 150)} Rue de la Republique, {zipc} {city}"
            pool.append((name, addr, "France"))
    return pool


def perturb_name(name: str) -> str:
    res = name
    # 1. Abbreviations / Legal suffix substitutions
    subs = [
        ("Private Limited", "Pvt Ltd"), ("Pvt Ltd", "Pvt. Ltd."), ("Corporation", "Corp"),
        ("Company", "Co."), ("Incorporated", "Inc"), ("LLC", "L.L.C."), ("SARL", "S.A.R.L."),
        ("SAS", "S.A.S."), ("&", "and"), ("and", "&"), ("Enterprises", "Ent."),
        ("Solutions", "Soln"), ("International", "Intl")
    ]
    if random.random() < 0.7:
        k, v = random.choice(subs)
        if k in res:
            res = res.replace(k, v)

    # 2. Typos
    if random.random() < 0.3 and len(res) > 5:
        idx = random.randint(1, len(res) - 2)
        c = res[idx]
        if c.isalpha():
            typo_action = random.choice(["swap", "delete", "replace"])
            if typo_action == "swap":
                res = res[:idx] + res[idx+1] + res[idx] + res[idx+2:]
            elif typo_action == "delete":
                res = res[:idx] + res[idx+1:]
            else:
                res = res[:idx] + "x" + res[idx+1:]

    # 3. Token drop/swap
    tokens = res.split()
    if len(tokens) > 3 and random.random() < 0.25:
        # drop generic suffix
        if tokens[-1].lower() in ["ltd", "corp", "inc", "co", "sarl", "sa"]:
            tokens = tokens[:-1]
            res = " ".join(tokens)

    return res


def perturb_address(addr: str, country: str) -> str:
    res = addr
    # Common address word perturbations
    addr_subs = [
        ("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"), ("Boulevard", "Blvd"),
        ("Industrial Area", "Indl Area"), ("Phase", "Ph"), ("Floor", "Flr"),
        ("Suite", "Ste"), ("Opposite", "Opp"), ("Near", "Nr"), ("Rue", "R."),
        ("Place", "Pl.")
    ]
    if random.random() < 0.6:
        k, v = random.choice(addr_subs)
        if k in res:
            res = res.replace(k, v)

    # Inject Landmark for India / US
    if country == "India" and random.random() < 0.4:
        landmarks = ["Near SBI ATM, ", "Opp Metro Pillar 104, ", "Behind Bus Depot, ", "Adj HP Petrol Pump, "]
        res = random.choice(landmarks) + res
    elif country == "United States" and random.random() < 0.3:
        landmarks = ["Suite 200, ", "Building B, ", "Corner of 5th Ave, "]
        res = res + ", " + random.choice(landmarks)

    # Drop PIN/Postal code occasionally
    if random.random() < 0.2:
        parts = res.split()
        if parts and parts[-1].isdigit():
            res = " ".join(parts[:-1])

    return res


def build_dataset(base_list, s1_start, s2_start, s3_start, is_train=True):
    s1_records = []
    s2_records = []
    s3_records = []
    ground_truth = {}

    s1_id_counter = s1_start
    s2_id_counter = s2_start
    s3_id_counter = s3_start

    for name, addr, country in base_list:
        s1_id = f"S1-{s1_id_counter:05d}"
        s1_id_counter += 1

        # S1 record (clean reference)
        s1_records.append({
            "entity_id": s1_id,
            "business_name": name,
            "business_address": addr,
            "country": country
        })

        # Match configuration:
        # 25% singletons (0 matches)
        # 45% 1 match (either S2 or S3)
        # 30% multi-match (S2 and S3, or 2 from S2)
        match_type = random.choices(["singleton", "single_match", "multi_match"], weights=[0.25, 0.45, 0.30])[0]

        matched_ids = []

        if match_type == "single_match":
            source = random.choice(["S2", "S3"])
            if source == "S2":
                s2_id = f"S2-{s2_id_counter:05d}"
                s2_id_counter += 1
                s2_records.append({
                    "entity_id": s2_id,
                    "business_name": perturb_name(name),
                    "business_address": perturb_address(addr, country),
                    "country": country
                })
                matched_ids.append(s2_id)
            else:
                s3_id = f"S3-{s3_id_counter:05d}"
                s3_id_counter += 1
                s3_records.append({
                    "entity_id": s3_id,
                    "business_name": perturb_name(name),
                    "business_address": perturb_address(addr, country),
                    "country": country
                })
                matched_ids.append(s3_id)

        elif match_type == "multi_match":
            # Add one S2
            s2_id = f"S2-{s2_id_counter:05d}"
            s2_id_counter += 1
            s2_records.append({
                "entity_id": s2_id,
                "business_name": perturb_name(name),
                "business_address": perturb_address(addr, country),
                "country": country
            })
            matched_ids.append(s2_id)

            # Add one S3
            s3_id = f"S3-{s3_id_counter:05d}"
            s3_id_counter += 1
            s3_records.append({
                "entity_id": s3_id,
                "business_name": perturb_name(name),
                "business_address": perturb_address(addr, country),
                "country": country
            })
            matched_ids.append(s3_id)

        ground_truth[s1_id] = matched_ids

    # Also add some distractor / unlinked records in S2 and S3 (noise in sources)
    for i in range(len(base_list) // 3):
        # S2 noise
        s2_id = f"S2-{s2_id_counter:05d}"
        s2_id_counter += 1
        name, addr, country = random.choice(base_list)
        s2_records.append({
            "entity_id": s2_id,
            "business_name": "Independent " + perturb_name(name),
            "business_address": perturb_address(addr, country),
            "country": country
        })

        # S3 noise
        s3_id = f"S3-{s3_id_counter:05d}"
        s3_id_counter += 1
        name, addr, country = random.choice(base_list)
        s3_records.append({
            "entity_id": s3_id,
            "business_name": perturb_name(name) + " Branch",
            "business_address": perturb_address(addr, country),
            "country": country
        })

    # Shuffle S2 and S3
    random.shuffle(s2_records)
    random.shuffle(s3_records)

    return (
        pd.DataFrame(s1_records),
        pd.DataFrame(s2_records),
        pd.DataFrame(s3_records),
        ground_truth,
        s1_id_counter,
        s2_id_counter,
        s3_id_counter
    )


def generate_all():
    os.makedirs("dataset/train", exist_ok=True)
    os.makedirs("dataset/test", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Generate hundreds of distinct unique business entities
    train_pool = generate_unique_pool(150, "United States") + generate_unique_pool(150, "India")
    random.shuffle(train_pool)

    # Test pool includes unseen France
    test_pool = generate_unique_pool(100, "United States") + generate_unique_pool(100, "India") + generate_unique_pool(100, "France")
    random.shuffle(test_pool)

    print(f"Generating synthetic train dataset ({len(train_pool)} S1 entities)...")
    df_s1_tr, df_s2_tr, df_s3_tr, gt_tr, s1_c, s2_c, s3_c = build_dataset(train_pool, 1, 1, 1, is_train=True)

    df_s1_tr.to_csv("dataset/train/train_source1.tsv", sep="\t", index=False)
    df_s2_tr.to_csv("dataset/train/train_source2.tsv", sep="\t", index=False)
    df_s3_tr.to_csv("dataset/train/train_source3.tsv", sep="\t", index=False)

    # Format ground truth TSV: source1_entity_id, matched_entity_ids (comma separated)
    gt_rows = []
    for s1_id, match_ids in gt_tr.items():
        gt_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(match_ids)
        })
    df_gt_tr = pd.DataFrame(gt_rows)
    df_gt_tr.to_csv("dataset/train/train_ground_truth.tsv", sep="\t", index=False)

    print(f"Generating synthetic test dataset ({len(test_pool)} S1 entities with France)...")
    df_s1_te, df_s2_te, df_s3_te, gt_te, _, _, _ = build_dataset(test_pool, s1_c, s2_c, s3_c, is_train=False)

    df_s1_te.to_csv("dataset/test/test_source1.tsv", sep="\t", index=False)
    df_s2_te.to_csv("dataset/test/test_source2.tsv", sep="\t", index=False)
    df_s3_te.to_csv("dataset/test/test_source3.tsv", sep="\t", index=False)

    print("Synthetic data generation complete!")
    print(f"Train files in dataset/train/: {len(df_s1_tr)} S1, {len(df_s2_tr)} S2, {len(df_s3_tr)} S3")
    print(f"Test files in dataset/test/: {len(df_s1_te)} S1, {len(df_s2_te)} S2, {len(df_s3_te)} S3")


if __name__ == "__main__":
    generate_all()
