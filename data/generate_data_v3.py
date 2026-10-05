"""
V3 synthetic data generator: bigger still.
- 8 diagnosis categories (added: CKD, hypothyroidism, osteoarthritis, GERD),
  2 confusable patients per category = 16 patients total.
- 5 repeated draws of the same lab test per patient (longer history).
"""
import json
import os
import random

random.seed(7)

OUT_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(OUT_DIR, exist_ok=True)

PROVIDERS = [
    "Dr. Mona Farid", "Dr. Youssef Nabil", "Dr. Laila Hassan", "Dr. Omar Sameh",
    "Dr. Nourhan Adel", "Dr. Karim Fathy", "Dr. Salma Reda", "Dr. Tamer Aziz",
    "Dr. Rania Kamal", "Dr. Hassan Ibrahim", "Dr. Dina Shawky", "Dr. Ziad Mostafa",
]

CATEGORIES = [
    {"name": "coronary_artery_disease", "specialty": "Cardiology",
     "test_name": "Lipid Panel", "unit_field": "LDL", "med": "Atorvastatin",
     "med_doses": ["10mg", "20mg", "40mg"], "symptom": "chest tightness on exertion"},
    {"name": "type_2_diabetes", "specialty": "Endocrinology",
     "test_name": "HbA1c", "unit_field": "HbA1c", "med": "Metformin",
     "med_doses": ["500mg", "1000mg", "1500mg"], "symptom": "increased thirst and fatigue"},
    {"name": "hypertension", "specialty": "Internal Medicine",
     "test_name": "Ambulatory Blood Pressure", "unit_field": "systolic_bp", "med": "Amlodipine",
     "med_doses": ["5mg", "10mg", "10mg"], "symptom": "intermittent headaches and dizziness"},
    {"name": "asthma", "specialty": "Pulmonology",
     "test_name": "Spirometry (FEV1 %predicted)", "unit_field": "FEV1_pct",
     "med": "Fluticasone/Salmeterol inhaler", "med_doses": ["low-dose", "medium-dose", "medium-dose"],
     "symptom": "shortness of breath and wheeze with exercise"},
    {"name": "chronic_kidney_disease", "specialty": "Nephrology",
     "test_name": "eGFR Panel", "unit_field": "eGFR", "med": "Lisinopril",
     "med_doses": ["5mg", "10mg", "20mg"], "symptom": "swelling in the ankles and fatigue"},
    {"name": "hypothyroidism", "specialty": "Endocrinology",
     "test_name": "Thyroid Function (TSH)", "unit_field": "TSH", "med": "Levothyroxine",
     "med_doses": ["50mcg", "75mcg", "100mcg"], "symptom": "unexplained weight gain and cold intolerance"},
    {"name": "osteoarthritis", "specialty": "Orthopedics",
     "test_name": "Knee X-ray Grading", "unit_field": "KL_grade", "med": "Meloxicam",
     "med_doses": ["7.5mg", "15mg", "15mg"], "symptom": "chronic knee pain worse with activity"},
    {"name": "gerd", "specialty": "Gastroenterology",
     "test_name": "Endoscopy Severity Grading", "unit_field": "LA_grade", "med": "Omeprazole",
     "med_doses": ["20mg", "40mg", "40mg"], "symptom": "frequent heartburn and regurgitation"},
]


def gen_lab_series(cat, n=5):
    uf = cat["unit_field"]
    if uf == "LDL":
        start = random.randint(150, 195)
        vals = [max(70, start - i * random.randint(8, 16)) for i in range(n)]
        return [f"{v} mg/dL" for v in vals]
    if uf == "HbA1c":
        start = round(random.uniform(7.8, 9.4), 1)
        vals = [round(max(5.5, start - i * random.uniform(0.3, 0.7)), 1) for i in range(n)]
        return [f"{v}%" for v in vals]
    if uf == "systolic_bp":
        start = random.randint(150, 172)
        vals = [max(115, start - i * random.randint(5, 10)) for i in range(n)]
        return [f"{v}/{v-52} mmHg" for v in vals]
    if uf == "FEV1_pct":
        start = random.randint(55, 68)
        vals = [min(98, start + i * random.randint(4, 8)) for i in range(n)]
        return [f"{v}% predicted" for v in vals]
    if uf == "eGFR":
        start = random.randint(35, 50)
        vals = [min(75, start + i * random.randint(2, 6)) for i in range(n)]
        return [f"{v} mL/min/1.73m2" for v in vals]
    if uf == "TSH":
        start = round(random.uniform(6.5, 9.5), 1)
        vals = [round(max(1.5, start - i * random.uniform(0.8, 1.5)), 1) for i in range(n)]
        return [f"{v} mIU/L" for v in vals]
    if uf == "KL_grade":
        grades = ["Grade 3 (severe)", "Grade 3 (severe)", "Grade 2 (moderate)", "Grade 2 (moderate)", "Grade 2 (moderate, stable)"]
        return grades[:n]
    if uf == "LA_grade":
        grades = ["LA Grade C", "LA Grade B", "LA Grade B", "LA Grade A", "LA Grade A (healed)"]
        return grades[:n]


DATE_POOL = [f"202{y}-{m:02d}-{d:02d}" for y in (1, 2, 3) for m in (2, 5, 8, 11) for d in (10, 20)]

manifest, qa_pairs = [], []
pid_counter = 3001
for cat in CATEGORIES:
    for rep in range(2):
        patient_id = f"PAT-{pid_counter}"
        provider = PROVIDERS[pid_counter % len(PROVIDERS)]
        dates = sorted(random.sample(DATE_POOL, k=5))
        lab_strs = gen_lab_series(cat, n=5)
        dose = random.choice(cat["med_doses"])

        lines = [f"## Encounter -- {dates[0]}",
                 f"Patient {patient_id} presented to {provider} ({cat['specialty']}) with {cat['symptom']}. "
                 f"Assessment: suspected {cat['name'].replace('_',' ')}. "
                 f"Plan: order {cat['test_name']}, start {cat['med']} {dose}, follow-up in 6 weeks.", "",
                 f"## Prescription -- {dates[0]}",
                 f"Prescribed by {provider}: {cat['med']} {dose}, ongoing, for management of {cat['name'].replace('_',' ')}.", ""]

        for i, d in enumerate(dates):
            lines.append(f"## Laboratory Result -- {d}")
            lines.append(
                f"Lab order: {cat['test_name']}, ordered by {provider} for {patient_id}. "
                f"Result summary: {cat['unit_field'].replace('_',' ')} = {lab_strs[i]}. "
                f"Report status: final. Report filed as document LAB-{patient_id}-{i} in private storage."
            )
            lines.append("")
            if i < len(dates) - 1:
                lines.append(f"## Encounter -- {dates[i+1]}")
                lines.append(
                    f"Follow-up with {provider}. Patient {'reports improvement' if i > 0 else 'tolerating therapy'} "
                    f"on {cat['med']} {dose}. Assessment: {cat['name'].replace('_',' ')}, "
                    f"{'improving' if i < len(dates)-2 else 'well controlled'}. "
                    f"Plan: continue current regimen, repeat {cat['test_name']} at next visit."
                )
                lines.append("")

        text = "\n".join(lines).strip()
        fname = f"{patient_id}_timeline.txt"
        with open(os.path.join(OUT_DIR, fname), "w") as f:
            f.write(text)

        manifest.append({"document_id": f"DOC-{patient_id}", "patient_id": patient_id,
                          "provider": provider, "specialty": cat["specialty"],
                          "consent_scope": "self", "document_type": "medical_timeline", "file": fname})

        # pick an UNRELATED category (different test entirely) for the
        # unanswerable/grounding question about this patient
        other_cats = [c for c in CATEGORIES if c["name"] != cat["name"]]
        unrelated = random.choice(other_cats)

        qa_pairs.append({
            "patient_id": patient_id, "category": cat["name"], "test_name": cat["test_name"],
            "unit_field": cat["unit_field"], "most_recent_value": lab_strs[-1],
            "most_recent_date": dates[-1], "first_value": lab_strs[0],
            "med": cat["med"], "dose": dose,
            "unrelated_test_name": unrelated["test_name"], "unrelated_unit_field": unrelated["unit_field"],
        })
        pid_counter += 1

with open(os.path.join(os.path.dirname(__file__), "documents_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
with open(os.path.join(os.path.dirname(__file__), "qa_ground_truth.json"), "w") as f:
    json.dump(qa_pairs, f, indent=2)

print(f"Generated {len(manifest)} patients across {len(CATEGORIES)} categories ({len(CATEGORIES)*2} confusable pairs)")
