"""Reference catalogs for the synthetic ecosystem.

Every drug name here is invented. Therapeutic classes are real-world categories
because the *shape* of the benefit design depends on them (specialty drugs carry
coinsurance, generics carry flat copays), but no real product, price, or
formulary is reproduced. See docs/limitations.md.
"""

from __future__ import annotations

import random

from careweave.domain.enums import DrugTier, PharmacyType
from careweave.domain.models import Drug, FormularyEntry, Pharmacy, Plan, Prescriber

FORMULARY_ID = "FRM-STD-2026"

# (name, generic_name, class, specialty, generic_available, list_price_30d)
_DRUG_SPECS: list[tuple[str, str | None, str, bool, bool, float]] = [
    ("Cardilex", "amlopranol", "cardiovascular", False, True, 42.0),
    ("Statorin", "rosuvastin", "cardiovascular", False, True, 38.0),
    ("Glucovane", "metfornide", "endocrine", False, True, 29.0),
    ("Insulara Pen", None, "endocrine", False, False, 410.0),
    ("Thyrolan", "levothyrox", "endocrine", False, True, 24.0),
    ("Pulmasyn", None, "respiratory", False, False, 385.0),
    ("Bronchivent", "albutamol", "respiratory", False, True, 55.0),
    ("Neurovant", None, "neurology", False, False, 310.0),
    ("Seramil", "sertrapine", "behavioral_health", False, True, 33.0),
    ("Calmivex", None, "behavioral_health", False, False, 240.0),
    ("Rheumatane", None, "immunology", True, False, 4850.0),
    ("Immunocel", None, "immunology", True, False, 6200.0),
    ("Oncatrex", None, "oncology", True, False, 9100.0),
    ("Hepavir", None, "infectious_disease", True, False, 7400.0),
    ("Amoxinex", "amoxicilan", "infectious_disease", False, True, 18.0),
    ("Gastrolyte", "omepradine", "gastroenterology", False, True, 26.0),
    ("Dermacort", "hydrocortane", "dermatology", False, True, 21.0),
    ("Osteoflex", None, "musculoskeletal", False, False, 275.0),
    ("Renaprove", None, "nephrology", True, False, 3900.0),
    ("Migraseal", None, "neurology", False, False, 295.0),
]

#: Chronic classes generate repeating fill cycles; acute ones do not.
CHRONIC_CLASSES = {
    "cardiovascular",
    "endocrine",
    "respiratory",
    "behavioral_health",
    "immunology",
    "neurology",
    "nephrology",
}

_PHARMACY_NAMES = [
    "Riverbend Pharmacy",
    "Oakline Drug",
    "Cedar Point Rx",
    "Northgate Pharmacy",
    "Harbor Health Drug",
    "Sunfield Pharmacy",
    "Milltown Rx",
]

_STATES = ["OH", "TX", "FL", "NC", "AZ", "MN", "PA", "GA"]
_CITIES = {
    "OH": "Dayton",
    "TX": "Round Rock",
    "FL": "Coral Springs",
    "NC": "Cary",
    "AZ": "Chandler",
    "MN": "Eagan",
    "PA": "Bethlehem",
    "GA": "Marietta",
}

_SPECIALTIES = [
    "internal_medicine",
    "family_medicine",
    "endocrinology",
    "cardiology",
    "rheumatology",
    "psychiatry",
    "oncology",
    "pulmonology",
]


def build_drugs() -> list[Drug]:
    drugs: list[Drug] = []
    for i, (name, generic, cls, specialty, gen_avail, price) in enumerate(_DRUG_SPECS):
        drugs.append(
            Drug(
                drug_id=f"DRG-{i + 1:03d}",
                name=name,
                generic_name=generic,
                therapeutic_class=cls,
                is_specialty=specialty,
                generic_available=gen_avail,
                list_price_30d=price,
            )
        )
    return drugs


def build_plans() -> list[Plan]:
    """Three benefit designs so cost behaviour varies across the population."""
    return [
        Plan(
            plan_id="PLN-VALUE",
            name="CareWeave Value Rx",
            deductible=250.0,
            copay_by_tier={
                DrugTier.GENERIC.value: 8.0,
                DrugTier.PREFERRED_BRAND.value: 40.0,
                DrugTier.NON_PREFERRED_BRAND.value: 90.0,
                DrugTier.SPECIALTY.value: 0.0,
            },
            specialty_coinsurance=0.30,
            mail_order_discount=0.20,
            out_of_network_penalty=0.40,
            formulary_id=FORMULARY_ID,
        ),
        Plan(
            plan_id="PLN-STANDARD",
            name="CareWeave Standard Rx",
            deductible=100.0,
            copay_by_tier={
                DrugTier.GENERIC.value: 5.0,
                DrugTier.PREFERRED_BRAND.value: 30.0,
                DrugTier.NON_PREFERRED_BRAND.value: 70.0,
                DrugTier.SPECIALTY.value: 0.0,
            },
            specialty_coinsurance=0.20,
            mail_order_discount=0.25,
            out_of_network_penalty=0.35,
            formulary_id=FORMULARY_ID,
        ),
        Plan(
            plan_id="PLN-PREMIER",
            name="CareWeave Premier Rx",
            deductible=0.0,
            copay_by_tier={
                DrugTier.GENERIC.value: 3.0,
                DrugTier.PREFERRED_BRAND.value: 20.0,
                DrugTier.NON_PREFERRED_BRAND.value: 50.0,
                DrugTier.SPECIALTY.value: 0.0,
            },
            specialty_coinsurance=0.15,
            mail_order_discount=0.25,
            out_of_network_penalty=0.30,
            formulary_id=FORMULARY_ID,
        ),
    ]


def build_formulary(drugs: list[Drug], rng: random.Random) -> list[FormularyEntry]:
    entries: list[FormularyEntry] = []
    for d in drugs:
        if d.is_specialty:
            tier = DrugTier.SPECIALTY
            pa = True
            step = rng.random() < 0.35
            qty = 30
        elif d.generic_available and d.generic_name:
            tier = DrugTier.GENERIC
            pa = False
            step = False
            qty = 90
        elif d.list_price_30d > 250:
            tier = DrugTier.NON_PREFERRED_BRAND
            pa = rng.random() < 0.6
            step = rng.random() < 0.4
            qty = 30
        else:
            tier = DrugTier.PREFERRED_BRAND
            pa = rng.random() < 0.15
            step = False
            qty = 60

        entries.append(
            FormularyEntry(
                formulary_id=FORMULARY_ID,
                drug_id=d.drug_id,
                tier=tier,
                pa_required=pa,
                step_therapy_required=step,
                quantity_limit_30d=qty,
                # a small number of drugs are genuinely non-covered, which is
                # what makes NOT_ON_FORMULARY rejections possible
                covered=not (tier == DrugTier.NON_PREFERRED_BRAND and rng.random() < 0.12),
            )
        )
    return entries


def build_pharmacies(rng: random.Random) -> list[Pharmacy]:
    pharmacies: list[Pharmacy] = []
    idx = 1
    for state in _STATES:
        for name in _PHARMACY_NAMES[:3]:
            ptype = (
                PharmacyType.RETAIL_OUT_OF_NETWORK
                if rng.random() < 0.15
                else PharmacyType.RETAIL_IN_NETWORK
            )
            pharmacies.append(
                Pharmacy(
                    pharmacy_id=f"PHM-{idx:04d}",
                    name=f"{name} ({_CITIES[state]})",
                    pharmacy_type=ptype,
                    city=_CITIES[state],
                    state=state,
                )
            )
            idx += 1
    pharmacies.append(
        Pharmacy(
            pharmacy_id="PHM-MAIL",
            name="CareWeave Home Delivery",
            pharmacy_type=PharmacyType.MAIL_ORDER,
            city="Columbus",
            state="OH",
        )
    )
    pharmacies.append(
        Pharmacy(
            pharmacy_id="PHM-SPEC",
            name="CareWeave Specialty Pharmacy",
            pharmacy_type=PharmacyType.SPECIALTY,
            city="Columbus",
            state="OH",
        )
    )
    return pharmacies


def build_prescribers(rng: random.Random, n: int = 60) -> list[Prescriber]:
    first = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H."]
    last = [
        "Marsh",
        "Okafor",
        "Ibarra",
        "Lindqvist",
        "Rao",
        "Whitfield",
        "Delacroix",
        "Nakamura",
        "Abbasi",
        "Ferreira",
        "Kowalski",
        "Osei",
    ]
    out: list[Prescriber] = []
    for i in range(n):
        out.append(
            Prescriber(
                prescriber_id=f"PRB-{i + 1:04d}",
                name=f"Dr. {rng.choice(first)} {rng.choice(last)}",
                specialty=rng.choice(_SPECIALTIES),
                # beta-ish shape: most prescribers are responsive, a tail is not
                responsiveness=round(min(1.0, max(0.05, rng.betavariate(5, 2))), 3),
            )
        )
    return out


STATES = _STATES
CITIES = _CITIES
