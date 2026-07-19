"""Antimicrobial drug properties and the deterministic target gate.

The challenge brief is explicit that a prediction must account for the presence of the
drug's molecular target — the system must not report "likely to work" merely because no
resistance marker was found. A drug whose target the organism does not possess is
intrinsically ineffective, and that is a fact about biology, not a thing to learn from
data.

Coverage below is limited to the drug classes commonly labelled in BV-BRC AMR panels.
Anything not listed is reported as out of scope rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass

# Gram stain determines outer-membrane permeability, which is the dominant intrinsic
# resistance mechanism for several drug classes.
GRAM_NEGATIVE = "gram-negative"
GRAM_POSITIVE = "gram-positive"

SPECIES_PROPERTIES: dict[str, dict] = {
    "escherichia coli": {"gram": GRAM_NEGATIVE, "has_mycolic_acid": False},
    "klebsiella pneumoniae": {"gram": GRAM_NEGATIVE, "has_mycolic_acid": False},
    "pseudomonas aeruginosa": {"gram": GRAM_NEGATIVE, "has_mycolic_acid": False},
    "acinetobacter baumannii": {"gram": GRAM_NEGATIVE, "has_mycolic_acid": False},
    "salmonella enterica": {"gram": GRAM_NEGATIVE, "has_mycolic_acid": False},
    "staphylococcus aureus": {"gram": GRAM_POSITIVE, "has_mycolic_acid": False},
    "streptococcus pneumoniae": {"gram": GRAM_POSITIVE, "has_mycolic_acid": False},
    "enterococcus faecium": {"gram": GRAM_POSITIVE, "has_mycolic_acid": False},
    "mycobacterium tuberculosis": {"gram": GRAM_POSITIVE, "has_mycolic_acid": True},
}


@dataclass(frozen=True)
class Drug:
    """One antimicrobial and the conditions under which it can act at all."""

    name: str
    drug_class: str
    target: str
    # AMRFinderPlus "Class" values that count as a known mechanism for this drug. It
    # must match the tool's vocabulary, not our own label: a combination drug like
    # TMP-SMX is reported under SULFONAMIDE and TRIMETHOPRIM separately, and if these
    # do not line up, genuine determinants get downgraded to "statistical association"
    # and the report understates the evidence it actually has.
    amr_classes: frozenset[str] = frozenset()
    # Species whose intrinsic biology makes this drug ineffective regardless of
    # acquired resistance genes.
    intrinsically_resistant: frozenset[str] = frozenset()
    requires_gram: str | None = None

    def is_applicable(self, species: str) -> tuple[bool, str]:
        """Deterministic gate. Returns (applicable, human-readable reason)."""
        species = species.strip().lower()

        if species in self.intrinsically_resistant:
            return False, (
                f"{species.title()} is intrinsically resistant to {self.name} — "
                f"the species lacks a susceptible {self.target} or excludes the drug."
            )

        if self.requires_gram is not None:
            props = SPECIES_PROPERTIES.get(species)
            if props is None:
                return True, f"Species '{species}' not in the property table; gate not applied."
            if props["gram"] != self.requires_gram:
                return False, (
                    f"{self.name} requires a {self.requires_gram} cell envelope; "
                    f"{species.title()} is {props['gram']}."
                )

        return True, f"Target ({self.target}) expected to be present in {species.title()}."


DRUGS: dict[str, Drug] = {
    "ampicillin": Drug(
        name="Ampicillin",
        drug_class="BETA-LACTAM",
        target="penicillin-binding proteins (cell wall synthesis)",
        amr_classes=frozenset({"BETA-LACTAM"}),
        intrinsically_resistant=frozenset({"klebsiella pneumoniae", "pseudomonas aeruginosa"}),
    ),
    "ceftriaxone": Drug(
        name="Ceftriaxone",
        drug_class="BETA-LACTAM",
        target="penicillin-binding proteins (cell wall synthesis)",
        amr_classes=frozenset({"BETA-LACTAM", "CEPHALOSPORIN"}),
        intrinsically_resistant=frozenset({"enterococcus faecium", "pseudomonas aeruginosa"}),
    ),
    "meropenem": Drug(
        name="Meropenem",
        drug_class="BETA-LACTAM",
        target="penicillin-binding proteins (cell wall synthesis)",
        amr_classes=frozenset({"BETA-LACTAM", "CARBAPENEM"}),
    ),
    "cefoxitin": Drug(
        name="Cefoxitin",
        drug_class="BETA-LACTAM",
        target="penicillin-binding proteins (cell wall synthesis)",
        # Cefoxitin is the laboratory surrogate for methicillin resistance in
        # S. aureus: it induces mecA expression more reliably than oxacillin, so a
        # cefoxitin result is how MRSA is called in practice. The determinant is mecA
        # (or mecC), which encodes PBP2a — an alternative penicillin-binding protein
        # with low affinity for essentially every beta-lactam. AMRFinderPlus reports
        # it under BETA-LACTAM with subclass METHICILLIN.
        amr_classes=frozenset({"BETA-LACTAM", "METHICILLIN", "CEPHALOSPORIN"}),
    ),
    "ciprofloxacin": Drug(
        name="Ciprofloxacin",
        drug_class="QUINOLONE",
        # In S. aureus the topoisomerase IV subunit is grlA/grlB (the parC/parE
        # homologue); resistance is usually stepwise point mutations in grlA then gyrA
        # rather than an acquired gene.
        target="DNA gyrase (gyrA/gyrB) and topoisomerase IV (grlA/grlB in S. aureus)",
        amr_classes=frozenset({"QUINOLONE", "FLUOROQUINOLONE"}),
    ),
    "erythromycin": Drug(
        name="Erythromycin",
        drug_class="MACROLIDE",
        target="23S rRNA of the 50S ribosomal subunit",
        # erm genes (ermA/ermB/ermC) methylate the ribosome and confer the MLSb
        # phenotype — macrolide, lincosamide and streptogramin B together. msrA is an
        # efflux pump giving macrolide resistance without lincosamide resistance.
        # AMRFinderPlus spans several class spellings for these, so all are accepted.
        # AMRFinderPlus emits the combined MLSb tag alphabetically, as
        # "LINCOSAMIDE/MACROLIDE/STREPTOGRAMIN". Guessing the component order wrong
        # means the tag never matches and a genuine erm hit is demoted to a bare
        # statistical association. Verified against the real tool, not assumed.
        amr_classes=frozenset({
            "MACROLIDE", "LINCOSAMIDE", "STREPTOGRAMIN", "ERYTHROMYCIN",
            "LINCOSAMIDE/MACROLIDE/STREPTOGRAMIN",
            "MACROLIDE/LINCOSAMIDE/STREPTOGRAMIN",
        }),
    ),
    "levofloxacin": Drug(
        name="Levofloxacin",
        drug_class="QUINOLONE",
        target="DNA gyrase (gyrA/gyrB) and topoisomerase IV (parC/parE)",
        amr_classes=frozenset({"QUINOLONE", "FLUOROQUINOLONE"}),
    ),
    "gentamicin": Drug(
        name="Gentamicin",
        drug_class="AMINOGLYCOSIDE",
        target="30S ribosomal subunit",
        amr_classes=frozenset({"AMINOGLYCOSIDE", "GENTAMICIN"}),
        # Aminoglycoside uptake is oxygen-dependent and poor in enterococci.
        intrinsically_resistant=frozenset({"enterococcus faecium"}),
    ),
    "trimethoprim-sulfamethoxazole": Drug(
        name="Trimethoprim-sulfamethoxazole",
        drug_class="SULFONAMIDE",
        target="dihydrofolate reductase (folA) and dihydropteroate synthase (folP)",
        # A combination drug: AMRFinderPlus reports each component's determinants
        # under its own class.
        amr_classes=frozenset({"SULFONAMIDE", "TRIMETHOPRIM", "FOLATE-PATHWAY-ANTAGONIST"}),
    ),
    "tetracycline": Drug(
        name="Tetracycline",
        drug_class="TETRACYCLINE",
        target="30S ribosomal subunit",
        amr_classes=frozenset({"TETRACYCLINE"}),
    ),
    "vancomycin": Drug(
        name="Vancomycin",
        drug_class="GLYCOPEPTIDE",
        target="D-Ala-D-Ala terminus of peptidoglycan precursors",
        amr_classes=frozenset({"GLYCOPEPTIDE", "VANCOMYCIN"}),
        # Too large to cross the gram-negative outer membrane.
        requires_gram=GRAM_POSITIVE,
    ),
    "colistin": Drug(
        name="Colistin",
        drug_class="POLYMYXIN",
        target="lipid A of the outer membrane",
        amr_classes=frozenset({"POLYMYXIN", "COLISTIN"}),
        # No outer membrane in gram-positives, so nothing to bind.
        requires_gram=GRAM_NEGATIVE,
    ),
}


def get_drug(name: str) -> Drug | None:
    """Look up a drug by name, case- and punctuation-tolerant."""
    key = name.strip().lower().replace("/", "-").replace(" ", "-")
    if key in DRUGS:
        return DRUGS[key]
    # BV-BRC sometimes writes the combination drug with a slash or as two words.
    aliases = {
        "trimethoprim-sulphamethoxazole": "trimethoprim-sulfamethoxazole",
        "cotrimoxazole": "trimethoprim-sulfamethoxazole",
        "tmp-smx": "trimethoprim-sulfamethoxazole",
    }
    return DRUGS.get(aliases.get(key, ""), None)


def class_features_for_drug(drug: Drug) -> list[str]:
    """Genome Reader feature names that count as a known mechanism for this drug.

    Used to tell "a known determinant for this drug class was found" apart from "the
    model picked up a statistical association elsewhere in the genome".
    """
    tags = drug.amr_classes or {drug.drug_class}
    return [f"class:{tag}" for tag in sorted(tags)]
