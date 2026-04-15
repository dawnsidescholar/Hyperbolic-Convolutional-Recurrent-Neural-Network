"""
Generalizable Phylogeny Tree Diagram Generator

Generates genotype and phenotype pedigree diagrams up to F2 for
crosses between two lines (A and B).  Supports:

  * Arbitrary ploidy  (haploid, diploid, tetraploid, …)
  * Autosomal and sex-linked (X-linked / Z-linked) inheritance
  * Cytoplasmic / maternal-effect factors
  * Sex-limited expression and sex-specific lethality
  * Fully custom offspring-generation functions

Usage:
    Modify the CONFIGURATION section at the bottom with your genotypes
    and phenotype rules, then run:

        python phylogeny_tree.py

Convention:
    X = dominant allele controlling the phenotype
    x = recessive allele
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Core genetics helpers
# ---------------------------------------------------------------------------

Genotype = tuple[str, ...]
Ratios   = dict[Genotype, float]

# A function that, given (sire_genotype, dam_genotype), returns a list of
# offspring genotypes (one entry per equally-likely outcome).
OffspringFn = Callable[[Genotype, Genotype], list[Genotype]]

# Phenotype function: (genotype, sex) → phenotype name
PhenotypeFn = Callable[[Genotype, str], str]


def gametes(genotype: Genotype) -> list[Genotype]:
    """Return all possible gametes from a genotype.

    For an organism with ploidy *n* the genotype has *n* alleles and each
    gamete carries *n // 2* alleles (like meiosis halving the chromosome
    number).  All unique combinations of choosing n//2 alleles from the
    genotype are returned.

    For haploid (single-allele) genotypes the gamete is the genotype itself.
    """
    n = len(genotype)
    if n <= 1:
        return [genotype]
    half = n // 2
    seen: set[Genotype] = set()
    result: list[Genotype] = []
    for combo in itertools.combinations(genotype, half):
        canon = tuple(sorted(combo))
        if canon not in seen:
            seen.add(canon)
            result.append(canon)
    return result


def cross(parent_a: Genotype, parent_b: Genotype) -> list[Genotype]:
    """Return every possible offspring genotype from two parents.

    Each gamete from *parent_a* is combined with each gamete from *parent_b*.
    Resulting genotypes are sorted tuples so that e.g. ('X', 'x') and
    ('x', 'X') are treated as the same genotype.
    """
    gam_a = gametes(parent_a)
    gam_b = gametes(parent_b)
    offspring: list[Genotype] = []
    for ga in gam_a:
        for gb in gam_b:
            child = tuple(sorted(ga + gb))
            offspring.append(child)
    return offspring


def offspring_ratios(offspring: list[Genotype]) -> Ratios:
    """Compute genotype → fractional ratio from a list of offspring."""
    counts = Counter(offspring)
    total = len(offspring)
    return {gt: count / total for gt, count in counts.items()}


def format_genotype(gt: Genotype) -> str:
    """Human-readable genotype string, e.g. ('X', 'x') → 'Xx'."""
    return "".join(gt)


# ---------------------------------------------------------------------------
# Built-in offspring functions
# ---------------------------------------------------------------------------

def autosomal_offspring(sire: Genotype, dam: Genotype) -> list[Genotype]:
    """Standard Mendelian cross — same genotypes for both sexes."""
    return cross(sire, dam)


def x_linked_male_offspring(sire: Genotype, dam: Genotype) -> list[Genotype]:
    """X-linked (XX♀ / XY♂): sons get their X only from their mother.

    The sire's X-linked genotype is NOT passed to sons (they get Y instead).
    Each son is hemizygous — carrying one maternal gamete.
    """
    return gametes(dam)


def x_linked_female_offspring(sire: Genotype, dam: Genotype) -> list[Genotype]:
    """X-linked (XX♀ / XY♂): daughters get X from both parents."""
    return cross(sire, dam)


# ---------------------------------------------------------------------------
# Sex-aware crossing engine
# ---------------------------------------------------------------------------

class Line:
    """A breeding line with separate male and female genotypes."""

    def __init__(self, name: str, male_gt: Genotype, female_gt: Genotype):
        self.name = name
        self.male_gt = male_gt
        self.female_gt = female_gt

    def __repr__(self) -> str:
        return (f"Line({self.name}, male={format_genotype(self.male_gt)}, "
                f"female={format_genotype(self.female_gt)})")


class CrossResult:
    """Stores the result of a single cross."""

    def __init__(
        self,
        label: str,
        sire_gt: Genotype,
        dam_gt: Genotype,
        male_ratios: Ratios,
        female_ratios: Ratios,
        phenotype_fn: PhenotypeFn,
    ):
        self.label = label
        self.sire_gt = sire_gt
        self.dam_gt = dam_gt
        self.male_ratios = male_ratios
        self.female_ratios = female_ratios
        self.phenotype_fn = phenotype_fn

    # Convenience -----------------------------------------------------------

    def _pheno_ratios(self, ratios: Ratios, sex: str) -> dict[str, float]:
        pheno: dict[str, float] = {}
        for gt, frac in ratios.items():
            p = self.phenotype_fn(gt, sex)
            pheno[p] = pheno.get(p, 0.0) + frac
        return pheno

    @property
    def male_pheno(self) -> dict[str, float]:
        return self._pheno_ratios(self.male_ratios, "male")

    @property
    def female_pheno(self) -> dict[str, float]:
        return self._pheno_ratios(self.female_ratios, "female")

    def summary(self) -> str:
        parts = [self.label]
        parts.append(f"  Sire: {format_genotype(self.sire_gt)}  ×  "
                     f"Dam: {format_genotype(self.dam_gt)}")
        parts.append("  Males:   " + self._ratio_str(self.male_ratios,
                                                      self.male_pheno, "male"))
        parts.append("  Females: " + self._ratio_str(self.female_ratios,
                                                      self.female_pheno, "female"))
        return "\n".join(parts)

    @staticmethod
    def _ratio_str(gt_ratios: Ratios,
                   ph_ratios: dict[str, float], sex: str) -> str:
        gt_parts = [f"{format_genotype(gt)} ({v:.2%})"
                    for gt, v in sorted(gt_ratios.items())]
        ph_parts = [f"{p} ({v:.2%})"
                    for p, v in sorted(ph_ratios.items())]
        return ("Geno: " + ", ".join(gt_parts) +
                "  |  Pheno: " + ", ".join(ph_parts))


def perform_cross(
    label: str,
    sire_gt: Genotype,
    dam_gt: Genotype,
    male_offspring_fn: OffspringFn,
    female_offspring_fn: OffspringFn,
    phenotype_fn: PhenotypeFn,
) -> CrossResult:
    """Cross *sire_gt* × *dam_gt* and compute male / female offspring ratios.

    *male_offspring_fn(sire, dam)* returns the list of equally-likely male
    offspring genotypes.  *female_offspring_fn(sire, dam)* does the same for
    female offspring.  This design lets you model autosomal, sex-linked, or
    any custom inheritance pattern.
    """
    male_raw  = male_offspring_fn(sire_gt, dam_gt)
    female_raw = female_offspring_fn(sire_gt, dam_gt)
    return CrossResult(
        label, sire_gt, dam_gt,
        offspring_ratios(male_raw),
        offspring_ratios(female_raw),
        phenotype_fn,
    )


# ---------------------------------------------------------------------------
# Full pedigree builder (P → F1 → F2)
# ---------------------------------------------------------------------------

def build_pedigree(
    line_a: Line,
    line_b: Line,
    male_offspring_fn: OffspringFn,
    female_offspring_fn: OffspringFn,
    phenotype_fn: PhenotypeFn,
) -> dict[str, CrossResult]:
    """Build the complete pedigree from two parental lines up to F2.

    Returns a dict keyed by cross label.
    """
    results: dict[str, CrossResult] = {}

    # --- F1 ----------------------------------------------------------------
    f1_I = perform_cross(
        "F1 Cross I (A♂ × B♀)", line_a.male_gt, line_b.female_gt,
        male_offspring_fn, female_offspring_fn, phenotype_fn,
    )
    f1_II = perform_cross(
        "F1 Cross II (B♂ × A♀)", line_b.male_gt, line_a.female_gt,
        male_offspring_fn, female_offspring_fn, phenotype_fn,
    )
    results["F1_I"] = f1_I
    results["F1_II"] = f1_II

    # Representative F1 genotypes (most common) for building F2
    def _pick_representative(ratios: Ratios) -> Genotype:
        return max(ratios, key=lambda k: ratios[k])

    f1_I_male = _pick_representative(f1_I.male_ratios)
    f1_I_female = _pick_representative(f1_I.female_ratios)
    f1_II_male = _pick_representative(f1_II.male_ratios)
    f1_II_female = _pick_representative(f1_II.female_ratios)

    # --- F2 ----------------------------------------------------------------
    f2_labels = [
        ("F2 I♂×I♀",   f1_I_male,  f1_I_female),
        ("F2 I♂×II♀",  f1_I_male,  f1_II_female),
        ("F2 II♂×I♀",  f1_II_male, f1_I_female),
        ("F2 II♂×II♀", f1_II_male, f1_II_female),
    ]
    for lbl, sire, dam in f2_labels:
        key = (lbl.replace(" ", "_").replace("♂", "m")
                   .replace("♀", "f").replace("×", "x"))
        results[key] = perform_cross(
            lbl, sire, dam,
            male_offspring_fn, female_offspring_fn, phenotype_fn,
        )

    return results


# ---------------------------------------------------------------------------
# Matplotlib tree diagram
# ---------------------------------------------------------------------------

_PHENOTYPE_COLORS: dict[str, str] = {
    "white": "#f0f0f0",
    "green": "#6abf69",
    "red": "#e06060",
    "blue": "#6090e0",
    "yellow": "#e0d060",
    "black": "#404040",
}


def _color_for(phenotype: str) -> str:
    return _PHENOTYPE_COLORS.get(phenotype.lower().strip(), "#cccccc")


def _box(ax: plt.Axes, x: float, y: float, text: str,
         color: str = "#ffffff", w: float = 2.4, h: float = 0.9,
         fontsize: int = 8, textcolor: str = "black"):
    """Draw a rounded box with centred text."""
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.1", facecolor=color,
        edgecolor="#444444", linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight="bold",
            linespacing=1.35, clip_on=False)


def _arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float,
           color: str = "#666666"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                        connectionstyle="arc3,rad=0"),
    )


def _ratio_line(ratios: dict[tuple[str, ...], float],
                phenotype_fn: Callable[[tuple[str, ...], str], str],
                sex: str) -> str:
    """One-line summary: 'Xx → white (50%)  xx → green (25%)' etc."""
    icon = "♂" if sex == "male" else "♀"
    parts: list[str] = []
    for gt, frac in sorted(ratios.items(), key=lambda kv: -kv[1]):
        p = phenotype_fn(gt, sex)
        pct = frac * 100
        pct_s = f"{int(pct)}%" if pct == int(pct) else f"{pct:.1f}%"
        parts.append(f"{format_genotype(gt)}→{p} ({pct_s})")
    return f"{icon} " + "  ".join(parts)


def draw_pedigree(
    line_a: Line,
    line_b: Line,
    pedigree: dict[str, CrossResult],
    title: str = "Phylogeny Tree Diagram",
    save_path: str = "phylogeny_tree.png",
):
    """Draw the full P → F1 → F2 pedigree tree and save to *save_path*."""

    # --- Canvas setup (wide, non-equal-aspect for readability) -------------
    fig, ax = plt.subplots(figsize=(26, 18))
    ax.set_xlim(-13, 13)
    ax.set_ylim(-2, 18)
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")

    # ---- Y levels ---------------------------------------------------------
    Y_TITLE = 17.0
    Y_P     = 15.5          # parental row
    Y_CROSS = 13.5          # cross-label row
    Y_F1    = 11.5          # F1 offspring row
    Y_F2LBL = 9.0           # F2 cross-label row
    Y_F2    = 5.5           # F2 offspring detail row

    BW = 2.8   # standard box width
    BH = 1.0   # standard box height
    FS = 8     # font size

    # ---- Title ------------------------------------------------------------
    ax.text(0, Y_TITLE, title, ha="center", va="center", fontsize=16,
            fontweight="bold", color="#222")

    # ---- Generation labels (left margin) ----------------------------------
    ax.text(-12.2, Y_P, "P", fontsize=14, fontweight="bold", color="#555",
            va="center")
    ax.text(-12.2, Y_F1, "F1", fontsize=14, fontweight="bold", color="#555",
            va="center")
    ax.text(-12.2, Y_F2LBL, "F2", fontsize=14, fontweight="bold", color="#555",
            va="center")

    # ---- P generation (4 boxes) -------------------------------------------
    p_positions = {
        "A_male":   -7,
        "A_female": -2.5,
        "B_male":    2.5,
        "B_female":  7,
    }
    _box(ax, p_positions["A_male"], Y_P,
         f"Line A  ♂\n{format_genotype(line_a.male_gt)}", "#cde4ff",
         w=BW, h=BH, fontsize=FS)
    _box(ax, p_positions["A_female"], Y_P,
         f"Line A  ♀\n{format_genotype(line_a.female_gt)}", "#ffd6ec",
         w=BW, h=BH, fontsize=FS)
    _box(ax, p_positions["B_male"], Y_P,
         f"Line B  ♂\n{format_genotype(line_b.male_gt)}", "#cde4ff",
         w=BW, h=BH, fontsize=FS)
    _box(ax, p_positions["B_female"], Y_P,
         f"Line B  ♀\n{format_genotype(line_b.female_gt)}", "#ffd6ec",
         w=BW, h=BH, fontsize=FS)

    # ---- F1 cross labels --------------------------------------------------
    cross_I_x  = -5.0
    cross_II_x =  5.0

    _box(ax, cross_I_x, Y_CROSS, "Cross I\nA♂ × B♀", "#fff8cc",
         w=BW, h=BH, fontsize=FS)
    _box(ax, cross_II_x, Y_CROSS, "Cross II\nB♂ × A♀", "#fff8cc",
         w=BW, h=BH, fontsize=FS)

    # Arrows: P → Cross labels
    # Cross I: A♂ and B♀
    _arrow(ax, p_positions["A_male"], Y_P - BH / 2,
           cross_I_x - 0.3, Y_CROSS + BH / 2)
    _arrow(ax, p_positions["B_female"], Y_P - BH / 2,
           cross_I_x + 0.3, Y_CROSS + BH / 2)
    # Cross II: B♂ and A♀
    _arrow(ax, p_positions["B_male"], Y_P - BH / 2,
           cross_II_x + 0.3, Y_CROSS + BH / 2)
    _arrow(ax, p_positions["A_female"], Y_P - BH / 2,
           cross_II_x - 0.3, Y_CROSS + BH / 2)

    # ---- F1 offspring boxes -----------------------------------------------
    f1_I  = pedigree["F1_I"]
    f1_II = pedigree["F1_II"]

    def _f1_text(cr: CrossResult, tag: str) -> str:
        m = _ratio_line(cr.male_ratios, cr.phenotype_fn, "male")
        f = _ratio_line(cr.female_ratios, cr.phenotype_fn, "female")
        return f"{tag} offspring\n{m}\n{f}"

    f1_bw = 4.5
    f1_bh = 1.2
    _box(ax, cross_I_x, Y_F1, _f1_text(f1_I, "F1-I"), "#dff0df",
         w=f1_bw, h=f1_bh, fontsize=7)
    _box(ax, cross_II_x, Y_F1, _f1_text(f1_II, "F1-II"), "#dff0df",
         w=f1_bw, h=f1_bh, fontsize=7)

    _arrow(ax, cross_I_x, Y_CROSS - BH / 2, cross_I_x, Y_F1 + f1_bh / 2)
    _arrow(ax, cross_II_x, Y_CROSS - BH / 2, cross_II_x, Y_F1 + f1_bh / 2)

    # ---- F2 cross labels --------------------------------------------------
    f2_keys = [
        ("F2_ImxIf",   "I♂ × I♀"),
        ("F2_ImxIIf",  "I♂ × II♀"),
        ("F2_IImxIf",  "II♂ × I♀"),
        ("F2_IImxIIf", "II♂ × II♀"),
    ]
    f2_xs = [-9.5, -3.2, 3.2, 9.5]

    for (key, lbl), cx in zip(f2_keys, f2_xs):
        cr = pedigree[key]
        _box(ax, cx, Y_F2LBL,
             f"{lbl}\n{format_genotype(cr.sire_gt)} × {format_genotype(cr.dam_gt)}",
             "#fff8cc", w=3.2, h=BH, fontsize=7)

    # Arrows from F1 → F2 cross labels
    # I♂×I♀  (both parents from F1-I)
    _arrow(ax, cross_I_x,  Y_F1 - f1_bh / 2, f2_xs[0], Y_F2LBL + BH / 2)
    _arrow(ax, cross_I_x,  Y_F1 - f1_bh / 2, f2_xs[1], Y_F2LBL + BH / 2)
    # I♂×II♀ also needs arrow from F1-II for the dam
    _arrow(ax, cross_II_x, Y_F1 - f1_bh / 2, f2_xs[1], Y_F2LBL + BH / 2)
    # II♂×I♀ arrows from F1-II (sire) and F1-I (dam)
    _arrow(ax, cross_II_x, Y_F1 - f1_bh / 2, f2_xs[2], Y_F2LBL + BH / 2)
    _arrow(ax, cross_I_x,  Y_F1 - f1_bh / 2, f2_xs[2], Y_F2LBL + BH / 2)
    # II♂×II♀ (both parents from F1-II)
    _arrow(ax, cross_II_x, Y_F1 - f1_bh / 2, f2_xs[3], Y_F2LBL + BH / 2)

    # ---- F2 offspring detail boxes ----------------------------------------
    for (key, lbl), cx in zip(f2_keys, f2_xs):
        cr = pedigree[key]

        lines = [f"F2 : {lbl}", ""]
        lines.append("── Males ──")
        for gt, frac in sorted(cr.male_ratios.items(), key=lambda kv: -kv[1]):
            p = cr.phenotype_fn(gt, "male")
            pct = frac * 100
            pct_s = f"{int(pct)}%" if pct == int(pct) else f"{pct:.1f}%"
            lines.append(f"{format_genotype(gt)} → {p}  ({pct_s})")
        lines.append("")
        lines.append("── Females ──")
        for gt, frac in sorted(cr.female_ratios.items(), key=lambda kv: -kv[1]):
            p = cr.phenotype_fn(gt, "female")
            pct = frac * 100
            pct_s = f"{int(pct)}%" if pct == int(pct) else f"{pct:.1f}%"
            lines.append(f"{format_genotype(gt)} → {p}  ({pct_s})")

        text = "\n".join(lines)
        detail_h = max(2.5, 0.28 * len(lines))
        detail_y = Y_F2 + detail_h / 2 - 1.0
        _box(ax, cx, detail_y, text, "#eef0ff",
             w=3.8, h=detail_h, fontsize=6.5)
        _arrow(ax, cx, Y_F2LBL - BH / 2, cx, detail_y + detail_h / 2)

    # ---- Phenotype colour legend ------------------------------------------
    used_phenotypes: set[str] = set()
    for cr in pedigree.values():
        for gt in cr.male_ratios:
            used_phenotypes.add(cr.phenotype_fn(gt, "male"))
        for gt in cr.female_ratios:
            used_phenotypes.add(cr.phenotype_fn(gt, "female"))

    legend_patches = [
        mpatches.Patch(facecolor=_color_for(p), edgecolor="#333", label=p)
        for p in sorted(used_phenotypes)
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
                  title="Phenotypes", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagram saved to {save_path}")


# ---------------------------------------------------------------------------
# Text-based summary printer
# ---------------------------------------------------------------------------

def print_summary(pedigree: dict[str, CrossResult]):
    """Print a text summary of all crosses."""
    order = ["F1_I", "F1_II", "F2_ImxIf", "F2_ImxIIf", "F2_IImxIf", "F2_IImxIIf"]
    print("=" * 70)
    print("PEDIGREE SUMMARY")
    print("=" * 70)
    for key in order:
        if key in pedigree:
            print(pedigree[key].summary())
            print("-" * 70)


# ===========================================================================
#  CONFIGURATION — edit this section for your specific experiment
# ===========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # 1.  Define parental genotypes.
    #     Each genotype is a TUPLE of allele strings.  The length of the
    #     tuple equals the ploidy for that locus.
    #
    #     Examples
    #       Diploid:   ("X", "X")  or  ("X", "x")
    #       Triploid:  ("X", "X", "x")
    #       Tetraploid:("X", "X", "x", "x")
    #       Haploid:   ("X",)
    # -----------------------------------------------------------------------

    A_MALE_GENOTYPE   = ("X", "X")       # Line A males
    A_FEMALE_GENOTYPE = ("X", "X")       # Line A females
    B_MALE_GENOTYPE   = ("x", "x")       # Line B males
    B_FEMALE_GENOTYPE = ("x", "x")       # Line B females

    # -----------------------------------------------------------------------
    # 2.  Offspring generation functions.
    #     These define HOW male and female offspring genotypes are produced
    #     from (sire_genotype, dam_genotype).
    #
    #     Built-in options:
    #       autosomal_offspring         — standard Mendelian (same for ♂/♀)
    #       x_linked_male_offspring     — sons get X from mother only
    #       x_linked_female_offspring   — daughters get X from both parents
    #
    #     You can also write a fully custom function:
    #       def my_offspring(sire, dam) -> list[tuple[str, ...]]:
    #           ...
    # -----------------------------------------------------------------------

    male_offspring_fn   = autosomal_offspring    # same as female for autosomal
    female_offspring_fn = autosomal_offspring

    # -----------------------------------------------------------------------
    # 3.  Phenotype function.
    #     Given a genotype tuple and sex string ("male" / "female"),
    #     return a phenotype name (str).
    #     Below: simple complete dominance — any 'X' allele → "white",
    #     all 'x' → "green".
    # -----------------------------------------------------------------------

    def phenotype_fn(genotype: tuple[str, ...], sex: str) -> str:
        """Return phenotype based on genotype and sex.

        Adjust this function for your specific inheritance model.
        """
        if "X" in genotype:
            return "white"
        else:
            return "green"

    # -----------------------------------------------------------------------
    # 4.  Build and display the pedigree
    # -----------------------------------------------------------------------

    line_a = Line("A", A_MALE_GENOTYPE, A_FEMALE_GENOTYPE)
    line_b = Line("B", B_MALE_GENOTYPE, B_FEMALE_GENOTYPE)

    pedigree = build_pedigree(
        line_a, line_b,
        male_offspring_fn=male_offspring_fn,
        female_offspring_fn=female_offspring_fn,
        phenotype_fn=phenotype_fn,
    )

    print_summary(pedigree)

    draw_pedigree(
        line_a, line_b, pedigree,
        title="Phylogeny Tree: Line A × Line B  (up to F2)",
        save_path="phylogeny_tree.png",
    )
