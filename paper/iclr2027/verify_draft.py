"""Read-only scientific/build checks; run after generating assets and both PDFs."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E2B = ROOT / "results/target_conditioned/e2b_fixed_cost_shortlist/domainnet_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalized_rendering(text: str) -> str:
    # PDF line wrapping introduces spaces, ligatures, and discretionary hyphens.
    return re.sub(r"[\s\-\u00ad]", "", unicodedata.normalize("NFKC", text))


def text_from_pdf(path: Path, first_page: bool = False) -> str:
    command = ["pdftotext", "-layout"]
    if first_page:
        command += ["-f", "1", "-l", "1"]
    command += [str(path), "-"]
    return subprocess.check_output(command, text=True)


def main() -> None:
    manifest = json.loads((HERE / "generated/asset_manifest.json").read_text())
    for relative, expected in manifest["input_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    print("PASS: asset input hashes")

    # Independently recover primary means from the twelve task-level audit tables.
    summaries = {r["method"]: r for r in read_csv(E2B / "summary/method_summary.csv")
                 if int(r["shortlist_size"]) == 227}
    values: dict[str, dict[str, list[dict[str, str]]]] = {}
    for encoder in ("resnet50", "dinov2_b14"):
        for row in read_csv(E2B / f"test_audit/{encoder}/shortlist_audit.csv"):
            if int(row["shortlist_size"]) == 227:
                values.setdefault(row["method"], {}).setdefault(row["target_domain"], []).append(row)
        selections = {(r["target_domain"], r["method"]): r["selected_combination"]
                      for r in read_csv(E2B / f"validation/{encoder}/selections.csv")
                      if int(r["shortlist_size"]) == 227 and r["method"] in ("target_a", "second_moment_mmd")}
        assert len(selections) == 12
        for domain, method in selections:
            if method == "target_a":
                assert selections[domain, method] == selections[domain, "second_moment_mmd"]
    for method, domains in values.items():
        assert len(domains) == 6
        for raw, aggregate in (("true_top_q_recall", "mean_true_top_q_recall"),
                               ("selected_normalized_regret", "mean_selected_normalized_regret")):
            mean = np.mean([np.mean([float(r[raw]) for r in task]) for task in domains.values()])
            assert np.isclose(mean, float(summaries[method][aggregate]), rtol=1e-12, atol=1e-15)
    print("PASS: eight primary method means and twelve A-opt/MMD validation choices")

    eye = np.eye(3)
    a = eye[[0, 0, 0, 0, 1]]
    b1, b2 = eye[[0, 0, 0, 0, 2]], eye[[0, 2, 2, 2, 2]]
    ranks = []
    for b in (b1, b2):
        assert np.allclose(np.linalg.svd(b, compute_uv=False), [2, 1, 0])
        s = np.linalg.svd(np.vstack([a, b]), compute_uv=False)
        p = s / s.sum()
        ranks.append(float(np.exp(-np.sum(p * np.log(p)))))
    assert np.allclose(ranks, [2.626012, 2.849536], atol=1e-6, rtol=0)
    # Numerical sanity checks supplement, and do not replace, the written proofs.
    rng = np.random.default_rng(17)
    for _ in range(50):
        x, h = rng.normal(size=(9, 4)), rng.normal(size=(7, 4))
        g, c = x.T @ x, h.T @ h / len(h)
        ev, u = np.linalg.eigh(g)
        sketch = (u[:, -2:] * ev[-2:]) @ u[:, -2:].T
        delta = float(ev[-3])
        f = lambda z: float(np.trace(c @ np.linalg.inv(np.eye(4) + z)))
        assert f(sketch + delta * np.eye(4)) <= f(g) + 1e-12 <= f(sketch) + 2e-12
        aa, bb, cc = rng.uniform(.1, 3), rng.uniform(.1, 3), rng.uniform(0, 1)
        v = np.array([cc, np.sqrt(1 - cc * cc)])
        block = np.diag([aa * aa, 0]) + bb * bb * np.outer(v, v)
        disc = np.sqrt((aa * aa - bb * bb)**2 + 4 * aa * aa * bb * bb * cc * cc)
        assert np.allclose(np.linalg.eigvalsh(block), [(aa*aa+bb*bb-disc)/2, (aa*aa+bb*bb+disc)/2])
    print("PASS: constructive counterexample, local eigenvalues, PSD interval sanity checks")

    keys = set(re.findall(r"@\w+\{([^,]+),", (HERE / "references.bib").read_text()))
    body_citations = []
    labels = []
    for language in ("en", "zh"):
        paths = [HERE / f"{part}_{language}.tex" for part in ("body", "appendix", "statements")]
        source = "\n".join(p.read_text(encoding="utf-8") for p in paths)
        citations = {key for group in re.findall(r"\\cite[pt]\{([^}]+)\}", source)
                     for key in group.split(",")}
        assert citations <= keys
        body_citations.append(citations)
        source_labels = re.findall(r"\\label\{([^}]+)\}", source)
        assert len(source_labels) == len(set(source_labels))
        labels.append(set(source_labels))
        assert set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", source)) <= set(source_labels)
    assert body_citations[0] == body_citations[1] == keys
    assert labels[0] == labels[1]
    print("PASS: bilingual label/citation parity, no missing or unused bibliography entries")

    md = (HERE / "abstract.md").read_text(encoding="utf-8")
    for lang, stem, opening, heading in (
        ("en", "main", "Can unlabeled", r"^\s*1\s+I\s*NTRODUCTION"),
        ("zh", "main_zh", "无标签", r"^\s*1\s+引言"),
    ):
        abstract = " ".join((HERE / f"abstract_{lang}.txt").read_text(encoding="utf-8").split())
        assert abstract in md
        tex = (HERE / f"generated/abstract_{lang}.tex").read_text(encoding="utf-8").strip()
        assert tex.replace(r"\%", "%") == abstract
        log = (HERE / f"{stem}.log").read_text(errors="replace")
        for problem in ("Overfull", "Missing character", "undefined", "There were undefined"):
            assert problem not in log, (stem, problem)
        first = text_from_pdf(HERE / f"{stem}.pdf", first_page=True)
        clean = "\n".join(line[6:] if re.match(r"^\d{3}(?:\s|$)", line) else line
                          for line in first.splitlines())
        end = re.search(heading, clean, re.M)
        assert end is not None
        actual = clean[clean.index(opening):end.start()]
        assert normalized_rendering(actual) == normalized_rendering(abstract), (lang, actual)
        full_text = text_from_pdf(HERE / f"{stem}.pdf")
        assert "\ufffd" not in full_text
        if lang == "zh":
            assert sum("\u4e00" <= c <= "\u9fff" for c in full_text) > 7000
        aux = (HERE / f"{stem}.aux").read_text(encoding="utf-8")
        end_page = int(re.search(r"\\newlabel\{main-text-end\}\{\{[^}]*\}\{(\d+)\}", aux).group(1))
        main_pages = [int(v) for v in re.findall(r"\\newlabel\{(?:fig:[^}]+|tab:[^}]+)\}\{\{[^}]*\}\{(\d+)\}", aux)]
        assert max(main_pages + [end_page]) <= 9
        print(f"PASS: {stem} abstract matches text/Markdown/PDF; main text ends on page {end_page}")
    print("All automated draft checks passed. Visual review and human scientific sign-off remain separate.")


if __name__ == "__main__":
    main()
