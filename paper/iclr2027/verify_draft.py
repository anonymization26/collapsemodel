"""Read-only scientific/build checks; run after generating assets and both PDFs."""
from __future__ import annotations

import csv
from collections import defaultdict
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
STYLE = HERE / "iclr2027-style"
STYLE_SHA256 = {
    "iclr2027_conference.sty": "797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6",
    "iclr2027_conference.bst": "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5",
    "fancyhdr.sty": "b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea",
    "natbib.sty": "88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d",
}


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


def verify_method_label(text: str, source: str) -> None:
    text = " ".join(text.split())
    assert not re.search(r"(?<!Target )\bA-opt\b", text), (source, "use Target A-opt consistently")
    assert "Target Target A-opt" not in text, (source, "duplicated method prefix")


def verify_method_naming() -> None:
    sources = [HERE / f"{part}_{lang}.{suffix}"
               for lang in ("en", "zh")
               for part, suffix in (("body", "tex"), ("appendix", "tex"), ("abstract", "txt"))]
    sources += [HERE / "abstract.md", HERE / "submission_metadata.md", HERE / "build_assets.py"]
    sources += list((HERE / "generated").glob("*.tex"))
    for path in sources:
        verify_method_label(path.read_text(encoding="utf-8"), path.name)
    for name in ("decision_budget", "quality_cost", "shortlist_recall"):
        text = text_from_pdf(HERE / "figures" / f"{name}.pdf")
        assert "Target A-opt" in " ".join(text.split()), name
        verify_method_label(text, name)
    print("PASS: Target A-opt naming in both manuscripts, abstracts, tables, and figure legends")


def verify_format(stem: str, log: str, full_text: str) -> None:
    # Severe box stretching needs review even when the PDF compiles successfully.
    assert not re.search(r"Underfull \\[hv]box \(badness 10000\)", log), (
        stem, "severe spacing warning; inspect pagination and line breaks"
    )
    recorded_inputs = {
        (HERE / line.removeprefix("INPUT ")).resolve()
        for line in (HERE / f"{stem}.fls").read_text().splitlines()
        if line.startswith("INPUT ")
    }
    for name in STYLE_SHA256:
        if name.endswith(".sty"):
            loaded = {path for path in recorded_inputs if path.name == name}
            # TeX also records existence probes of the identical legacy copies.
            assert (STYLE / name).resolve() in loaded, (stem, name, loaded)
            assert all(hashlib.sha256(path.read_bytes()).hexdigest() == STYLE_SHA256[name]
                       for path in loaded), (stem, name, "non-template dependency")
            assert f"(./iclr2027-style/{name}" in log, (stem, name, "not loaded")
    assert "The style file: iclr2027-style/iclr2027_conference.bst" in (HERE / f"{stem}.blg").read_text()
    for field, expected in {
        "FONT": (10, 11),
        "TEXT": (5.5 * 72.27, 9 * 72.27),
        "PAGE": (8.5 * 72.27, 11 * 72.27),
        "PAR": (0, 6),
    }.items():
        match = re.search(rf"^ICLR-FORMAT-{field}: (.+)$", log, re.M)
        assert match is not None, (stem, field, "missing layout diagnostics")
        actual = [float(value.removesuffix("pt")) for value in match[1].split("; ")]
        assert np.allclose(actual, expected, rtol=0, atol=1e-3), (stem, field, actual)
    info = subprocess.check_output(["pdfinfo", str(HERE / f"{stem}.pdf")], text=True)
    assert re.search(r"Page size:\s+612 x 792 pts", info), (stem, "not US Letter")
    page_count = int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1])
    pages = full_text.split("\f")
    assert not pages.pop().strip(), (stem, "missing final PDF page delimiter")
    assert len(pages) == page_count
    for number, page in enumerate(pages, 1):
        assert "Under review as a conference paper at ICLR 2027" in page, (stem, number, "header")
        assert page.rstrip().splitlines()[-1].strip() == str(number), (stem, number, "footer")
    assert "Anonymous authors" in pages[0] and "Paper under double-blind review" in pages[0]
    print(f"PASS: {stem} official dependencies, 10/11pt text, layout, headers and page numbers")


def verify_construction_results() -> None:
    revision = ROOT / "results/target_conditioned/revision_20260922"
    root = revision / "e2b_decision_cost_v1"
    summary = json.loads((revision / "challenge_summary/summary.json").read_text())
    assert summary["independent_dataset"] is False
    curves, conditional = defaultdict(list), defaultdict(list)
    for encoder in ("resnet50", "dinov2_b14"):
        original = json.loads((root / encoder / "original/screen/candidate_membership.json").read_text())
        hashed = json.loads((root / encoder / "hash_partition/screen/candidate_membership.json").read_text())
        for domain in {n.split("__")[0] for n in original}:
            left = [s for n, ids in original.items() if n.startswith(domain + "__") for s in ids]
            right = [s for n, ids in hashed.items() if n.startswith(domain + "__") for s in ids]
            assert len(set(left)) == len(set(right)) == 1536
            assert set(left) == set(right)
            assert all(len(ids) == 512 for n, ids in hashed.items() if n.startswith(domain + "__"))
        for construction in ("original", "hash_partition"):
            path = root / encoder / construction / "test-audit"
            audit = json.loads((path / "audit.json").read_text())["rows"]
            full = {(r["target_domain"], r["readout"], r["metric"]): r["test_loss"]
                    for r in audit if r["group"] == "exhaustive"}
            for r in audit:
                key = construction, r["readout"], r["metric"], r["group"].split(":")[0], r["shortlist_size"]
                curves[key].append(r)
                reference = full[r["target_domain"], r["readout"], r["metric"]]
                if r["group"] == "exhaustive":
                    filename = f"{r['target_domain']}__{r['readout']}_losses.json"
                    validation = json.loads((path.parent / "validate" / filename).read_text())
                    test = json.loads((path / filename).read_text())
                    choice = min(validation, key=lambda c: (validation[c][r["metric"]], c))
                    assert r["combination"] == choice and reference == test[choice][r["metric"]]
            for r in json.loads((path / "within_composition.json").read_text())["rows"]:
                assert r["candidate_count"] in (9, 27)
                assert r["shortlist_size"] == r["candidate_count"] // 2
                key = construction, r["readout"], r["metric"], r["group"].split(":")[0]
                conditional[key].append(r)
    for r in summary["curves"]:
        group = curves[r["construction"], r["readout"], r["metric"], r["method"], r["shortlist_size"]]
        assert len(group) == (240 if r["method"] == "random" else 12)
        for field in ("test_loss", "absolute_omission", "recall_at_top_q"):
            assert np.isclose(r[field], np.mean([v[field] for v in group]), atol=1e-14)
    for r in summary["within_composition"]:
        group = conditional[r["construction"], r["readout"], r["metric"], r["method"]]
        assert len(group) == (7200 if r["method"] == "random" else 360)
        for field in ("oracle_retained", "absolute_omission"):
            assert np.isclose(r[field], np.mean([v[field] for v in group]), atol=1e-14)
    assert len(summary["curves"]) == 312 and len(summary["within_composition"]) == 60
    print("PASS: equal-cost repartition, 312 challenge curves, 60 within-composition summaries")


def verify_cost_results() -> None:
    revision = ROOT / "results/target_conditioned/revision_20260922"
    root = revision / "e2b_decision_cost_v1"
    observed = defaultdict(list)
    for encoder in ("resnet50", "dinov2_b14"):
        profile = json.loads((root / "feature_cost" / encoder / "profile.json").read_text())
        blocks = {r["block_id"]: r for r in profile["source_blocks"]}
        groups = {(r["domain"], r["role"]): r["seconds"] for r in profile["groups"]}
        rankings = json.loads((root / encoder / "original/screen/rankings.json").read_text())
        for readout, folder in (("ridge", "timing_ridge_float64"), ("logistic", "timing")):
            timing = json.loads((root / folder / encoder / "timing.json").read_text())
            if readout == "ridge":
                assert timing["ridge_precision"] == "float64"
            tests = {d: json.loads((root / encoder / "original/test-audit" /
                                   f"{d}__{readout}_losses.json").read_text()) for d in rankings}
            for r in timing["records"]:
                if r["readout"] != readout:
                    continue
                domain, method, size = r["target_domain"], r["method"], r["shortlist_size"]
                if method == "random":
                    prefix = rankings[domain][f"random:{r['repeat']}"][:size]
                    used = {n for c in prefix for n in c.split("|")}
                else:
                    used = {n for n, b in blocks.items() if b["domain"] != domain}
                assert all(blocks[n]["domain"] != domain for n in used)
                feature = profile["model_load_seconds"] + groups[domain, "target_validation"]
                feature += sum(blocks[n]["seconds"] for n in used)
                if method not in ("random", "exhaustive"):
                    feature += groups[domain, "target_selection"]
                values = {"resident_seconds": r["decision_seconds"], "feature_seconds": feature,
                          "raw_additive_seconds": feature + r["decision_seconds"]}
                values.update({"test_" + m: tests[domain][choice][m] for m, choice in r["selections"].items()})
                observed[readout, method, size].append(values)
    rows = read_csv(revision / "cost_summary/curves.csv")
    assert len(rows) == len(observed) == 32
    for r in rows:
        values = observed[r["readout"], r["method"], int(r["shortlist_size"])]
        assert len(values) == 36
        for field in values[0]:
            assert np.isclose(float(r[field]), np.mean([v[field] for v in values]), rtol=0, atol=1e-10), (r, field)
        assert np.isclose(float(r["raw_additive_seconds"]),
                          float(r["feature_seconds"]) + float(r["resident_seconds"]), atol=1e-10, rtol=0)
        assert np.isclose(float(r["cached_two_encoder_batch_seconds"]),
                          12 * float(r["resident_seconds"]) + float(r["batch_preparation_seconds"]), atol=1e-10, rtol=0)
    print("PASS: 32 matched quality/cost curves, exact timed choices, random block-union accounting")


def main() -> None:
    verify_method_naming()
    for name, expected in STYLE_SHA256.items():
        assert hashlib.sha256((STYLE / name).read_bytes()).hexdigest() == expected, name
    print("PASS: unmodified official style bundle")
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
    print("PASS: eight primary method means and twelve Target A-opt/MMD validation choices")

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
    def singular_rank(matrix):
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        probabilities = singular_values[singular_values > 0] / singular_values.sum()
        return np.exp(-np.sum(probabilities * np.log(probabilities)))
    example = np.diag([9.0, 1.0])
    assert np.isclose(singular_rank(example), 1.384145488461686, rtol=1e-12)
    assert np.isclose(singular_rank(np.eye(2)), 2.0)
    assert np.isclose(singular_rank(np.tile(example, (3, 1))), singular_rank(example))
    # Numerical sanity checks supplement, and do not replace, the written proofs.
    rng = np.random.default_rng(17)
    for _ in range(50):
        x, h = rng.normal(size=(9, 4)), rng.normal(size=(7, 4))
        g, c = x.T @ x, h.T @ h / len(h)
        ev, u = np.linalg.eigh(g)
        directional_score = np.sum(np.diag(u.T @ c @ u) / (1 + ev))
        assert np.isclose(directional_score, np.trace(c @ np.linalg.inv(np.eye(4) + g)))
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
        assert r"\paragraph{" not in source, (language, "run-in phrase heading")
        for table in re.findall(r"\\begin\{table\}.*?\\end\{table\}", source, re.S):
            assert not re.search(r"\\(?:small|footnotesize|scriptsize|tiny)\b", table), language
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
    submission = (HERE / "submission_metadata.md").read_text(encoding="utf-8")
    metadata = json.loads((HERE / "submission_metadata.json").read_text(encoding="utf-8"))
    generated_metadata = (HERE / "generated/submission_metadata.tex").read_text(encoding="utf-8")
    for lang in ("en", "zh"):
        assert metadata[f"title_{lang}"] in submission
        assert metadata[f"title_{lang}"] in generated_metadata
        assert all(keyword in submission for keyword in metadata[f"keywords_{lang}"])
    assert len(metadata["keywords_en"]) == len(metadata["keywords_zh"])
    supplements = manifest["supplemental_diagnostics_percent"]
    assert np.isclose(supplements["BootstrapRecall"], 97.42041666666668)
    assert np.isclose(supplements["BootstrapOverlap"], 85.15375000000002)
    assert np.isclose(supplements["BrierErrorOverlap"], 29.166666666666668)
    print("PASS: submission metadata and recorded supplemental statistics")
    followup = ROOT / "results/target_conditioned/revision_20260917/revision_results/e2b_readout_projection_v1/summary"
    groups = {}
    for row in read_csv(followup / "unit_rows.csv"):
        if int(row["projection_seed"]) == 20260911:
            continue
        key = (row["readout"], row["method"], row["metric"], row["shortlist_size"])
        groups.setdefault(key, {}).setdefault(row["target_domain"], []).append(row)
    for row in read_csv(followup / "new_seeds.csv"):
        domains = groups[row["readout"], row["method"], row["metric"], row["shortlist_size"]]
        assert len(domains) == 6 and all(len(task) == 8 for task in domains.values())
        for field in ("recall_at_top_q", "relative_candidate_span", "relative_regret", "absolute_omission"):
            mean = np.mean([np.mean([float(r[field]) for r in task]) for task in domains.values()])
            assert np.isclose(mean, float(row[field]), rtol=1e-12, atol=1e-15)
    followup_values = manifest["readout_followup_percent"]
    for name, expected in {
        "ReadoutRidgeA": 98.33333333333333,
        "ReadoutLogisticA": 98.33333333333333,
        "ReadoutRidgeMMD": 99.375,
        "ReadoutLogisticMMD": 98.95833333333333,
        "RidgeSpan": 0.247864,
        "LogisticSpan": 14.652095,
        "WorstErrorRecall": 40,
        "WorstErrorGap": 1.66015625,
    }.items():
        assert np.isclose(followup_values[name], expected, rtol=0, atol=1e-6), name
    print("PASS: 240 follow-up means independently recovered from domain-level repetitions")
    budget_root = ROOT / "results/target_conditioned/revision_20260922/budget_audit"
    budget = read_csv(budget_root / "curves.csv")
    full_losses = {}
    for job in sorted(followup.parent.glob("*__*/validate")):
        if job.parent.name.endswith("__20260911"):
            continue
        for path in sorted(job.glob("*__*_losses.json")):
            domain, readout = path.stem.removesuffix("_losses").split("__")
            validation = json.loads(path.read_text())
            test = json.loads((job.parent / "test-audit" / path.name).read_text())
            for metric in ("brier_score", "nll", "error_rate"):
                choice = min(validation, key=lambda c: (validation[c][metric], c))
                full_losses.setdefault((readout, metric), {}).setdefault(domain, []).append(test[choice][metric])
    for row in budget:
        domains = full_losses[row["readout"], row["metric"]]
        assert len(domains) == 6 and all(len(v) == 8 for v in domains.values())
        full = np.mean([np.mean(v) for v in domains.values()])
        assert np.isclose(float(row["full_validation_test_loss"]), full, atol=1e-14)
        assert np.isclose(float(row["test_loss"]) - full, float(row["gap_to_full"]), atol=1e-14)
        if row["method"] == "exhaustive":
            assert int(row["shortlist_size"]) == 455
            assert float(row["gap_to_full"]) == 0
    assert len(budget) == 96
    print("PASS: 96 decision-budget rows and validation-selected exhaustive references")
    verify_construction_results()
    verify_cost_results()
    for lang, stem, opening, heading in (
        ("en", "main", "Selecting source-data", r"^\s*1\s+I\s*NTRODUCTION"),
        ("zh", "main_zh", "为目标任务", r"^\s*1\s+引言"),
    ):
        abstract = " ".join((HERE / f"abstract_{lang}.txt").read_text(encoding="utf-8").split())
        assert abstract in md and abstract in submission
        assert len(re.findall(r"\d+(?:\.\d+)?%", abstract)) <= 2, (lang, "abstract result density")
        body = (HERE / f"body_{lang}.tex").read_text(encoding="utf-8")
        introduction = body.split(r"\section{")[1]
        assert r"\begin{itemize}" not in introduction
        assert r"\begin{enumerate}" not in introduction
        assert r"\label{sec:limitations}" in body
        assert body.count(r"\begin{algorithm}[H]") == 1
        assert body.index(r"\label{alg:target-a}") < body.index(r"\label{eq:risk}")
        algorithm = body.split(r"\begin{algorithm}[H]", 1)[1].split(r"\end{algorithm}", 1)[0]
        assert r"\widehat C_T(I+G_S)^{-1}" in algorithm
        assert not re.search(r"\\(?:small|footnotesize|scriptsize|tiny)\b", algorithm)
        tex = (HERE / f"generated/abstract_{lang}.tex").read_text(encoding="utf-8").strip()
        assert tex.replace(r"\%", "%") == abstract
        log = (HERE / f"{stem}.log").read_text(errors="replace")
        for problem in ("Overfull", "Missing character", "undefined", "There were undefined"):
            assert problem not in log, (stem, problem)
        first = text_from_pdf(HERE / f"{stem}.pdf", first_page=True)
        clean = "\n".join(line[6:] if re.match(r"^\d{3}(?:\s|$)", line) else line
                          for line in first.splitlines())
        assert normalized_rendering(metadata[f"title_{lang}"]).casefold() in normalized_rendering(clean).casefold()
        info = subprocess.check_output(["pdfinfo", str(HERE / f"{stem}.pdf")], text=True)
        assert metadata[f"title_{lang}"] in info
        assert ", ".join(metadata["keywords_en"]) in info
        end = re.search(heading, clean, re.M)
        assert end is not None
        actual = clean[clean.index(opening):end.start()]
        assert normalized_rendering(actual) == normalized_rendering(abstract), (lang, actual)
        full_text = text_from_pdf(HERE / f"{stem}.pdf")
        verify_format(stem, log, full_text)
        assert "\ufffd" not in full_text
        if lang == "zh":
            assert sum("\u4e00" <= c <= "\u9fff" for c in full_text) > 7000
        aux = (HERE / f"{stem}.aux").read_text(encoding="utf-8")
        end_page = int(re.search(r"\\newlabel\{main-text-end\}\{\{[^}]*\}\{(\d+)\}", aux).group(1))
        main_labels = set(re.findall(r"\\label\{((?:fig|tab|alg):[^}]+)\}", body))
        main_pages = [int(page) for label, page in re.findall(
            r"\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}", aux) if label in main_labels]
        assert len(main_pages) == len(main_labels)
        assert max(main_pages + [end_page]) <= 9
        if lang == "en":
            assert end_page == 9, ("main", "English Conclusion must end on page 9")
        print(f"PASS: {stem} title/keywords and abstract match submission/text/Markdown/PDF; main text ends on page {end_page}")
    print("All automated draft checks passed. Visual review and human scientific sign-off remain separate.")


if __name__ == "__main__":
    main()
