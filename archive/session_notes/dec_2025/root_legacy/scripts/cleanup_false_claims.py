#!/usr/bin/env python3
"""
Systematic detection and reporting of false claims across documentation.

Usage:
    python scripts/cleanup_false_claims.py --scan    # Scan and report
    python scripts/cleanup_false_claims.py --report  # Generate detailed report
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# False claims patterns with severity levels
FALSE_CLAIM_PATTERNS = {
    # CRITICAL: Numerically wrong
    r"88%.*recovery": ("CRITICAL", "Should be 0% (bug fix)"),
    r"8\.8/10": ("CRITICAL", "Should be 0/10 (bug fix)"),
    r"similarity.*1\.[2-9]": ("CRITICAL", "Max cosine similarity is 1.0, this is impossible"),
    r"1\.28.*similarity": ("CRITICAL", "Should be ~0.39 (bug fix)"),

    # HIGH: Rejected hypotheses
    r"basis ambiguity": ("HIGH", "Hypothesis rejected - subspace overlap is 14%, not 90%"),
    r"different bases.*same subspace": ("HIGH", "Rejected - SAEs learn orthogonal subspaces"),

    # MEDIUM: Overclaiming
    r"across ALL architectures": ("MEDIUM", "Should be 'for TopK architecture' (only TopK verified)"),
    r"multi-architecture.*verification": ("MEDIUM", "Only TopK thoroughly tested"),
    r"stability.*ALL.*architecture": ("MEDIUM", "Only TopK shows clear pattern"),

    # LOW: Incomplete/misleading
    r"sparse.*improves.*stability": ("LOW", "Should note this FAILED (negative result)"),
    r"sparse ground truth.*high.*stability": ("LOW", "Theory predicted high, but observed low"),
}

# Directories to skip
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "_archived_buggy_experiments",
    "_archive_rejected_hypotheses",
    "venv",
    ".venv",
}


def should_skip_path(path: Path) -> bool:
    """Check if path should be skipped."""
    return any(skip_dir in path.parts for skip_dir in SKIP_DIRS)


def scan_file(file_path: Path) -> List[Tuple[str, int, str, str]]:
    """
    Scan a single file for false claims.

    Returns:
        List of (pattern, line_num, line_text, severity, explanation)
    """
    if should_skip_path(file_path):
        return []

    findings = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()

            for pattern, (severity, explanation) in FALSE_CLAIM_PATTERNS.items():
                if re.search(pattern, line_lower):
                    findings.append((
                        pattern,
                        line_num,
                        line.strip(),
                        severity,
                        explanation
                    ))
    except (UnicodeDecodeError, PermissionError):
        pass  # Skip binary files or inaccessible files

    return findings


def scan_repository(root_dir: Path = Path(".")) -> Dict[Path, List]:
    """Scan entire repository for false claims."""
    all_findings = {}

    # Scan markdown files
    for md_file in root_dir.rglob("*.md"):
        findings = scan_file(md_file)
        if findings:
            all_findings[md_file] = findings

    # Also scan Python files (docstrings)
    for py_file in root_dir.rglob("*.py"):
        findings = scan_file(py_file)
        if findings:
            all_findings[py_file] = findings

    return all_findings


def print_summary(all_findings: Dict):
    """Print summary of findings."""
    total_files = len(all_findings)
    total_issues = sum(len(findings) for findings in all_findings.values())

    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for findings in all_findings.values():
        for _, _, _, severity, _ in findings:
            severity_counts[severity] += 1

    print("\n" + "="*80)
    print("FALSE CLAIMS SCAN SUMMARY")
    print("="*80)
    print(f"\nFiles with issues: {total_files}")
    print(f"Total issues found: {total_issues}")
    print(f"\nBy severity:")
    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = severity_counts[severity]
        print(f"  {severity:8s}: {count:3d}")
    print()


def print_detailed_findings(all_findings: Dict):
    """Print detailed findings."""
    for file_path, findings in sorted(all_findings.items()):
        print(f"\n{'='*80}")
        print(f"FILE: {file_path}")
        print(f"{'='*80}")

        for pattern, line_num, line_text, severity, explanation in findings:
            print(f"\n[{severity}] Line {line_num}:")
            print(f"  Pattern: {pattern}")
            print(f"  Text: {line_text[:100]}...")
            print(f"  → {explanation}")


def generate_markdown_report(all_findings: Dict, output_file: Path):
    """Generate markdown report of findings."""
    with open(output_file, 'w') as f:
        f.write("# False Claims Detection Report\n\n")
        f.write("**Generated:** Automated scan\n\n")

        # Summary
        total_files = len(all_findings)
        total_issues = sum(len(findings) for findings in all_findings.values())
        f.write(f"**Files with issues:** {total_files}\n")
        f.write(f"**Total issues:** {total_issues}\n\n")

        # Severity breakdown
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for findings in all_findings.values():
            for _, _, _, severity, _ in findings:
                severity_counts[severity] += 1

        f.write("## By Severity\n\n")
        f.write("| Severity | Count |\n")
        f.write("|----------|-------|\n")
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            f.write(f"| {severity} | {severity_counts[severity]} |\n")
        f.write("\n")

        # By file
        f.write("## Detailed Findings\n\n")
        for file_path, findings in sorted(all_findings.items()):
            f.write(f"### `{file_path}`\n\n")
            f.write(f"Issues found: {len(findings)}\n\n")

            for pattern, line_num, line_text, severity, explanation in findings:
                f.write(f"#### Line {line_num} [{severity}]\n\n")
                f.write(f"**Pattern:** `{pattern}`\n\n")
                f.write(f"**Text:**\n```\n{line_text}\n```\n\n")
                f.write(f"**Action required:** {explanation}\n\n")

        # Action summary
        f.write("\n## Action Items\n\n")
        f.write("### Priority 1: CRITICAL Issues\n\n")
        critical_files = {
            file_path: findings
            for file_path, findings in all_findings.items()
            if any(sev == "CRITICAL" for _, _, _, sev, _ in findings)
        }
        for file_path in sorted(critical_files.keys()):
            f.write(f"- [ ] `{file_path}` - Fix numerical errors\n")

        f.write("\n### Priority 2: HIGH Issues\n\n")
        high_files = {
            file_path: findings
            for file_path, findings in all_findings.items()
            if any(sev == "HIGH" for _, _, _, sev, _ in findings)
        }
        for file_path in sorted(high_files.keys()):
            f.write(f"- [ ] `{file_path}` - Remove rejected hypotheses\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scan repository for false claims"
    )
    parser.add_argument(
        '--scan',
        action='store_true',
        help='Scan and print summary'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed markdown report'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Print detailed findings to console'
    )

    args = parser.parse_args()

    if not (args.scan or args.report or args.detailed):
        parser.print_help()
        return

    print("Scanning repository for false claims...")
    all_findings = scan_repository()

    if args.scan or args.detailed:
        print_summary(all_findings)

    if args.detailed:
        print_detailed_findings(all_findings)

    if args.report:
        report_file = Path("FALSE_CLAIMS_DETECTION_REPORT.md")
        generate_markdown_report(all_findings, report_file)
        print(f"\n✅ Detailed report saved to: {report_file}")


if __name__ == "__main__":
    main()
