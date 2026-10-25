import pytest

from custom_agent.evaluate import answer_matches_expected, compare


@pytest.mark.parametrize(
    ("text", "expected"),
    [("B", True), (" **B**. ", True), ("Storage A remains our best choice.", False), ("A", False)],
)
def test_decision_answer_requires_expected_choice(text, expected):
    assert answer_matches_expected(text, "B") is expected


async def test_snapshot_comparison_reports_deterministic_evidence_probe(tmp_path):
    output = tmp_path / "comparison.json"
    report = await compare(output)
    assert output.is_file()
    assert report["mode"] == "deterministic_context_probe"
    rows = report["results"]
    assert len(rows) == 9
    for policy in ("full_history", "observation_masking", "structured_memory"):
        assert len([row for row in rows if row["policy"] == policy]) == 3
    structured = [row for row in rows if row["policy"] == "structured_memory"]
    assert all(row["evidence_probe_passed"] for row in structured)
