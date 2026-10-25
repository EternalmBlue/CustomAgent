# Reference Fixtures

Applies to this directory. Read ../AGENTS.md and the tools.py contract in
../../src/custom_agent/AGENTS.md.

This directory holds tool-readable UTF-8 fixtures, not development specifications.
read_docs can read this directory as part of the configured workspace. Its root
is selected by config.toml's workspace_root, not this fixture directory.

Keep sample.md available at tests/fixtures/sample.md within the project workspace.
Its recovery marker
ORCHID-742 supports tool, evidence-retrieval and live acceptance checks.
Preserve the fixture's explicit mock search/weather and user/session ownership
statements. Do not insert credentials or private workspace content.
