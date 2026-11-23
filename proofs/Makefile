.PHONY: build cache audit roadmap check-fixtures mdgen clean

LIB := DecisionAnalysisProofs

# Fetch prebuilt Mathlib oleans (run once per Mathlib revision)
cache:
	lake exe cache get

# Build the default target (everything except Roadmap.lean); must be sorry-free
build:
	lake build --wfail

# Reject missing audit results, nonstandard axioms and source proof escape hatches.
audit:
	@mkdir -p .lake
	@lake env lean Audit.lean > .lake/audit.log 2>&1 || { cat .lake/audit.log; exit 1; }
	@cat .lake/audit.log
	python3 scripts/audit.py .lake/audit.log

# Build the remaining-work module (prose only; no declarations)
roadmap:
	lake build $(LIB).Roadmap

# Run check_certificate on every fixture and compare the exit codes with EXPECTED.md
check-fixtures:
	lake build check_certificate
	sh scripts/check_fixtures.sh

# One Markdown file per Lean module, from the module docstrings (the .md/.html/.pdf documents
# of the sibling projects are not generated here yet).
mdgen:
	rm -rf .mdgen_tmp
	lake exe mdgen $(LIB) .mdgen_tmp

clean:
	rm -rf .mdgen_tmp
