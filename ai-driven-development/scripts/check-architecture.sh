#!/usr/bin/env bash
set -euo pipefail

# Clean Architecture Dependency Guard
# Validates that inner layers do not import from outer layers.
#
# Layer hierarchy (inner -> outer):
#   domain -> application -> infrastructure / presentation

ERRORS=0
TS_SRC="apps/desktop/src"
PY_SRC="packages/analysis/src"

echo "=== Clean Architecture Dependency Guard ==="

# --- TypeScript checks (path aliases AND relative imports) ---

# domain must NOT import from application, infrastructure, or presentation
echo ""
echo "--- TypeScript: domain layer ---"
if grep -rn --include="*.ts" --include="*.tsx" \
    -E '(@application|@infrastructure|@presentation|from\s+["\x27]\.\./(application|infrastructure|presentation))' \
    "$TS_SRC/domain/" 2>/dev/null; then
  echo "VIOLATION: domain imports from outer layers"
  ERRORS=$((ERRORS + 1))
else
  echo "OK: domain has no forbidden imports"
fi

# application must NOT import from infrastructure or presentation
echo ""
echo "--- TypeScript: application layer ---"
if grep -rn --include="*.ts" --include="*.tsx" \
    -E '(@infrastructure|@presentation|from\s+["\x27]\.\./(infrastructure|presentation))' \
    "$TS_SRC/application/" 2>/dev/null; then
  echo "VIOLATION: application imports from outer layers"
  ERRORS=$((ERRORS + 1))
else
  echo "OK: application has no forbidden imports"
fi

# --- Python checks (relative and absolute imports) ---

if [ -d "$PY_SRC" ]; then
  echo ""
  echo "--- Python: domain layer ---"
  if grep -rn --include="*.py" \
      -E '(from\s+(\.\.?)?\.?(application|infrastructure|presentation)|import\s+(application|infrastructure|presentation))' \
      "$PY_SRC/domain/" 2>/dev/null; then
    echo "VIOLATION: Python domain imports from outer layers"
    ERRORS=$((ERRORS + 1))
  else
    echo "OK: Python domain has no forbidden imports"
  fi

  echo ""
  echo "--- Python: application layer ---"
  if grep -rn --include="*.py" \
      -E '(from\s+(\.\.?)?\.?(infrastructure|presentation)|import\s+(infrastructure|presentation))' \
      "$PY_SRC/application/" 2>/dev/null; then
    echo "VIOLATION: Python application imports from outer layers"
    ERRORS=$((ERRORS + 1))
  else
    echo "OK: Python application has no forbidden imports"
  fi
fi

echo ""
if [ "$ERRORS" -gt 0 ]; then
  echo "FAILED: $ERRORS architecture violation(s) found"
  exit 1
else
  echo "PASSED: All architecture dependency rules satisfied"
fi
