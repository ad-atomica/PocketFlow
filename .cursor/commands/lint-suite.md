# Fix Lint Issues

## Overview

Run project linters, apply fixes, and ensure the codebase meets formatting and
style requirements before merging changes.

## Steps

1. **Execute linters**
    - Run this project's lint commands: `mise run fix`, `mise run check`.
    - Capture any remaining errors or warnings from the output
    - Identify files requiring manual attention
2. **Resolve findings**
    - Apply targeted fixes, keeping edits minimal and idiomatic
    - Refactor repeated issues such as unused variables or long functions
    - Update configuration or suppressions only when justified
3. **Verify cleanliness**
    - Re-run the lint command to ensure a zero-issue result
    - Spot-check key files for formatting and readability

## Lint Checklist

- [ ] Linter executed with latest config
- [ ] Autofix results reviewed
- [ ] Manual issues resolved
- [ ] Final lint run passes cleanly
- [ ] Changes staged or ready for PR
