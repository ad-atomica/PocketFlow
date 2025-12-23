# Fix Merge Conflicts

## Overview

Help resolve merge conflicts in the highlighted/selected text by identifying the conflicting section, understanding the context of both changes, and providing a clean resolution that preserves the intended functionality from both branches.

## Steps

1. **Identify the Highlighted Conflict**
    - Focus on the specific conflict section that is selected
    - Locate the merge conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in the selection
    - Understand the scope and nature of this specific conflict
2. **Understand Context**
    - Review the changes from both branches (HEAD and incoming) for this conflict
    - Understand the intent behind each conflicting change
    - Check related code in the same file that might provide context
    - Review commit messages or PR descriptions for context if needed
3. **Resolve the Conflict**
    - Ensure the resolved code maintains functionality from both branches when appropriate
    - Remove conflict markers and ensure proper code formatting
    - Verify imports, dependencies, and references are correct for the resolved section
4. **Validate Resolution**
    - Check that the resolved code maintains proper syntax and structure
    - Ensure the resolution aligns with project conventions and standards
    - Verify no duplicate code or logic was introduced in the resolved section

## Fix Merge Conflicts Checklist

- [ ] Identified the conflict markers in the highlighted/selected text
- [ ] Understood the context and intent of changes from both branches for this conflict
- [ ] Reviewed related code in the same file that provides context
- [ ] Reviewed related code in other parts of the code base that provides context
- [ ] Chose appropriate resolution strategy for this conflict
- [ ] Resolved conflict while preserving intended functionality
- [ ] Removed all conflict markers from the selection
- [ ] Ensured proper code formatting and style
- [ ] Verified imports and dependencies are correct
- [ ] Verified no duplicate code or logic introduced
- [ ] Ensured resolution follows project conventions

