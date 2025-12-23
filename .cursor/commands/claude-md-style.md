# Claude.md (Directory-level) — Style Guide

## Purpose & placement
- **MUST** place the file at the target directory root: `./CLAUDE.md` (e.g., `/apps/web/CLAUDE.md`).
- **MUST** state scope at the top: “This file governs only `./` and subpaths.”
- **IMPORTANT:** When both root and directory guides exist, **MUST** treat this file as authoritative for its directory; **SHOULD** link to repo-wide rules in `/CLAUDE.md` if present.
- **SHOULD** link to sibling/parent docs using relative paths (e.g., `../README.md`, `../../docs/build.md`).

## Audience & tone
- **MUST** address Claude directly (“You **MUST**…”).
- **SHOULD** write as precise, command-style bullets (≤ ~200 words/section).
- **AVOID** hedging (“try”, “maybe”); prefer testable wording.

## Structure & formatting
- **MUST** use headings + bullets only; no prose paragraphs.
- **IMPORTANT:** **MUST** distinguish requirements (**MUST**) vs preferences (**SHOULD**) vs prohibitions (**AVOID**).
- **MUST** bold **MUST/SHOULD/AVOID** and prefix criticals with **IMPORTANT:**.
- **MUST** reference exact paths (e.g., `./src/index.ts`, `../shared/config.ts`).
- **SHOULD** include working code/command blocks with language fences.
- **AVOID** absolute URLs for internal docs; **SHOULD** use relative links and anchors (e.g., `../README.md#development`).
- Provide example patterns where necessary.

## Scope & precedence
- **MUST** enumerate in-scope tasks for this directory (build, tests, codegen).
- **MUST** list “do-not-touch” areas (e.g., `../infra/*`, prod configs) with reasons.
- **SHOULD** declare precedence rules when this file conflicts with repo-wide guidance (directory file **wins** here).
- **SHOULD** link to parent policies for shared rules (security, release, coding style).

## Path & linking rules
- **MUST** use relative links that resolve from this file’s directory.
- **SHOULD** link specific sections via heading anchors (e.g., `../docs/ops.md#rollbacks`).

## Template (copy into `./CLAUDE.md`)

````md
# CLAUDE.md — <directory name> scope

## Scope
- **MUST** apply only to `./` and children.
- **SHOULD** also read: `../../CLAUDE.md`.

## Layout
- 

## Tasks You’ll Do Here
- Build/test entry points: `./src/…`, tests: `./tests/…`.

## Canonical commands (optional)
```bash
< commands + descriptions >
```

## Rules
- 

## Tests
- 

## Links to directory guides (optional)
- < only include links to guides in child directories >

## Common pitfalls and fixes
- < common errors and how to fix them >
```
