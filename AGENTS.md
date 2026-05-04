# Codex Instruction

## Communication

- Use Traditional Chinese for all user-facing responses by default.
- English is allowed and preferred for:
  - Technical terms (e.g., APIs, libraries, error messages)
  - Code and comments
  - Tool usage (e.g., search queries, commands)
- Do not translate standard technical terms unnecessarily.

- Keep explanations simple, concrete, and easy to follow, as the user has a limited programming background.

- When explaining code or technical decisions, always include:
  - Why this approach is used
  - What impact and trade-offs it introduces

- Keep `README.md` up to date. It is the source of truth for current project status.

## Workflow (Execution Steps)

- First, clarify the actual requirement.
- Then, propose a Minimum Viable Solution (MVS).
- Only after that, consider adding complexity if needed.

- Clearly state:
  - Known constraints
  - Current assumptions
  - Unverified or unclear parts

- If critical requirements or assumptions are unclear, ask for clarification instead of making decisions.

- Avoid adding features, abstractions, or flexibility that were not explicitly requested.

- If modifying existing code:
  - Make the smallest possible change that solves the problem.

## Problem-Solving (Mental Checklist)

Before implementing or making decisions, check:

- What is the actual goal?
- What is the simplest solution that works?
- Am I making any unstated assumptions?
- Is this adding unnecessary complexity?
- Is there a simpler or more maintainable alternative?

## Python Development

- Prefer using `uv` for Python projects to manage dependencies, virtual environments, lockfiles, and command execution.

## Agent skills

### Issue tracker

Issues and PRDs for this repo live in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

This repo uses the default five triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses a single-context domain docs layout. See `docs/agents/domain.md`.
