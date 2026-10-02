---
name: format-export
description: Use when the user runs "/format-export", or asks to "reformat that export", "format the log I exported", or "make the export pretty" — converts a raw Claude Code `/export` transcript (.txt) into the user's personal aesthetic markdown style, modeled on `example-log.txt` in this directory.
version: 1.1.0
---

# Format Export

Converts a raw `/export` transcript into the user's hand-curated markdown aesthetic: colored
`<span>` markup, reconstructed wrapped paragraphs, and specific line-break/escaping conventions.
The canonical reference is `example-log.txt` in this directory — **always re-read it before
formatting**, and re-check your output against it for any overlapping content. The rules below
were reverse-engineered by diffing it against its source raw export
(`2026-06-23-131900-26623-read-the-handoffmd-file-and-give-of-b.txt`) and confirmed line-by-line.
**This is genuinely judgment-based, not a fixed algorithm** — a first mechanical attempt at this
got the paragraph boundaries wrong in several places (see "Paragraph segmentation" below). Don't
try to reduce this to a regex; read each chunk and decide.

## Input / Output

- **Input**: path to a raw exported transcript `.txt` file. If no argument given, use the most
  recently modified file in the cwd matching the `/export` naming pattern
  (`YYYY-MM-DD-HHMMSS-...txt`), excluding `example-log.txt` itself.
- **Output**: a **new** file, never overwrite the raw export: `<original-stem>-formatted.md` in
  the same directory.
- Never modify `example-log.txt` — it's the permanent style reference.
- Process the **entire** raw file, not just the part that overlaps with the example. Don't drop
  or summarize any turn.

## Step 1 — Fixed header

Every output file starts with exactly:
```
* * *

* * *

```
(two markdown horizontal rules, blank line after each), unconditionally.

## Step 2 — The banner (special case, hardcode this)

The Claude Code startup banner is always 3 distinct fields (ASCII-art logo, version/model line,
cwd line) — **not** a single wrapped sentence, so it is never joined. Each of its 3 lines is its
own `<span style="color: rgb(191, 237, 210);">...</span>`, each ending in a hard break (two
trailing spaces) except handled per Step 5. Leading whitespace: see `example-log.txt` lines 5-7
for the exact precedent (first line: `&nbsp;` glued directly to `<span>`; second line: no
prefix; third line: 2 literal leading spaces). Reuse this pattern verbatim, substituting the
actual version/model/directory text.

## Step 3 — Color palette

Via `<span style="color: rgb(R,G,B);">...</span>`:
- `rgb(191, 237, 210)` (green): banner, and **genuine assistant prose** — actual text responses
  to the user.
- `rgb(126, 140, 141)` (gray): everything system/meta — `⎿` tool-result lines, `✻ Churned/
  Crunched/Sautéed for Xs` timing lines, `Thought for Xs...` lines, `※ recap:` lines, tool
  invocation lines (`● Bash(...)`, `● Read(...)`, `● Explore(...)`, etc.), and `●`-prefixed
  system/plan-management notices ("Updated plan", "User approved Claude's plan"). **Test for a
  `●` line: is Claude actually saying something substantive to the user, or reporting a
  side-effect/status?** First → green. Second → gray.
- `❯` user-input lines: **no span** — plain text, but still go through the same paragraph-join
  and escaping treatment as everything else (see Step 4 — confirmed against the example: a
  3-line wrapped user prompt becomes one plain unstyled line, words rejoined with single spaces,
  no per-line spans).

## Step 4 — Paragraph segmentation and joining (the hard part — use judgment)

The raw export hard-wraps text at the terminal's column width. Reconstructing it means deciding,
for each contiguous non-blank chunk, where one logical "unit" ends and the next begins, then
**joining all physical lines of one unit into a single markdown output line** (plain text for
`❯` lines, or one `<span>` per original physical line, concatenated space-separated, for colored
content).

A unit boundary is **not** simply "blank line" — confirmed cases from the example:
- A multi-line `❯` user prompt (3 physical lines, one sentence) → joined into **one** line.
- A `●` headline that wraps onto a second physical line *with no blank line between* (because
  it's the same sentence, e.g. "● The error bars come from TEfficiency::CreateGraph() (line
  59-67), which is" / "ROOT's standard binomial-efficiency uncertainty:") → joined into **one**
  line, both spans, headline's color.
- A `●` headline followed by a **blank line** then a separate paragraph (e.g. "● Summary of
  HANDOFF.md...:" then a blank line then "Main investigation...") → **two** separate units, not
  joined, even though both are green and adjacent.
- A numbered list (`1. foo` / `2. bar`, no blank lines between items in the raw file) → **each
  numbered item is its own unit** — wrap-continuation lines belonging to item N join with item
  N's first line, but item N+1 starts a fresh unit. (Confirmed: "Outstanding items:" / "1. Revert
  ..." / "2. git stash pop..." each render as separate output lines, not one giant merged line,
  even though there's no blank line anywhere in that raw block.)
- A `⎿` tool-output block with multiple lines of literal content (e.g. grep/code output) is
  **not** reflowed into prose — keep each physical line of literal content visually distinct
  (still gray-spanned, but don't merge unrelated output lines into one run-on sentence).

**Rule of thumb**: ask "if I read this physical line and the next one aloud, are they the same
sentence/field broken by column width, or are they two distinct things that happen to be
adjacent?" Join only the former.

## Step 5 — Hard line breaks

After a unit's single output line, append exactly two trailing spaces if the next unit is
non-blank (forces a markdown `<br>`); omit them if the next unit is blank (natural paragraph
break already applies). Collapse multiple consecutive blank lines in the raw file into exactly
one blank line in the output.

## Step 6 — Leading whitespace / `&nbsp;`

Not perfectly formula-driven (confirmed: the 3-line banner alone uses two different conventions
for the same nominal 2-space indent). General guidance:
- A single leading space in the source often becomes a bare `&nbsp;` glued directly to the
  `<span>` (no space between).
- Glyph-prefixed lines (`⎿`/`●`/`❯`/`✻`/`※`) tend to keep literal leading spaces.
- Indented prose continuations (no glyph at the very start) tend to use the `&nbsp;` conversion.
- When unsure, match whichever precedent in `example-log.txt` is closest to the line type you're
  formatting.

## Step 7 — Markdown escaping (confirmed necessary — content sits inside spans/plain text, but
markdown still parses the surrounding line)

- A line/span starting with a digit + period that looks like an ordered-list marker: escape the
  period — `1.` → `1\.` (confirmed: "Outstanding items:" list).
- A line/span starting with `- ` (would look like a bullet): escape the hyphen — `- ` → `\- `
  (confirmed: the four `\-`-prefixed bullets in the error-bars explanation).
- Underscores and asterisks that could pair up into unintended emphasis: escape **the first of
  the pair** (not necessarily every occurrence) so CommonMark can't find a matching delimiter —
  confirmed example: `_ef_*` → `\_ef_\*` (first underscore and the asterisk escaped; the middle
  underscore left alone because escaping the first one already breaks the pairing). Apply the
  same minimal-escaping judgment elsewhere underscores/asterisks appear in identifiers.
- Square brackets that could be misread as link syntax: `[eyl, eyh]` → `\[eyl, eyh\]`.
- When in doubt, escape the minimum needed to stop CommonMark from interpreting it as structure,
  not every special character on principle — over-escaping (e.g. `\_lower` when only one
  underscore needs it) is visually noisy and not what the example does everywhere.

## Step 8 — Preserve content exactly

All original wording, unicode glyphs (`❯ ⎿ ● ✻ ※ ▐▛███▜▌ ▝▜█████▛▘ ▘▘▝▝`), code, paths, and
blank-line paragraph breaks carry over. Only structure/markup changes — never edit, summarize,
or correct the original text. If you hit a content/message type with no precedent in
`example-log.txt` (multi-line `⎿` blocks with `… +N lines (ctrl+o to expand)`, image-read
confirmations, error blocks, Explore-agent results), extend these same patterns by analogy and
flag genuinely ambiguous cases in your summary rather than guessing silently.

## Verification

After writing the output, diff it against `example-log.txt` for whatever portion of the input
overlaps with the example's source conversation — color classification, the header, hard-breaks,
unit-joining, and escaping should match. Leading-whitespace characters may differ slightly per
Step 6; that's expected.
