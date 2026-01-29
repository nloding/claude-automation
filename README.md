# Claude Automation

A personal experiment for running pre-built Claude Code prompts sequentially.

## What it does

This CLI tool uses the Claude Code SDK to run multiple prompts one after another, with automatic error detection and early termination on failure. Useful for automating repetitive multi-step tasks.

## Installation

```bash
pip install -r requirements.txt
```

Also requires Claude Code CLI, of course.

## Usage

Run prompts from command line:
```bash
python cli.py "First prompt" "Second prompt" "Third prompt"
```

Run prompts from a directory (one prompt per file, processed alphabetically):
```bash
python cli.py --dir ./prompts-example/
```

Run a single prompt from a file:
```bash
python cli.py --file prompt.txt
```

### Options

| Flag | Description |
|------|-------------|
| `-d, --dir` | Directory containing prompt files |
| `-f, --file` | Single file containing a prompt |
| `--stop-on-error` | Stop on first error (default: true) |
| `--no-stop-on-error` | Continue even after errors |
| `--stop-on-warning` | Stop on first warning |
| `--tools` | Comma-separated list of allowed tools |
| `--working-dir` | Working directory for Claude Code |
| `--max-turns` | Max agentic turns per prompt |
| `-v, --verbose` | Enable verbose output |

## Prompt Files

Create a directory with numbered files to control execution order:
```
prompts/
  01-analyze.txt
  02-find-todos.txt
  03-check-errors.txt
```

Each file contains a single prompt (the entire file content is sent to Claude).
