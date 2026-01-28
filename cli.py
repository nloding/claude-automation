#!/usr/bin/env python3
"""
Claude Automation CLI

A CLI tool for running multiple Claude Code prompts sequentially,
with automatic error detection and early termination on failure.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

try:
    from claude_code_sdk import query, ClaudeCodeOptions
except ImportError:
    print("Error: claude-code-sdk not installed.")
    print("Install with: pip install claude-code-sdk")
    sys.exit(1)


# Default error keywords to detect failures
DEFAULT_ERROR_KEYWORDS = [
    "error",
    "failed",
    "exception",
    "cannot",
    "unable",
    "traceback",
    "fatal",
    "abort",
]

# Default warning keywords
DEFAULT_WARNING_KEYWORDS = [
    "warning",
    "deprecated",
    "caution",
]


async def run_prompt(
    prompt: str,
    allowed_tools: Optional[list[str]] = None,
    error_keywords: Optional[list[str]] = None,
    warning_keywords: Optional[list[str]] = None,
    working_dir: Optional[str] = None,
    max_turns: Optional[int] = None,
    verbose: bool = False,
) -> tuple[bool, bool, str]:
    """
    Run a single prompt and detect success/failure/warnings.

    Returns:
        tuple of (success, has_warnings, result_text)
    """
    result_text = ""
    had_error = False
    had_warning = False

    error_keywords = error_keywords or DEFAULT_ERROR_KEYWORDS
    warning_keywords = warning_keywords or DEFAULT_WARNING_KEYWORDS

    options_kwargs = {}
    if allowed_tools:
        options_kwargs["allowed_tools"] = allowed_tools
    if max_turns:
        options_kwargs["max_turns"] = max_turns
    if working_dir:
        options_kwargs["cwd"] = working_dir

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeCodeOptions(**options_kwargs) if options_kwargs else None,
        ):
            if verbose:
                print(f"  [DEBUG] Message type: {type(message).__name__}")

            # Capture the final result
            if hasattr(message, "result"):
                result_text = message.result

            # Check for tool errors
            if hasattr(message, "is_error") and message.is_error:
                had_error = True
                if verbose:
                    print(f"  [DEBUG] Tool error detected")

    except Exception as e:
        return (False, False, f"Exception: {str(e)}")

    # Check result text for error indicators
    result_lower = result_text.lower()

    for keyword in error_keywords:
        if keyword.lower() in result_lower:
            had_error = True
            if verbose:
                print(f"  [DEBUG] Error keyword found: {keyword}")
            break

    for keyword in warning_keywords:
        if keyword.lower() in result_lower:
            had_warning = True
            if verbose:
                print(f"  [DEBUG] Warning keyword found: {keyword}")
            break

    return (not had_error, had_warning, result_text)


async def run_prompts_sequential(
    prompts: list[tuple[str, str]],
    stop_on_error: bool = True,
    stop_on_warning: bool = False,
    allowed_tools: Optional[list[str]] = None,
    error_keywords: Optional[list[str]] = None,
    warning_keywords: Optional[list[str]] = None,
    working_dir: Optional[str] = None,
    max_turns: Optional[int] = None,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """
    Run multiple prompts sequentially.

    Args:
        prompts: list of (name, prompt_content) tuples

    Returns:
        tuple of (completed_count, error_count, warning_count)
    """
    completed = 0
    errors = 0
    warnings = 0

    for i, (name, prompt) in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Running: {name}")
        print(f"  {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print("-" * 60)

        success, has_warning, result = await run_prompt(
            prompt=prompt,
            allowed_tools=allowed_tools,
            error_keywords=error_keywords,
            warning_keywords=warning_keywords,
            working_dir=working_dir,
            max_turns=max_turns,
            verbose=verbose,
        )

        # Print result (truncated if very long)
        if result:
            result_display = result[:500] + "..." if len(result) > 500 else result
            print(f"\nResult:\n{result_display}")

        if not success:
            errors += 1
            print(f"\n[ERROR] {name} failed.")
            if stop_on_error:
                print("Stopping due to error (--stop-on-error is enabled).")
                break
        elif has_warning:
            warnings += 1
            completed += 1
            print(f"\n[WARNING] {name} completed with warnings.")
            if stop_on_warning:
                print("Stopping due to warning (--stop-on-warning is enabled).")
                break
        else:
            completed += 1
            print(f"\n[OK] {name} completed successfully.")

    return (completed, errors, warnings)


def load_prompt_from_file(filepath: Path) -> str:
    """Load a single prompt from a file (entire file content is the prompt)."""
    with open(filepath, "r") as f:
        return f.read().strip()


def load_prompts_from_directory(dirpath: Path) -> list[tuple[str, str]]:
    """
    Load prompts from all files in a directory, sorted alphabetically.

    Returns:
        list of (filename, prompt_content) tuples
    """
    prompts = []

    # Get all files in directory, sorted alphabetically
    files = sorted([f for f in dirpath.iterdir() if f.is_file()])

    for filepath in files:
        prompt_content = load_prompt_from_file(filepath)
        if prompt_content:
            prompts.append((filepath.name, prompt_content))

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Run Claude Code prompts sequentially with error detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run prompts from command line
  python cli.py "First prompt" "Second prompt" "Third prompt"

  # Run prompts from a directory (one prompt per file, alphabetical order)
  python cli.py --dir ./prompts/

  # Run a single prompt from a file
  python cli.py --file prompt.txt

  # Stop on warnings too
  python cli.py --dir ./prompts/ --stop-on-warning

  # Allow specific tools
  python cli.py --dir ./prompts/ --tools "Read,Edit,Bash"

  # Custom error keywords
  python cli.py "Do something" --error-keywords "error,fail,crash"
        """,
    )

    parser.add_argument(
        "prompts",
        nargs="*",
        help="Prompts to run (if not using --file)",
    )

    parser.add_argument(
        "-d", "--dir",
        type=Path,
        help="Directory containing prompt files (one prompt per file, processed alphabetically)",
    )

    parser.add_argument(
        "-f", "--file",
        type=Path,
        help="Single file containing a prompt",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        default=True,
        help="Stop execution on first error (default: True)",
    )

    parser.add_argument(
        "--no-stop-on-error",
        action="store_true",
        help="Continue execution even after errors",
    )

    parser.add_argument(
        "--stop-on-warning",
        action="store_true",
        default=False,
        help="Stop execution on first warning (default: False)",
    )

    parser.add_argument(
        "--tools",
        type=str,
        help="Comma-separated list of allowed tools (e.g., 'Read,Edit,Bash')",
    )

    parser.add_argument(
        "--error-keywords",
        type=str,
        help="Comma-separated list of error keywords to detect",
    )

    parser.add_argument(
        "--warning-keywords",
        type=str,
        help="Comma-separated list of warning keywords to detect",
    )

    parser.add_argument(
        "--working-dir",
        type=str,
        help="Working directory for Claude Code",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum number of agentic turns per prompt",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Collect prompts as list of (name, content) tuples
    prompts: list[tuple[str, str]] = []

    if args.dir:
        if not args.dir.exists():
            print(f"Error: Directory not found: {args.dir}")
            sys.exit(1)
        if not args.dir.is_dir():
            print(f"Error: Not a directory: {args.dir}")
            sys.exit(1)
        prompts = load_prompts_from_directory(args.dir)
    elif args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        content = load_prompt_from_file(args.file)
        if content:
            prompts = [(args.file.name, content)]
    elif args.prompts:
        # Command-line prompts: use index as name
        prompts = [(f"prompt-{i}", p) for i, p in enumerate(args.prompts, 1)]
    else:
        parser.print_help()
        sys.exit(1)

    if not prompts:
        print("Error: No prompts found.")
        sys.exit(1)

    # Parse options
    stop_on_error = args.stop_on_error and not args.no_stop_on_error
    allowed_tools = args.tools.split(",") if args.tools else None
    error_keywords = args.error_keywords.split(",") if args.error_keywords else None
    warning_keywords = args.warning_keywords.split(",") if args.warning_keywords else None

    print(f"Running {len(prompts)} prompt(s) sequentially...")
    print(f"  Stop on error: {stop_on_error}")
    print(f"  Stop on warning: {args.stop_on_warning}")
    if allowed_tools:
        print(f"  Allowed tools: {allowed_tools}")

    # Run prompts
    completed, errors, warnings = asyncio.run(
        run_prompts_sequential(
            prompts=prompts,
            stop_on_error=stop_on_error,
            stop_on_warning=args.stop_on_warning,
            allowed_tools=allowed_tools,
            error_keywords=error_keywords,
            warning_keywords=warning_keywords,
            working_dir=args.working_dir,
            max_turns=args.max_turns,
            verbose=args.verbose,
        )
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Completed: {completed}")
    print(f"  Errors: {errors}")
    print(f"  Warnings: {warnings}")

    # Exit code
    if errors > 0:
        sys.exit(1)
    elif warnings > 0 and args.stop_on_warning:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
