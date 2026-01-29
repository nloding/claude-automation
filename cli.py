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

from natsort import natsorted

try:
    from claude_agent_sdk import query, ClaudeAgentOptions, ProcessError
except ImportError:
    print("Error: claude-agent-sdk not installed.")
    print("Install with: pip install claude-agent-sdk")
    sys.exit(1)


# Default allowed tools for common development workflows
DEFAULT_ALLOWED_TOOLS = [
    "Read",      # Read file contents
    "Write",     # Write/create files
    "Edit",      # Edit existing files
    "Bash",      # Run shell commands (cargo build, cargo test, npm, etc.)
    "Glob",      # Find files by pattern
    "Grep",      # Search file contents
]


async def run_prompt(
    prompt: str,
    allowed_tools: Optional[list[str]] = None,
    working_dir: Optional[str] = None,
    max_turns: Optional[int] = None,
    verbose: bool = False,
) -> tuple[bool, str]:
    """
    Run a single prompt and detect success/failure via SDK error signals.

    Returns:
        tuple of (success, result_text)
    """
    result_text = ""
    text_parts: list[str] = []  # Collect text content from messages
    stderr_output: list[str] = []  # Capture stderr for error details
    had_error = False

    # Use default tools if none specified
    tools_to_use = allowed_tools if allowed_tools is not None else DEFAULT_ALLOWED_TOOLS

    # Capture stderr output via callback
    def capture_stderr(line: str) -> None:
        stderr_output.append(line)
        if verbose:
            print(f"  [STDERR] {line}")

    options_kwargs = {
        "allowed_tools": tools_to_use,
        "stderr": capture_stderr,
    }
    if max_turns:
        options_kwargs["max_turns"] = max_turns
    if working_dir:
        options_kwargs["cwd"] = working_dir

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(**options_kwargs),
        ):
            msg_type = type(message).__name__
            if verbose:
                print(f"  [DEBUG] Message type: {msg_type}")

            # Capture text content from assistant messages
            if hasattr(message, "content") and isinstance(message.content, str):
                text_parts.append(message.content)
                if verbose:
                    print(f"  [DEBUG] Captured text: {message.content[:100]}...")

            # Check for ResultMessage (final message with is_error status)
            if msg_type == "ResultMessage":
                if hasattr(message, "result") and message.result:
                    result_text = message.result
                if hasattr(message, "is_error") and message.is_error:
                    had_error = True
                    if verbose:
                        print(f"  [DEBUG] ResultMessage.is_error=True")

            # Check for tool result errors (ToolResultBlock)
            elif hasattr(message, "is_error") and message.is_error:
                had_error = True
                if verbose:
                    print(f"  [DEBUG] Tool error detected in {msg_type}")

    except ProcessError as e:
        # Extract detailed error info from ProcessError
        error_details = e.stderr if e.stderr else "\n".join(stderr_output)
        if not error_details:
            error_details = str(e)
        return (False, f"Process error (exit code {e.exit_code}):\n{error_details}")

    except Exception as e:
        # Include any captured stderr in generic exceptions
        error_msg = str(e)
        if stderr_output:
            error_msg += f"\n\nStderr output:\n" + "\n".join(stderr_output)
        return (False, f"Exception: {error_msg}")

    # Use result_text if available, otherwise combine captured text parts
    if not result_text and text_parts:
        result_text = "\n".join(text_parts)

    return (not had_error, result_text)


async def run_prompts_sequential(
    prompts: list[tuple[str, str]],
    stop_on_error: bool = True,
    allowed_tools: Optional[list[str]] = None,
    working_dir: Optional[str] = None,
    max_turns: Optional[int] = None,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Run multiple prompts sequentially.

    Args:
        prompts: list of (name, prompt_content) tuples

    Returns:
        tuple of (completed_count, error_count)
    """
    completed = 0
    errors = 0

    for i, (name, prompt) in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Running: {name}")
        print(f"  {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print("-" * 60)

        success, result = await run_prompt(
            prompt=prompt,
            allowed_tools=allowed_tools,
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
        else:
            completed += 1
            print(f"\n[OK] {name} completed successfully.")

    return (completed, errors)


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

    # Get all files in directory, sorted naturally (so task-9 comes before task-10)
    files = natsorted([f for f in dirpath.iterdir() if f.is_file()])

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

  # Continue even after errors
  python cli.py --dir ./prompts/ --no-stop-on-error

  # Override default tools with specific set
  python cli.py --dir ./prompts/ --tools "Read,Edit,Bash"

  # Disable all tools (text-only mode)
  python cli.py "Explain this concept" --no-tools

Default allowed tools: Read, Write, Edit, Bash, Glob, Grep
  - Read:  Read file contents
  - Write: Write/create files
  - Edit:  Edit existing files
  - Bash:  Run shell commands (cargo build, cargo test, npm, etc.)
  - Glob:  Find files by pattern
  - Grep:  Search file contents

Error detection uses SDK structured signals (ResultMessage.is_error,
ToolResultBlock.is_error) rather than keyword matching for reliability.
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
        "--tools",
        type=str,
        help=f"Comma-separated list of allowed tools (default: {','.join(DEFAULT_ALLOWED_TOOLS)})",
    )

    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable all tools (text-only mode)",
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
    if args.no_tools:
        allowed_tools = []
    elif args.tools:
        allowed_tools = args.tools.split(",")
    else:
        allowed_tools = None  # Will use DEFAULT_ALLOWED_TOOLS

    print(f"Running {len(prompts)} prompt(s) sequentially...")
    print(f"  Stop on error: {stop_on_error}")
    tools_display = allowed_tools if allowed_tools is not None else DEFAULT_ALLOWED_TOOLS
    print(f"  Allowed tools: {tools_display}")

    # Run prompts
    completed, errors = asyncio.run(
        run_prompts_sequential(
            prompts=prompts,
            stop_on_error=stop_on_error,
            allowed_tools=allowed_tools,
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

    # Exit code
    if errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
