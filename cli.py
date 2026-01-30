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
    from claude_agent_sdk import (
        query,
        ClaudeAgentOptions,
        ClaudeSDKError,
        ProcessError,
        CLIConnectionError,
        CLIJSONDecodeError,
    )
except ImportError:
    print("Error: claude-agent-sdk not installed.")
    print("Install with: pip install claude-agent-sdk")
    sys.exit(1)


# Error types for distinguishing SDK crashes from tool errors
ERROR_TYPE_NONE = "none"
ERROR_TYPE_TOOL = "tool"  # A tool Claude used returned an error (e.g., test failure)
ERROR_TYPE_SDK = "sdk"    # SDK/process crash - always fatal


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
) -> tuple[bool, str, str]:
    """
    Run a single prompt and detect success/failure via SDK error signals.

    Returns:
        tuple of (success, error_type, result_text)
        - success: True if completed without errors
        - error_type: ERROR_TYPE_NONE, ERROR_TYPE_TOOL, or ERROR_TYPE_SDK
        - result_text: The result or error message
    """
    result_text = ""
    text_parts: list[str] = []  # Collect text content from messages
    stderr_output: list[str] = []  # Capture stderr for error details
    had_tool_error = False

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
                    had_tool_error = True
                    if verbose:
                        print(f"  [DEBUG] ResultMessage.is_error=True (tool error, non-fatal by default)")

            # Check for tool result errors (ToolResultBlock)
            elif hasattr(message, "is_error") and message.is_error:
                had_tool_error = True
                if verbose:
                    print(f"  [DEBUG] Tool error detected in {msg_type}")

    except (ProcessError, CLIConnectionError, CLIJSONDecodeError) as e:
        # SDK-specific errors
        if verbose:
            print(f"  [DEBUG] SDK error type: {type(e).__name__}")

        error_details = ""
        if isinstance(e, ProcessError):
            error_details = e.stderr if e.stderr else "\n".join(stderr_output)
            if not error_details:
                error_details = str(e)
            error_msg = f"Process error (exit code {e.exit_code}):\n{error_details}"
        elif isinstance(e, CLIJSONDecodeError):
            error_msg = f"JSON decode error on line: {e.line}\nOriginal error: {e.original_error}"
            if stderr_output:
                error_msg += f"\n\nStderr output:\n" + "\n".join(stderr_output)
        else:
            error_msg = str(e)
            if stderr_output:
                error_msg += f"\n\nStderr output:\n" + "\n".join(stderr_output)

        # If we already saw a tool error before this exception, treat as tool error
        if had_tool_error:
            if verbose:
                print(f"  [DEBUG] Exception after tool error - treating as tool error")
            combined_result = result_text if result_text else ""
            if combined_result:
                combined_result += f"\n\n"
            combined_result += f"(Exception during cleanup: {error_msg})"
            return (False, ERROR_TYPE_TOOL, combined_result)

        return (False, ERROR_TYPE_SDK, f"SDK Error ({type(e).__name__}): {error_msg}")

    except ClaudeSDKError as e:
        # Catch-all for other SDK errors
        if verbose:
            print(f"  [DEBUG] SDK error type: {type(e).__name__}")
        error_msg = str(e)
        if stderr_output:
            error_msg += f"\n\nStderr output:\n" + "\n".join(stderr_output)

        # If we already saw a tool error before this exception, treat as tool error
        if had_tool_error:
            if verbose:
                print(f"  [DEBUG] Exception after tool error - treating as tool error")
            combined_result = result_text if result_text else ""
            if combined_result:
                combined_result += f"\n\n"
            combined_result += f"(Exception during cleanup: {error_msg})"
            return (False, ERROR_TYPE_TOOL, combined_result)

        return (False, ERROR_TYPE_SDK, f"SDK Error ({type(e).__name__}): {error_msg}")

    except Exception as e:
        # Unknown exceptions
        if verbose:
            print(f"  [DEBUG] Unknown exception type: {type(e).__name__}")
        error_msg = str(e)
        if stderr_output:
            error_msg += f"\n\nStderr output:\n" + "\n".join(stderr_output)

        # If we already saw a tool error before this exception, treat as tool error
        # This handles the case where SDK throws after ResultMessage.is_error=True
        if had_tool_error:
            if verbose:
                print(f"  [DEBUG] Exception after tool error - treating as tool error")
            combined_result = result_text if result_text else ""
            if combined_result:
                combined_result += f"\n\n"
            combined_result += f"(Exception during cleanup: {error_msg})"
            return (False, ERROR_TYPE_TOOL, combined_result)

        return (False, ERROR_TYPE_SDK, f"Exception ({type(e).__name__}): {error_msg}")

    # Use result_text if available, otherwise combine captured text parts
    if not result_text and text_parts:
        result_text = "\n".join(text_parts)

    if had_tool_error:
        # Include stderr output for tool errors if available
        if stderr_output:
            result_text += f"\n\nStderr output:\n" + "\n".join(stderr_output)
        return (False, ERROR_TYPE_TOOL, result_text)

    return (True, ERROR_TYPE_NONE, result_text)


async def run_prompts_sequential(
    prompts: list[tuple[str, str]],
    stop_on_sdk_error: bool = True,
    stop_on_tool_error: bool = False,
    allowed_tools: Optional[list[str]] = None,
    working_dir: Optional[str] = None,
    max_turns: Optional[int] = None,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """
    Run multiple prompts sequentially.

    Args:
        prompts: list of (name, prompt_content) tuples
        stop_on_sdk_error: Stop on SDK/process errors (default: True)
        stop_on_tool_error: Stop on tool errors like test failures (default: False)

    Returns:
        tuple of (completed_count, tool_error_count, sdk_error_count)
    """
    completed = 0
    tool_errors = 0
    sdk_errors = 0

    for i, (name, prompt) in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Running: {name}")
        print(f"  {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print("-" * 60)

        success, error_type, result = await run_prompt(
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

        if error_type == ERROR_TYPE_SDK:
            sdk_errors += 1
            print(f"\n[SDK ERROR] {name} failed due to SDK/process error.")
            if stop_on_sdk_error:
                print("Stopping due to SDK error (always fatal).")
                break
        elif error_type == ERROR_TYPE_TOOL:
            tool_errors += 1
            print(f"\n[TOOL ERROR] {name} completed but a tool reported an error (e.g., test failure).")
            if stop_on_tool_error:
                print("Stopping due to tool error (--stop-on-tool-error is enabled).")
                break
            else:
                # Tool errors are non-fatal by default, count as completed
                completed += 1
        else:
            completed += 1
            print(f"\n[OK] {name} completed successfully.")

    return (completed, tool_errors, sdk_errors)


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

  # Stop if a tool (like cargo test) returns an error
  python cli.py --dir ./prompts/ --stop-on-tool-error

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

Error handling:
  - SDK errors (connection failures, process crashes): Always stop by default
  - Tool errors (test failures, command errors): Continue by default
  Use --stop-on-tool-error to also stop on tool errors.
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
        "--stop-on-tool-error",
        action="store_true",
        default=False,
        help="Stop execution when a tool returns an error (e.g., test failure). Default: continue",
    )

    parser.add_argument(
        "--no-stop-on-sdk-error",
        action="store_true",
        help="Continue execution even after SDK/process errors (not recommended)",
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
    stop_on_sdk_error = not args.no_stop_on_sdk_error
    stop_on_tool_error = args.stop_on_tool_error
    if args.no_tools:
        allowed_tools = []
    elif args.tools:
        allowed_tools = args.tools.split(",")
    else:
        allowed_tools = None  # Will use DEFAULT_ALLOWED_TOOLS

    print(f"Running {len(prompts)} prompt(s) sequentially...")
    print(f"  Stop on SDK error: {stop_on_sdk_error}")
    print(f"  Stop on tool error: {stop_on_tool_error}")
    tools_display = allowed_tools if allowed_tools is not None else DEFAULT_ALLOWED_TOOLS
    print(f"  Allowed tools: {tools_display}")

    # Run prompts
    completed, tool_errors, sdk_errors = asyncio.run(
        run_prompts_sequential(
            prompts=prompts,
            stop_on_sdk_error=stop_on_sdk_error,
            stop_on_tool_error=stop_on_tool_error,
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
    print(f"  Tool errors: {tool_errors}")
    print(f"  SDK errors: {sdk_errors}")

    # Exit code: 1 for SDK errors, 2 for tool errors only, 0 for success
    if sdk_errors > 0:
        sys.exit(1)
    elif tool_errors > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
