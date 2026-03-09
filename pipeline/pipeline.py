"""
This pipeline is meant to automate the data inference pipeline predicting 'Locomotion' 
from the sensor data. It is designed to be as generalizable to new datasets as possible.
"""

"""
Two-Agent Inference Pipeline using the Anthropic SDK
=====================================================
Agent 1: Reads user_request.txt + documentation.html → writes query_specs.txt
Agent 2: Reads query_specs.txt + ADL_no_label.parquet  → writes results file

Usage:
    pip install anthropic pandas pyarrow beautifulsoup4
    export ANTHROPIC_API_KEY="your-key-here"
    python inference_pipeline.py --input-dir /path/to/your/folder
"""

import os
import json
import argparse
import textwrap
from pathlib import Path

import anthropic
import pandas as pd
from bs4 import BeautifulSoup

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL = "claude-opus-4-5" # Why this model specifically? Seems like the largest of the bunch 
MAX_TOKENS = 4096 # What does this control? 
MAX_HTML_CHARS = 40_000   # truncate very large HTML files before sending - do we even want to do this? How long is our HTML file? 


# ── Helpers ────────────────────────────────────────────────────────────────────

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def html_to_text(path: Path) -> str:
    """Strip HTML tags; truncate if very long."""
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    if len(text) > MAX_HTML_CHARS:
        text = text[:MAX_HTML_CHARS] + "\n\n[...documentation truncated for brevity...]"
    return text


def parquet_summary(path: Path, max_rows: int = 5) -> tuple[pd.DataFrame, str]:
    """Return the full DataFrame and a compact text summary for the LLM."""
    df = pd.read_parquet(path)
    lines = [
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"Columns: {list(df.columns)}",
        f"Dtypes:\n{df.dtypes.to_string()}",
        f"\nFirst {max_rows} rows (JSON):\n{df.head(max_rows).to_json(orient='records', indent=2)}",
    ]
    null_counts = df.isnull().sum()
    if null_counts.any():
        lines.append(f"\nNull counts:\n{null_counts[null_counts > 0].to_string()}")
    return df, "\n".join(lines)


def call_claude(client: anthropic.Anthropic, system: str, user: str) -> str:
    """Single-turn Claude call; returns the assistant text."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip()


# ── Agent 1: Query Specification Writer ───────────────────────────────────────

def agent1_create_query_specs(
    client: anthropic.Anthropic,
    user_request_path: Path,
    documentation_path: Path,
    output_path: Path,
) -> str:
    """
    Reads the user request and documentation, then produces a structured
    query_specs file that gives Agent 2 precise instructions.
    """
    print("\n━━━ Agent 1 | Creating query specs ━━━")

    user_request = read_text(user_request_path)
    documentation = html_to_text(documentation_path)

    print(f"  • User request ({len(user_request)} chars): {user_request_path.name}")
    print(f"  • Documentation ({len(documentation)} chars): {documentation_path.name}")

    system_prompt = textwrap.dedent("""\
        You are a meticulous data-analysis architect.
        Your job is to read a user's natural-language data request together with
        technical documentation and produce a self-contained, machine-readable
        specification (query_specs) that a downstream data-processing agent can
        follow WITHOUT having to re-read the original documentation.

        The specification MUST be valid JSON and include:
        {
          "objective":       "<one-sentence summary of what the analysis must achieve>",
          "data_context":    "<key facts from the documentation relevant to this dataset>",
          "operations": [
            {
              "step": 1,
              "action": "<filter | aggregate | transform | sort | select | label | compute>",
              "description": "<precise, actionable instruction referencing column names, values, thresholds>",
              "output_column": "<name of new/modified column if applicable, else null>"
            }
          ],
          "output_format":   "<csv | json | parquet>",
          "output_filename": "<suggested filename without path>",
          "notes":           "<any caveats, assumptions, or edge-cases Agent 2 should know>"
        }

        Respond with ONLY the JSON object — no markdown fences, no prose.
    """)

    user_message = textwrap.dedent(f"""\
        ## User Request
        {user_request}

        ## Technical Documentation
        {documentation}

        Produce the query_specs JSON now.
    """)

    print("  • Calling Claude (Agent 1)…")
    raw = call_claude(client, system_prompt, user_message)

    # Validate JSON
    try:
        specs = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Attempt to extract JSON block if the model added any wrapper text
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            specs = json.loads(match.group())
            raw = match.group()
        else:
            raise ValueError(f"Agent 1 returned invalid JSON: {exc}\n\nRaw output:\n{raw}")

    output_path.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    print(f"  ✓ query_specs written → {output_path}")
    return raw


# ── Agent 2: Data Processing Agent ────────────────────────────────────────────

def agent2_apply_query_specs(
    client: anthropic.Anthropic,
    query_specs_path: Path,
    parquet_path: Path,
    output_dir: Path,
) -> Path:
    """
    Reads query_specs + the parquet file, asks Claude to produce
    Python pandas code, executes it, and saves the result.
    """
    print("\n━━━ Agent 2 | Applying query specs to data ━━━")

    specs_text = read_text(query_specs_path)
    specs = json.loads(specs_text)
    df, data_summary = parquet_summary(parquet_path)

    print(f"  • query_specs: {query_specs_path.name}")
    print(f"  • Parquet: {parquet_path.name}  {df.shape}")

    system_prompt = textwrap.dedent("""\
        You are an expert Python / pandas data engineer.
        You will receive:
          1. A query_specs JSON describing the exact operations to perform.
          2. A summary of the target DataFrame (schema + sample rows).

        Your task: write a single self-contained Python function called
        `process(df: pd.DataFrame) -> pd.DataFrame` that implements every
        step in the query_specs["operations"] list and returns the result.

        Rules:
        - Import only from the Python standard library or pandas/numpy.
        - Do NOT read or write any files inside the function.
        - Handle missing values gracefully.
        - Add a short comment above each logical block referencing the step number.
        - Respond with ONLY a Python code block (no explanation, no markdown fences).
    """)

    user_message = textwrap.dedent(f"""\
        ## query_specs
        {specs_text}

        ## DataFrame Summary
        {data_summary}

        Write the `process(df)` function now.
    """)

    print("  • Calling Claude (Agent 2) for processing code…")
    code = call_claude(client, system_prompt, user_message)

    # Strip markdown fences if present
    import re
    code = re.sub(r"^```[^\n]*\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"```$", "", code, flags=re.MULTILINE).strip()

    print("  • Executing generated code…")
    exec_globals: dict = {}
    try:
        import pandas as pd  # noqa: F401 – ensure it's in scope for exec
        import numpy as np   # noqa: F401
        exec(f"import pandas as pd\nimport numpy as np\n\n{code}", exec_globals)  # noqa: S102
        process_fn = exec_globals["process"]
        result_df = process_fn(df.copy())
    except Exception as exc:
        raise RuntimeError(
            f"Agent 2's generated code raised an error: {exc}\n\nCode:\n{code}"
        ) from exc

    print(f"  ✓ Processing complete → result shape: {result_df.shape}")

    # Determine output format and filename from specs
    fmt = specs.get("output_format", "csv").lower()
    suggested_name = specs.get("output_filename", f"results.{fmt}")
    output_path = output_dir / suggested_name

    if fmt == "csv":
        result_df.to_csv(output_path, index=False)
    elif fmt == "json":
        result_df.to_json(output_path, orient="records", indent=2)
    elif fmt == "parquet":
        result_df.to_parquet(output_path, index=False)
    else:
        # Fallback to CSV
        output_path = output_path.with_suffix(".csv")
        result_df.to_csv(output_path, index=False)

    print(f"  ✓ Results written → {output_path}")
    return output_path


# ── Pipeline Orchestrator ──────────────────────────────────────────────────────

def run_pipeline(input_dir: str, output_dir: str | None = None) -> None:
    input_path = Path(input_dir).resolve()
    out_path = Path(output_dir).resolve() if output_dir else input_path
    out_path.mkdir(parents=True, exist_ok=True)

    # Locate required files
    user_request_file = input_path / "user_request.txt"
    documentation_file = input_path / "documentation.html"
    parquet_file = input_path / "ADL_no_label.parquet"

    for f in [user_request_file, documentation_file, parquet_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    query_specs_file = out_path / "query_specs.json"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    print("=" * 60)
    print("  Inference Pipeline — Two-Agent Mode")
    print("=" * 60)

    # ── Agent 1 ──
    agent1_create_query_specs(
        client=client,
        user_request_path=user_request_file,
        documentation_path=documentation_file,
        output_path=query_specs_file,
    )

    # ── Agent 2 ──
    result_file = agent2_apply_query_specs(
        client=client,
        query_specs_path=query_specs_file,
        parquet_path=parquet_file,
        output_dir=out_path,
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  • query_specs : {query_specs_file}")
    print(f"  • results     : {result_file}")
    print("=" * 60)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-agent Anthropic SDK inference pipeline."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing user_request.txt, documentation.html, ADL_no_label.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write query_specs.json and results (default: same as --input-dir)",
    )
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_dir)