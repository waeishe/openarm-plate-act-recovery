"""Isolated model invocation; extracted unchanged from the experiment runner."""
import json
from pathlib import Path
DISABLED_FEATURES = (
    "shell_tool",
    "memories",
    "hooks",
    "apps",
    "plugins",
    "browser_use",
    "computer_use",
    "in_app_browser",
    "multi_agent",
    "skill_search",
    "image_generation",
    "view_image",
)


def codex_args(directory, model, effort):
    args = [
        "codex",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--json",
        "--output-schema",
        str(directory / "schema.json"),
        "-o",
        str(directory / "response.json"),
        "-C",
        str(directory),
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        'web_search="disabled"',
        "--enable",
        "skip_host_skill_discovery",
    ]
    for feature in DISABLED_FEATURES:
        args.extend(["--disable", feature])
    # skip_host_skill_discovery alone does not suppress filesystem skill catalogs.
    skill_dirs = sorted(
        {
            str(p.parent)
            for root in (Path.home() / ".codex/skills", Path.home() / ".agents/skills")
            for p in root.rglob("SKILL.md")
        }
    )
    if skill_dirs:
        overrides = ",".join("{path=" + json.dumps(p) + ",enabled=false}" for p in skill_dirs)
        args.extend(["-c", "skills.config=[" + overrides + "]"])
    return args


def parse_events(events):
    """Reject tool use and incomplete accounting, rather than treating it as zero."""
    ids, usages = [], []
    allowed_items = {"agent_message", "reasoning", "error"}
    for event in events:
        if event["type"] == "thread.started":
            ids.append(event["thread_id"])
        if event["type"] == "turn.completed":
            usages.append(event["usage"])
        if event["type"].startswith("item.") and event.get("item", {}).get("type") not in allowed_items:
            raise ValueError("isolated_agent_attempted_tool_use")
    if len(ids) != 1 or len(usages) != 1:
        raise ValueError("missing_or_ambiguous_agent_usage")
    usage = usages[0]
    if any(type(usage.get(k)) is not int for k in ("input_tokens", "output_tokens")):
        raise ValueError("missing_agent_token_counts")
    return {
        "thread_id": ids[0],
        "usage": usage,
        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
    }

