"""Phase 0: smoke test for Langfuse self-host.

Sends a single trace via the v4 SDK and prints the trace id + UI URL.
Usage:
    cd ~/works/langfuse-handson
    uv run --env-file=.env scripts/01_hello_smoke.py
"""

from langfuse import Langfuse

langfuse = Langfuse()

with langfuse.start_as_current_observation(
    name="hello-smoke", as_type="span"
) as obs:
    obs.update(
        input={"question": "Hello, world?"},
        output={"answer": "Self-host is alive."},
        metadata={"phase": "0", "purpose": "smoke-test"},
    )
    trace_id = langfuse.get_current_trace_id()

langfuse.flush()

print(f"trace_id={trace_id}")
print(f"UI: http://localhost:3000/project/handson-project/traces/{trace_id}")
