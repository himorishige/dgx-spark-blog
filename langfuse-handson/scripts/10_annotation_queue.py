"""Phase 4: build a Human Annotation Queue, fill it from recent traces.

Phoenix's Annotations are span-level scores you POST individually. Langfuse
wraps the same data model into a queueing workflow: create a queue, attach
score configs, then push observations into it for reviewers to grade in the UI.

Usage:
    cd ~/works/langfuse-handson
    uv run --env-file=.env scripts/10_annotation_queue.py
"""

from langfuse import Langfuse

QUEUE_NAME = "handson-review-queue"

langfuse = Langfuse()

# 1. Create a numeric score config that the queue will collect.
helpful_config = langfuse.api.score_configs.create(
    name="reviewer-helpful",
    data_type="NUMERIC",
    min_value=0.0,
    max_value=1.0,
    description="Reviewer-graded helpfulness, 0.0 (not helpful) to 1.0 (very helpful).",
)
print(f"score config={helpful_config.name} id={helpful_config.id}")

# 2. Create the queue.
queue = langfuse.api.annotation_queues.create_queue(
    name=QUEUE_NAME,
    description="Manual review queue for the langfuse handson article",
    score_config_ids=[helpful_config.id],
)
print(f"queue={queue.name} id={queue.id}")

# 3. Pull the most recent qa-handson traces from session 1.1 and enqueue them.
traces = langfuse.api.trace.list(name="qa-handson", limit=3).data
for trace in traces:
    item = langfuse.api.annotation_queues.create_queue_item(
        queue_id=queue.id,
        object_id=trace.id,
        object_type="TRACE",
    )
    print(f"  enqueued trace_id={trace.id[:14]}.. queue_item={item.id[:14]}..")

print(
    f"UI: http://localhost:3000/project/handson-project/annotation-queues/{queue.id}"
)
