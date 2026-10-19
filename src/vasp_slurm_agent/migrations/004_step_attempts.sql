CREATE TABLE step_attempts (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK(attempt > 0),
    state TEXT NOT NULL,
    conversation_path TEXT,
    run_id TEXT,
    environment_id TEXT REFERENCES environments(id),
    source_context TEXT NOT NULL DEFAULT '{}',
    approved_digest TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id,workflow_id,step_id,attempt),
    FOREIGN KEY(project_id,workflow_id,step_id) REFERENCES steps(project_id,workflow_id,id),
    FOREIGN KEY(project_id,run_id) REFERENCES runs(project_id,id)
);
