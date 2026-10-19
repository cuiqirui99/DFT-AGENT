CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    opened_at TEXT,
    archived_at TEXT,
    deleted_at TEXT,
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0)
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    title TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, id)
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(project_id, session_id) REFERENCES sessions(project_id, id)
);
CREATE INDEX messages_session ON messages(project_id, session_id, id);

CREATE TABLE workflows (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    goal TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planned',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, id),
    FOREIGN KEY(project_id, session_id) REFERENCES sessions(project_id, id)
);

CREATE TABLE steps (
    project_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    id TEXT NOT NULL,
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    engine TEXT NOT NULL CHECK(engine IN ('vasp', 'qe', 'wannier90', 'python')),
    status TEXT NOT NULL DEFAULT 'planned',
    spec TEXT NOT NULL,
    PRIMARY KEY(project_id, workflow_id, id),
    UNIQUE(project_id, workflow_id, position),
    FOREIGN KEY(project_id, workflow_id) REFERENCES workflows(project_id, id)
);

CREATE TABLE dependencies (
    project_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    depends_on TEXT NOT NULL,
    PRIMARY KEY(project_id, workflow_id, step_id, depends_on),
    CHECK(step_id != depends_on),
    FOREIGN KEY(project_id, workflow_id, step_id) REFERENCES steps(project_id, workflow_id, id),
    FOREIGN KEY(project_id, workflow_id, depends_on) REFERENCES steps(project_id, workflow_id, id)
);

CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    run_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('run', 'batch')),
    relative_path TEXT NOT NULL,
    status TEXT NOT NULL,
    state TEXT NOT NULL,
    workflow_id TEXT,
    step_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, id),
    UNIQUE(project_id, run_id),
    UNIQUE(project_id, relative_path),
    CHECK(step_id IS NULL OR workflow_id IS NOT NULL),
    FOREIGN KEY(project_id, workflow_id) REFERENCES workflows(project_id, id),
    FOREIGN KEY(project_id, workflow_id, step_id) REFERENCES steps(project_id, workflow_id, id)
);

CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    relative_path TEXT NOT NULL,
    kind TEXT NOT NULL,
    run_id TEXT,
    size INTEGER NOT NULL CHECK(size >= 0),
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, id),
    UNIQUE(project_id, relative_path),
    FOREIGN KEY(project_id, run_id) REFERENCES runs(project_id, id)
);

CREATE TABLE memory (
    project_id TEXT NOT NULL REFERENCES projects(id),
    key TEXT NOT NULL,
    text TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(project_id, key)
);

CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id),
    kind TEXT NOT NULL,
    details TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX events_project ON events(project_id, id);
CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN
    SELECT RAISE(ABORT, 'Workspace events are append-only');
END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN
    SELECT RAISE(ABORT, 'Workspace events are append-only');
END;
