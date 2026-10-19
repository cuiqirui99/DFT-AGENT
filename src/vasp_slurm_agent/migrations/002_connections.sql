CREATE TABLE connections (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('model','hpc')),
    name TEXT NOT NULL,
    settings TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    secret_version INTEGER NOT NULL DEFAULT 0,
    auth_status TEXT NOT NULL DEFAULT 'unknown',
    connection_status TEXT NOT NULL DEFAULT 'untested',
    probe TEXT,
    last_checked TEXT,
    revoked_at TEXT,
    deleted_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE environments (
    id TEXT PRIMARY KEY,
    connection_id TEXT NOT NULL REFERENCES connections(id),
    name TEXT NOT NULL,
    engine TEXT NOT NULL CHECK(engine IN ('vasp','qe','wannier90','python')),
    config TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE project_connections (
    project_id TEXT NOT NULL REFERENCES projects(id),
    connection_id TEXT NOT NULL REFERENCES connections(id),
    role TEXT NOT NULL CHECK(role IN ('model','hpc')),
    created_at TEXT NOT NULL,
    PRIMARY KEY(project_id,connection_id,role)
);

CREATE TABLE connection_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    connection_id TEXT NOT NULL REFERENCES connections(id),
    kind TEXT NOT NULL,
    details TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TRIGGER connection_events_no_update BEFORE UPDATE ON connection_events BEGIN
    SELECT RAISE(ABORT, 'Connection events are append-only');
END;
CREATE TRIGGER connection_events_no_delete BEFORE DELETE ON connection_events BEGIN
    SELECT RAISE(ABORT, 'Connection events are append-only');
END;
