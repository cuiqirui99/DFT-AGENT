-- ABINIT, CP2K and Elk steps and software environments.
-- SQLite cannot alter a CHECK constraint, so both tables are rebuilt with the
-- same columns and keys; the runner disables foreign keys for the rebuild and
-- verifies every reference before it commits.
CREATE TABLE steps_v5 (
    project_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    id TEXT NOT NULL,
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    engine TEXT NOT NULL CHECK(engine IN ('vasp', 'qe', 'abinit', 'cp2k', 'elk', 'wannier90', 'python')),
    status TEXT NOT NULL DEFAULT 'planned',
    spec TEXT NOT NULL,
    PRIMARY KEY(project_id, workflow_id, id),
    UNIQUE(project_id, workflow_id, position),
    FOREIGN KEY(project_id, workflow_id) REFERENCES workflows(project_id, id)
);
INSERT INTO steps_v5(project_id, workflow_id, id, position, name, engine, status, spec)
    SELECT project_id, workflow_id, id, position, name, engine, status, spec FROM steps;
DROP TABLE steps;
ALTER TABLE steps_v5 RENAME TO steps;

CREATE TABLE environments_v5 (
    id TEXT PRIMARY KEY,
    connection_id TEXT NOT NULL REFERENCES connections(id),
    name TEXT NOT NULL,
    engine TEXT NOT NULL CHECK(engine IN ('vasp','qe','abinit','cp2k','elk','wannier90','python')),
    config TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
INSERT INTO environments_v5(id, connection_id, name, engine, config, revision, created_at, updated_at)
    SELECT id, connection_id, name, engine, config, revision, created_at, updated_at FROM environments;
DROP TABLE environments;
ALTER TABLE environments_v5 RENAME TO environments;
