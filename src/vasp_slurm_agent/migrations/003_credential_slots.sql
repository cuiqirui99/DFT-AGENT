-- Track attempted keychain writes even when the following database commit fails.
-- No credential contents are stored here.
CREATE TABLE credential_slots (
    connection_id TEXT NOT NULL REFERENCES connections(id),
    version INTEGER NOT NULL CHECK(version > 0),
    created_at TEXT NOT NULL,
    PRIMARY KEY(connection_id,version)
);
