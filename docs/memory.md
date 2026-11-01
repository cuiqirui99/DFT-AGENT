# Memory

DFT Agent includes scientific guidance and checked calculation cases. Planning
and repair retrieve relevant entries automatically. Open **Memory** to search
them and inspect their conditions, inputs and evidence.

Cases describe previous calculations. Their settings are not defaults for your
material, and their results are not results of your new run. A verified failure
case records what failed; it does not claim a successful repair.

## Status

Only **verified** entries with intact evidence supply advice. **Candidate** and
**shadow_verified** entries need further review. **Rejected** and **deprecated**
entries remain visible but never supply valid recommendations. Importing a
record does not upgrade its status.

The bundled cases include portable inputs, result summaries and hashes of the
original solver output. Raw VASP output and POTCAR files are not included.
[Sources and migration record](memory-sources.md).

## Local records

In **Memory → Import local records**, enter a saved run folder, an old
`records.jsonl`, or the old platform's `platform.db`. Click **Preview import**,
select records, then **Import selected**. Nothing is imported during preview.

Selected records are stored under `.dft-agent-memory` in your **Run folder**.
Legacy records remain local and need review before they can supply advice.
Selected current-format runs are rechecked against their saved inputs and
outputs whenever they are used. Keep their original folders available.

The importer does not upload records or change the old database. Planning and
repair can share checked summaries of selected runs with your chosen model,
as they do for other runs in the Run folder. Cluster settings and raw imported
records are not model context.

For the [bundled CLI](install.md#command-line):

```bash
dft-agent memory list --query SOC
dft-agent memory list --status rejected
dft-agent memory show RECORD_ID
dft-agent memory import /path/to/platform.db --runs ./runs
dft-agent memory import /path/to/platform.db --ids RECORD_ID --runs ./runs
```

Omitting `--ids` previews the import. Add `--runs ./runs` to `memory list` or
`memory show` to include your local records, and to `plan` to use selected runs.
