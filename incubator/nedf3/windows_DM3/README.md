# DM3 Upload Support

`windows_DM3` now keeps only the shared DM3 API / OSS upload helpers. The normal
operator workflow is unified in the main Vive/DM3 collection window:

```bat
Vive_DM3\run_vive_dm3.bat
```

An upload-only console remains available for troubleshooting or re-upload work:

```bat
Vive_DM3\run_upload_console.bat
```

The old launcher is still available for compatibility and opens the main window:

```bat
windows_DM3\run_dm3_gui.bat
```

It opens the same unified collection, upload, and performance monitor window.

## Unified Upload Behavior

- Uploads finalized `episode_*` folders only; `_temp` folders are skipped.
- Automatic upload applies the default safety policy: older than 30 minutes and not
  within the latest 10 episodes.
- `确认全部上传` bypasses that safety policy for end-of-day upload.
- Failed uploads can be retried from the same queue.
- Per-episode state is stored in `auto_upload_state.json` under the selected data
  directory.
- OSS upload still uses the existing STS token, resumable multipart upload, skip
  checks for already-uploaded objects, and transient error retry support from
  `windows_DM3/oss_uploader.py`.

## Episode Discovery And Maintenance

The upload manager maintains an in-memory episode cache so saving one episode does
not require recursively reading every historical episode again.

1. When the upload manager starts, it initializes the cache with one background full
   scan. If an episode finishes before initialization completes, that maintenance
   request performs the one-time full initialization itself (`mode=full_init`).
2. After a finalized episode is saved, the saver sends its exact `output_dir` and
   `episode_name` to the upload panel. The manager validates that the name is a plain
   finalized `episode_*` name and that the resolved path is inside a managed data
   directory.
3. Once initialized, maintenance scans only that episode directory and updates the
   cached entry (`mode=incremental`). Upload eligibility is then evaluated from the
   cache while preserving the existing age and latest-episode protection policy.
4. Changing the configured data directories invalidates the cache and requires one
   new full initialization. Explicit operations such as force upload and failed-job
   retry may also request a full scan because they must reconcile operator-selected
   paths with current disk contents.

The maintenance result reports separate timings:

- `scan`: filesystem discovery and inspection time.
- `enqueue`: upload-policy evaluation, upload-state persistence, and queue insertion.
- `elapsed`: total maintenance time, including both phases and coordination overhead.

A normal post-save log after initialization should contain `mode=incremental`.
`episodes=<count>` is the number of cached episodes considered by policy; it does not
mean all of them were recursively scanned during that maintenance request.

## Upload State And Safety

- Full scans remove stale `auto_upload_state.json` entries for episodes that no
  longer exist locally, preventing the state file from growing indefinitely after
  local cleanup.
- Protection-state updates are batched into one state-file write. Unchanged protected
  entries are not rewritten, avoiding repeated rewrites of a large state file.
- Full and incremental scans share a scan lock so an older full-scan result cannot
  overwrite a concurrently added episode.
- Before queue insertion, an episode is checked again for existence and NEDF
  completeness. If an operator deletes it after a scan, it is removed from cache and
  is not queued.
- A deleted cached episode does not consume one of the `latest 10` protection slots;
  protection is calculated from complete episodes that still exist.
- The incremental path does not delete successful recordings. Existing upload-worker
  behavior, including the behavior of a deletion performed while an upload is already
  running, is unchanged by this optimization.

Neither discovery nor duration accounting modifies an episode's `metadata.json`.
The OSS and HTTP upload implementations are also unchanged; this work changes only
episode discovery, queue maintenance, and upload-state bookkeeping around them.

## Daily Duration Accounting

The collection GUI displays two totals across all task directories under the selected
daily data root:

- **Total collection time** includes accepted episodes and anomaly-error episodes.
- **Effective data time** includes accepted episodes only.

Duration is calculated from the first and last recorded data timestamps. Outcomes are
appended outside episode directories to `.episode_time_ledger.jsonl`, so an anomaly
episode can still contribute to total collection time after its data is discarded.
On GUI refresh, finalized legacy episodes missing from the ledger are backfilled once;
this prevents later local deletion from reducing an already counted total. Legacy
episodes deleted before backfill cannot be reconstructed.

The statistics refresh runs off the Tk UI thread at GUI startup, after an output
directory change, after task selection, and after episode completion. It does not run
inside the acquisition loop and does not change the collection process frequency.

## Focused Verification

Run the upload-maintenance and duration-regression tests with the repository virtual
environment:

```powershell
.venv\Scripts\python.exe -m unittest tests.Vive_DM3.test_upload_manager_incremental -v
```

The focused suite covers initial and incremental scans, directory changes, concurrent
scan serialization, deletion before enqueue, stale-state pruning, batched state
writes, duration backfill, duplicate prevention, and preservation of `metadata.json`.
