# Fresh FA Backups

Daily backups can be created with:

```bash
cd /home/ab/Fresh-FA
.venv/bin/python scripts/backup.py
```

By default archives are written to:

```text
/home/ab/Fresh-FA/backups/
```

Each archive is named `fresh-fa-YYYYMMDD-HHMMSS.tar.gz` and contains:

- `fresh_fa.db`, copied with the SQLite online backup API
- `.env`
- `02.Transcript/`
- `03.Evaluation/`
- `04.Meta/`
- `config/`

The archive does not include `01.Video/` media files.

Retention defaults to 30 days. Override it with:

```bash
.venv/bin/python scripts/backup.py --keep-days 14
```

To store backups elsewhere:

```bash
.venv/bin/python scripts/backup.py --backup-dir /mnt/backups/fresh-fa
```

Example cron entry for a daily backup at 03:15:

```cron
15 3 * * * cd /home/ab/Fresh-FA && mkdir -p /home/ab/Fresh-FA/backups && .venv/bin/python scripts/backup.py >> /home/ab/Fresh-FA/backups/backup.log 2>&1
```

Restore outline:

1. Stop the Fresh FA server.
2. Extract the archive into `/home/ab/Fresh-FA`.
3. Start the Fresh FA server.

The `.env` file is included so recovery is simple. Treat backup archives as secrets.
