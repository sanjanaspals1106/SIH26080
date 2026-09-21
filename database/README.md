# database/

PostgreSQL schema for district-level data, metadata, results, events and lookups
(PRD section 20.2). Cell-level data is **not** here; it stays in Parquet (PRD D16).

**Local PostgreSQL 14 or newer only. No Docker. No PostGIS** (PRD D15). District outlines are
stored as GeoJSON text (JSONB).

## Create the database and user

Install PostgreSQL locally first (macOS: Homebrew or Postgres.app; Linux: package manager),
then in `psql` as a superuser (on macOS usually `psql postgres`; on Linux `sudo -u postgres psql`):

```sql
CREATE USER sih WITH PASSWORD 'CHANGE_ME';
CREATE DATABASE sih_rain OWNER sih;
```

Use the same password in `DATABASE_URL` in your `.env`
(`postgresql+psycopg://sih:CHANGE_ME@localhost:5432/sih_rain`). Do not commit `.env`.

## Apply the schema

From the repository root:

```bash
psql -U sih -h localhost -d sih_rain -f database/schema.sql
```

`schema.sql` is not idempotent (plain `CREATE TABLE`). To start again, drop and recreate the
database, then apply it again.

## Notes

- `schema.sql` is copied verbatim from PRD section 20.2. The four `CREATE INDEX` lines at the end
  come from the index list printed after the DDL in the PRD; the PRD gives columns only, so the
  index names are ours.
- Changes to the schema are contract changes. Tell everyone who reads or writes the affected tables
  (see `CONTRIBUTING.md`).
