"""Locked holdout run (PRD section 10.5). Owner: M4.

STUB. Not implemented. When implemented it must:
  - write protocol/holdout.lock (git commit, config hash, time),
  - refuse to run a second time unless --force is given,
  - log a forced run in pipeline_events.
"""


def main() -> None:
    raise NotImplementedError("run_holdout.py is a stub; see PRD section 10.5")


if __name__ == "__main__":
    main()
