"""Reports a biologist reads after a lane has run, built from the registry alone.

`collect` resolves and loads what a report shows and `imaging` turns volumes into
pictures; both are numpy only. Drawing lives in `sections` and assembly in
`pdf`, which need matplotlib — the `viz` extra — so nothing here imports them:
a command that never draws must not need a plotting library to start.

A report reads; it never runs an external program, never writes to the
registry, and never decides that something passed or failed. The verdicts on
its pages are the ones each step's own QC recorded.
"""
