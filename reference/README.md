# Reference

The shared material the ported marketing skills read: design references and
anti-patterns, the brand-voice schema, and the voice guide. It is data, not
skills — several skills read the same files, so they have one home.

From `melgarafael/growthOS` (MIT); see `../THIRD_PARTY_NOTICES.md`. The skills
referenced these as `growthOS/<dir>/...`, a path that only existed inside that
repository — so a skill pointing there read nothing here. They point at
`reference/<dir>/` now.

## What is NOT here

Three skills need the growthOS APPLICATION rather than its data — a Remotion
project, and a `growthOS_shared` Python module. That is a marketing platform
with its own compose file, publisher and review server, and porting it is a
different job from porting skills. Those three say so at the top rather than
carrying an import that resolves to nothing.
