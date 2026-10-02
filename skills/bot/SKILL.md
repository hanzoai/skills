---
name: skills-hub
description: Use the Hanzo Skills Hub CLI to search, install, update, and publish agent skills from the Hanzo skills hub (hub.hanzo.bot, API at api.hanzo.ai/v1/skills). Use when you need to fetch new skills on the fly, sync installed skills to latest or a specific version, or publish new/updated skill folders with the npm-installed skills-hub CLI.
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["skills-hub"] },
        "install":
          [
            {
              "id": "node",
              "kind": "node",
              "package": "skills-hub",
              "bins": ["skills-hub"],
              "label": "Install Hanzo Skills Hub CLI (npm)",
            },
          ],
      },
  }
---

# Hanzo Skills Hub CLI

Install

```bash
npm i -g skills-hub
```

Auth (publish)

```bash
skills-hub login
skills-hub whoami
```

Search

```bash
skills-hub search "postgres backups"
```

Install

```bash
skills-hub install my-skill
skills-hub install my-skill --version 1.2.3
```

Update (hash-based match + upgrade)

```bash
skills-hub update my-skill
skills-hub update my-skill --version 1.2.3
skills-hub update --all
skills-hub update my-skill --force
skills-hub update --all --no-input --force
```

List

```bash
skills-hub list
```

Publish

```bash
skills-hub publish ./my-skill --slug my-skill --name "My Skill" --version 1.2.0 --changelog "Fixes + docs"
```

Notes

- Default registry: https://api.hanzo.ai, every route under /v1/skills (override with SKILLS_HUB_REGISTRY or --registry)
- Default workdir: cwd (falls back to Hanzo Bot workspace); install dir: ./skills (override with --workdir / --dir / SKILLS_HUB_WORKDIR)
- Update command hashes local files, resolves matching version, and upgrades to latest unless --version is set
