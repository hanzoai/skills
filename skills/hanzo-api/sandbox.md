---
name: hanzo-api-sandbox
description: The ONE compute primitive: a sandbox is a gVisor pod that runs somebody else's code, and every lifetime is the same object.
---

# Hanzo Sandbox

**Scope**: The ONE compute primitive: a sandbox is a gVisor pod that runs somebody else's code, and every lifetime is the same…
**Surface**: 19 operations on `/v1/sandboxes`
**Lines**: ~157
**Last Updated**: 2026-08-20
**Source**: https://api.hanzo.ai/v1/openapi.json (`x-app: sandboxes`) — generated, do not hand-edit

## When to Use This Skill

Activate this skill when:
- Running code you did not write — a model wrote it, or a user submitted it
- You need a filesystem, a terminal or a screen that outlives a single call
- Isolating a long-running job from the caller's process and from other tenants
- Driving a desktop: the screen and terminal websockets are on the same object
- Calling `/v1/sandboxes` (19 operations)

## Authentication

Every operation takes a Hanzo IAM bearer token. `hanzo auth token` prints the
active short-lived access token for the identity you are signed in as; the org
is carried in the token's claim, never in a path or a query parameter.

```bash
TOKEN=$(hanzo auth token)
curl -sS https://api.hanzo.ai/v1/... -H "Authorization: Bearer $TOKEN"
```

The document declares one scheme, `bearer`, at the top level: `Authorization:
Bearer <token>`. `X-Authorization` and HTTP Basic carrying the token as the
password are accepted spellings of the same header; a browser presents the
session cookie instead. There is no product-specific key scheme.

## Surface

| Family | Operations | Addresses under it |
|---|---|---|
| `/v1/sandboxes` | 19 | `end`, `lease`, `read`, `run`, `stop`, `write` |

## Operations

### `/v1/sandboxes`

| Method | Path | Summary |
|---|---|---|
| `GET` | `/v1/sandboxes` | The sandboxes this org holds |
| `POST` | `/v1/sandboxes` | Lease a sandbox |
| `POST` | `/v1/sandboxes/end` | End a sandbox and release it |
| `POST` | `/v1/sandboxes/lease` | Lease a sandbox — a real computer — or resume one you hold |
| `POST` | `/v1/sandboxes/read` | Read a file from a sandbox you hold |
| `POST` | `/v1/sandboxes/run` | Run a command in a sandbox you hold and read its output |
| `POST` | `/v1/sandboxes/stop` | Stop what a sandbox is running, and keep the sandbox |
| `POST` | `/v1/sandboxes/write` | Write a file into a sandbox you hold |
| `DELETE` | `/v1/sandboxes/{id}` | End a sandbox |
| `GET` | `/v1/sandboxes/{id}` | One sandbox |
| `POST` | `/v1/sandboxes/{id}/exec` | Run a command in a sandbox |
| `GET` | `/v1/sandboxes/{id}/fs` | Read a file, or list a directory |
| `POST` | `/v1/sandboxes/{id}/fs` | Write a file |
| `GET` | `/v1/sandboxes/{id}/screen` | The screen, as a page |
| `POST` | `/v1/sandboxes/{id}/screen/ticket` | Open a screen |
| `GET` | `/v1/sandboxes/{id}/screen/ws` | The screen, as a socket |
| `GET` | `/v1/sandboxes/{id}/terminal` | The terminal, as a page |
| `POST` | `/v1/sandboxes/{id}/terminal/ticket` | Open a terminal |
| `GET` | `/v1/sandboxes/{id}/terminal/ws` | The terminal, as a socket |

## Example Calls

```bash
TOKEN=$(hanzo auth token)
```

**Run a command in a sandbox you hold and read its output**

```bash
curl -sS -X POST "https://api.hanzo.ai/v1/sandboxes/run" \
  -H "Authorization: Bearer $TOKEN"
```

Body — `RunIn`:

| Field | Type | Required | What it is |
|---|---|---|---|
| `argv` | array of string | — | Argv is the program and its arguments, already split — the form no shell can misread. Give this or Command,… |
| `blind` | array of string | — | Blind is the set of secrets this command must never publish. |
| `command` | string | — | Command is a shell line, run by `sh -c`. Use it when a pipeline or a redirection is the point, and Argv when… |
| `dir` | string | — | Dir runs the command somewhere other than the sandbox's working directory, which Leased.Workdir names. |
| `id` | string | — | ID is the sandbox to run in, from an earlier lease. |
| `session` | string | — | Session is the live agent session this command narrates into: its output is appended there AS IT IS PRODUCED,… |
| `stdin` | string | — | Stdin is fed to the program on standard input. This is how bytes reach a file without being quoted into a… |
| `timeoutSec` | integer | — | TimeoutSec bounds this ONE command, so a wedged program holds the caller for its own timeout rather than for… |

Runs one command inside the caller's sandbox and answers its exit code, stdout and stderr. A non-zero exit is a successful call carrying a failed program, so it comes back as data and not as an error.

**Lease a sandbox — a real computer — or resume one you hold**

```bash
curl -sS -X POST "https://api.hanzo.ai/v1/sandboxes/lease" \
  -H "Authorization: Bearer $TOKEN"
```

Body — `LeaseIn`:

| Field | Type | Required | What it is |
|---|---|---|---|
| `class` | string | — | Class is what KIND of computer to lease, and the set is closed: |
| `id` | string | — | ID names a sandbox to RESUME, and is the id an earlier lease answered with. Empty asks for a new one. A… |
| `project` | string | — | Project names the disk to attach, and is REQUIRED for every class but `exec`. |
| `runtime` | string | — | Runtime is the isolation boundary asked for: `gvisor` shares a filesystem and holds a project volume,… |
| `ttlSec` | integer | — | TTLSec bounds the lease in seconds. Unset takes the class default. Nothing runs forever, because a sandbox is… |

Leases the caller's sandbox, or returns the one it named if that lease is still running.

**Stop what a sandbox is running, and keep the sandbox**

```bash
curl -sS -X POST "https://api.hanzo.ai/v1/sandboxes/stop" \
  -H "Authorization: Bearer $TOKEN"
```

Body — `StopIn`:

| Field | Type | Required | What it is |
|---|---|---|---|
| `id` | string | — |  |

Interrupts whatever the caller's sandbox is running and answers how many commands it ended. The sandbox stays leased — stop ends the WORK, end ends the RESOURCE — so whoever stopped a run can still read what it left behind.

## Answers

Documented status codes across this product: `200` (5), `204` (1).

## Composition

- `ai` writes the code, this runs it — the pairing behind every code-interpreter loop
- `exec` is the one-shot form of the same primitive: no lease, no id, no lifetime to manage
- `storage` and `s3` hold what a sandbox produced after the sandbox is gone
- `visor` rents the machines underneath; a sandbox is a pod, not a VM you manage

## Related Skills

- `hanzo-api/agent.md` — Autonomous agents for your org: define them, run them, keep every run.
- `hanzo-api/task.md` — Hanzo Tasks: durable workflows that survive a crash, with every run visible and…
- `hanzo-api/tool.md` — Everything your org can call, in one list: connector actions, functions, agents, skills…
- `hanzo-api/automation.md` — Workflows that run themselves, on a schedule or a webhook.
- `hanzo-cloud-architecture/SKILL.md` — how one binary serves all of this
- `hanzo-api/INDEX.md` — every Hanzo product, by domain

---

**Last Updated**: 2026-08-20
**Generated from**: https://api.hanzo.ai/v1/openapi.json
