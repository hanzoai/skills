---
name: hanzo
description: Onboarding manifest for an AI agent operating Hanzo — install, authenticate with a human operator, and work the cloud safely. Read this before running anything against api.hanzo.ai.
---

# Hanzo — Agent Onboarding Manifest (skill.md)

> Hanzo is an AI-native cloud: one API, one identity, one balance. An agent gets
> an identity it can prove, money it can spend under a cap, and a durable place
> to run. This file tells **you, the agent**, how to onboard a human operator and
> operate without doing damage.

> **Read the whole file before you run anything.** The section *What actually
> answers today* is measured, not aspirational, and it names the surfaces that
> are mounted but broken. Skipping it is how you promise something that 500s.

## How to operate — rules that come before any command

You are onboarding an **operator**: the human who owns the account and the bill.

- **Authenticate *with* them. Never invent a credential.** Ask whether they want
  to sign in or create an account, then run the command. Never guess an email,
  never echo a token, never write one into a file the repo tracks.
- **Ask before you grant.** Say which capabilities you intend to use and get a
  yes. Nothing is granted by omission — but everything you *do* write down is real.
- **Stop before anything that spends or cannot be undone.** Topping up, issuing
  a key, deploying, rotating a secret, deleting. Confirm first. Hanzo also
  decides this server-side and can hand the action back for a named human.
- **Verify before you promise.** Run the check, read the answer, then speak. A
  `200` from this platform is not proof of success — see *What actually answers
  today*.
- **Never build a second source of truth.** Auth is Hanzo IAM. Secrets are Hanzo
  KMS. Do not write your own token check, your own key file, your own budget
  arithmetic. Each exists once, and duplicating one is how a governed agent
  stops being governed.
- **Show every URL verbatim.** Checkout and portal links are signed; a trimmed
  one 404s.

## Getting on

**Install.** One of:

```bash
curl -fsSL https://hanzo.sh | sh          # the stack
curl -fsSL https://hanzo.sh/cli | sh      # the CLI alone
brew install hanzoai/tap/hanzo            # macOS
```

**Sign in.** Identity is Hanzo IAM at `hanzo.id` — OIDC, not an API key in a
dotfile. Ask the operator which they want:

```bash
hanzo auth login          # opens the IAM flow; stores a short-lived credential
hanzo auth show           # who you are and which org you are acting as
hanzo auth token          # print the current bearer (do not log it)
hanzo auth list / use     # several identities, one active
```

**Check you are actually connected.**

```bash
curl -s https://api.hanzo.ai/v1/health     # {"revision":"…","status":"ok"}
hanzo billing balance                       # what the operator has to spend
hanzo usage                                 # what has been spent, per run
```

`/v1/health` reports the running revision. It answers `ok` whenever the process
is up — it does **not** mean the model providers behind it are healthy. Read the
next section before you trust it.

## The one API

Everything is `api.hanzo.ai` and everything is under `/v1/`. There is no `/api/`
prefix anywhere and there is no second version — a surface that grows, grows at
`/v1`.

```bash
curl -s https://api.hanzo.ai/v1/models -H "Authorization: Bearer $(hanzo auth token)"
curl -s https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer $(hanzo auth token)" -H 'content-type: application/json' \
  -d '{"model":"fireworks/gpt-oss-120b","messages":[{"role":"user","content":"hi"}]}'
```

The CLI is generated from the same contract the API and the MCP tools serve, so
a thing you can do in one, you can do in all three. It carries **188 command
groups** — `hanzo <group> --help` is the reliable way to find a verb, not guessing.

## What actually answers today — read before you promise

This section is measured against production, not copied from a design doc. It
will age; re-measure rather than trusting it.

**The gateway is up and most of its catalogue is not.** `/v1/models` lists **533**
models. On a serial sweep, roughly **25 of them answer** — about five percent.
Two independent upstream failures, both live:

- **OpenRouter — `402`, out of credit.** `Insufficient credits. Add more using
  https://openrouter.ai/settings/credits`
- **DigitalOcean — `401`, credential rejected.** `Unable to authenticate you`.
  Every bare-name chat model routes here.

**Most of those failures arrive as `HTTP 200` with the error in the body.** This
is the single most important operational fact on this page. A status-code check
reads a dead gateway as healthy:

```json
{"status":"error","msg":"model \"claude-haiku-4.5\": every provider refused — tried openrouter (402)…"}
```

So: **parse the body, not the status.** If `status` is `"error"`, it failed,
whatever the HTTP code said. Report that to the operator plainly rather than
retrying — neither `402` nor `401` clears by retrying.

**Known-good right now:** `fireworks/gpt-oss-120b` and `fireworks/gpt-oss-20b`
(uncatalogued passthrough — they work but are absent from `/v1/models`), plus
the `provider=hanzo` free pool. Prefer one of these when you need a call to
succeed.

**`best` is not a model.** It is absent from `/v1/models` and answers `400`. If
a config names it, that config is wrong.

**`/v1/messages` — the Anthropic-shaped endpoint — has been broken and is being fixed.** It answered `500` (`writer does not implement http.Flusher`), streaming or not. The fix landed on main (`ai v1.833.111 — /v1/messages can answer`) and reaches production with the next release, so **check it before you rely on it**: `curl -s -o /dev/null -w '%{http_code}' -X POST https://api.hanzo.ai/v1/messages …`. `/v1/chat/completions` works either way and is the safe choice.

## The capability set — one name each

Every capability answers at `/v1/<name>`, and that one word is also its OpenAPI tag, its MCP tool, its `hanzo` command group, its SDK class and its docs page (HIP-0139). Know the name and you know the address, the command and the tool — nothing to look up.

**Model** · `ai` `agents` `engine` `evals` `benchmark` `zen` `prompts` `ask`
**Memory** · `brain` `knowledge` `index` `search` `dataset` `crawl` `websearch`
**Compute** · `visor` `sandbox` `exec` `functions` `provisioning` `code`
**Ship** · `git` `projects` `platform` `deploy` `registry` `ingress` `gateway` `domain` `dns` `network` `templates` `blueprint`
**Identity** · `iam` `account` `authz` `kms` `vault` `flags` `entitlements` `sessions`
**Money** · `commerce` `billing` `pricing` `usage` `x402` `captable` `treasury`
**Watch** · `o11y` `event` `audit` `logs`
**Web3** · `web3` `wallets` `validators` `explorer`
**Work** · everything an operator runs — `tasks` `todo` `auto` `flow` `crm` `campaign` `content` `notify` `channels` `help` `guide` `company` `esign` `dataroom` `books` and the rest.

**121 are GA. One — `graph` — is alpha** and answers `404` to an org without the flag. There is no beta. Our own planes (`admin`, `research`, `admission`) are not in the customer contract at all, by design.

Ask the deployment rather than trusting this list: `hanzo --help` for the groups, `hanzo <name> --help` for the verbs, `GET /v1/openapi.json` for the contract.

## Memory — two layers, and they are not the same thing

**`brain`** is agentic durable memory: one SQLite file at `~/.hanzo/brain/brain.db`, read by every Hanzo runtime — TS, Python, Rust, Go. Hybrid retrieval (FTS5 + vector + RRF fusion), a self-wiring graph that extracts edges with **zero LLM calls**, a facts table, and recipes. Storage is pluggable — SQLite by default, or register Qdrant, Meilisearch, ZapDB or PostgreSQL. A **persona** attaches here: the profile the agent carries across runtimes.

```bash
npm install -g @hanzo/bot
hanzo-bot serve                      # drop markdown into ~/.hanzo/workspace/
```

It is **beta**, and it becomes a cloud subsystem under HIP-0106 Phase 1.5 — so `/v1/brains` does not answer on `api.hanzo.ai` yet. Use it locally today.

**`/v1/knowledge`** is the cloud KB: the org's documents and retrieval over them, scoped by the credential.

One is what an agent remembers **across its runtimes**; the other is what an **org** holds in the cloud. Compose them; do not confuse them.

## The loop — local, then shipped

```bash
hanzo up cloud                        # the whole /v1 API on one listener, locally
export HANZO_BASE_URL=http://localhost:8080
```

`hanzoai/cloud` is one binary serving every capability. Develop against it — same addresses, same CLI, same MCP server — then ship:

1. **`git push`** to `git.hanzo.ai`. The forge is a capability (`/v1/git` serves the smart-HTTP endpoints), so an ordinary push **is** the deploy — it fires the platform hook. There is no separate deploy command.
2. **`ci.hanzo.ai`** builds — `hanzo platform builds`.
3. **`cd.hanzo.ai`** reconciles GitOps — `hanzo deploy applications`, `sync`, `rollback`. The declared state is git, so a change applied by hand is reverted on the next sync. Commit it instead.
4. **`platform.hanzo.ai`** runs it — `hanzo platform apps`, `projects`, `fleet`.

## MCP — the tighter binding

One endpoint: `POST https://api.hanzo.ai/v1/mcp`, JSON-RPC 2.0 over ZAP. One tool per capability plus `describe`, so a model searches the list and fetches the schema for the one it picked.

```bash
hanzo code                            # a coding session with the toolset attached
hanzo code --no-mcp                   # opt out
```

Any MCP-native agent reaches the same endpoint:

```json
{ "mcpServers": { "hanzo": { "type": "http", "url": "https://api.hanzo.ai/v1/mcp",
  "headers": { "Authorization": "Bearer sk-live-…" } } } }
```

## The surfaces

| where | what it is |
|---|---|
| **hanzo.ai** | canonical — the company and the product |
| **hanzo.app** | the app builder |
| **hanzo.codes** | code with an agent in your own repo |
| **console.hanzo.ai** | the cloud console |
| **api.hanzo.ai** | the one API, `/v1/*` |
| **hanzo.id** | sign in and account management (IAM) |
| **pay.hanzo.ai** | top up |
| **billing.hanzo.ai** | invoices and usage |
| **hanzoskills.com** | this manifest, every skill, and every HIP — as plain markdown |

## Secrets

Secrets live in Hanzo KMS and nowhere else — not in `.env`, not in the repo, not
in a CI variable. Read them at the point of use:

```bash
hanzo kms secrets list                       # names only, never values
hanzo kms secrets get <NAME>                 # one value
```

Over HTTP the env is part of the address, and omitting it is the usual mistake:

```
GET https://kms.hanzo.ai/v1/kms/secrets/<NAME>?env=prod
```

Without `?env=prod` you get `404 secret not found`, which reads like a missing
secret and is not one.

## The standards

Hanzo's design decisions are written down as **HIPs**, and there are 142 of them.
They are the contract; a doc that disagrees with a HIP is wrong. Read the one
that governs what you are about to build:

```bash
curl hanzoskills.com/hip-0004.md    # the AI gateway
curl hanzoskills.com/hip-0106.md    # one binary
curl hanzoskills.com/hip-0111.md    # auth — read before touching identity
curl hanzoskills.com/hip-0114.md    # ZAP transport
```

The two that will bite you first: **auth is always IAM** (never roll your own,
never add a second gate), and **the cloud is one binary plus per-app plugins**
speaking ZAP — not a fleet of microservices, and not gRPC.

## Everything an agent can read

Every skill and every HIP is plain markdown at a stable address, no auth:

```bash
curl hanzoskills.com/llms.txt        # the index
curl hanzoskills.com/<name>.md       # one document
curl hanzoskills.com/all.md          # the whole corpus, one file
```

## A first run that does not break anything

```bash
curl -fsSL https://hanzo.sh/cli | sh
hanzo auth login                  # WITH the operator — ask first
hanzo auth show                   # confirm the identity and the org
hanzo billing balance             # confirm there is something to spend
curl -s https://api.hanzo.ai/v1/health

# a call that works today
curl -s https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer $(hanzo auth token)" -H 'content-type: application/json' \
  -d '{"model":"fireworks/gpt-oss-120b","messages":[{"role":"user","content":"say ok"}],"max_tokens":8}'
```

Read the body of that last one. If it carries `"status":"error"`, tell the
operator what refused and why — do not retry it and do not route around it.
