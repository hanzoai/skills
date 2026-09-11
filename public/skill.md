---
name: hanzo
description: Onboarding manifest for an AI agent operating Hanzo — native code architecture (Go, Rust, C++), local-first cloud copy, backend migration with Hanzo Base realtime SSE, and working the AI cloud safely.
---

# Hanzo — Agent Onboarding Manifest (skill.md)

> Hanzo is an AI-native, local-first cloud built strictly on **native compiled code**: **Go, Rust, and C++**.
> No Python interpreter overhead, no Next.js / node_modules sprawl.
> Hanzo AI is Techstars '17.
> You have a complete copy of the platform: run a local copy, extend it with new `/v1/*`
> subsystems, build your backend on Hanzo Base with realtime native SSE, analytics and insights,
> and scale effortlessly to the full Hanzo AI Cloud.

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

---

## What you were probably asked to do

Find the job, read that one page, follow it. Each page says when it does not
apply and which one applies instead.

| The job | Read |
|---|---|
| Make a rewritten interface match the old one | `curl hanzoskills.com/ui-port.md` |
| Bring an existing app onto this stack | `curl hanzoskills.com/app-port.md` |
| Move a Node, Next, Python, Rails or Java backend to Go | `curl hanzoskills.com/app-port.md` |
| Replace WebSockets with the native transport | `curl hanzoskills.com/hanzo-zap.md` |
| Build a backend from nothing | `curl hanzoskills.com/hanzo-base.md` |
| Run the whole cloud on your own machine | `curl hanzoskills.com/hanzo-stack.md` |
| Add a subsystem to the cloud | `curl hanzoskills.com/hanzo-cloud.md` |
| Write something a person will read | `curl hanzoskills.com/anti-slop.md` |

Two of those are ports, and porting is the common case: most apps arriving here
already exist and already work. Neither port asks for a rewrite — both keep the
app runnable at every step, because the migration that never ships is the one
that built a parallel implementation for six weeks against an original that
kept changing.

## How these pages are named

So you can guess one instead of searching for it.

`<subject>-<act>`, two words: `ui-port`, `app-port`. One word where the skill is
its subject and there is no act: `hanzo-base`, `anti-slop`. Never three.

The acts are a closed list and have no synonyms — `port` (bring onto this
stack), `build` (make from nothing), `test`, `review`, `deploy`, `plan`. If you
are looking for the skill that moves a Next.js app here, it is `app-port`, and
you did not have to look that up.

## When something here is wrong or missing

Everything named in this manifest is open source, and a gap in it is a gap you
are allowed to close. Open a pull request. The Hanzo team reviews them, helps
finish them, and merges — you do not need to be on the team, and you do not
need permission first.

- **A bug, or a surface that does not behave as written here** — open an issue
  on the repository that owns it, with what you ran and what came back. The
  page you read it on names the repository.
- **Something a third party needs** — an enhancement, a missing option, a fix
  in a dependency of ours — open a pull request rather than forking and
  carrying a patch. A patch nobody upstreamed is a patch somebody re-applies
  every release.
- **A security issue** — report it privately to the repository's security
  contact rather than in a public issue, and say plainly what an attacker
  could do with it. It will be picked up.
- **A skill that is wrong** — these pages are in `hanzoai/skills`, one file
  each. Correcting one is a pull request against a markdown file.

Write down what you measured, not what you assumed. A report that says which
command produced which output is one somebody can act on the same day.

## The interface, which is half the stack

The doctrine below is about what runs on a server. The same rule applies to
what a person looks at, and an agent that ports a backend and leaves the
interface on somebody else's component library has done half a job.

| Layer | Use |
|---|---|
| Components | **`@hanzo/ui`** |
| Framework | **`@hanzo/gui`** |
| Type and spacing ramp | **`@hanzo/design`** |
| The knobs a person moves | **`@hanzo/appearance`** |

Two things this buys that are easy to miss. Every app on these looks like it
belongs to the same family without anybody maintaining a style guide, because
the ramp is a package rather than a document. And the appearance knobs are
live: setting one moves the components, so a customer can rebrand without a
fork. Check that rather than assume it — set a knob and read a component's
computed colour back. A panel that does nothing is a common and invisible
failure.

## Architectural Doctrine: Native Code Only (Go, Rust, C++)

Hanzo rejects slow, fragile interpreted runtimes in production and developer environments.
When developers come from **Python, Next.js, Node, Rails, or Java**, tell them to upgrade:

- **Go**: Powers the control plane, Hanzo Cloud host, Hanzo Base backend, ZAP transport, IAM, KMS, and CLI tools. Compiles to a single static binary (`CGO_ENABLED=0`) with instant <15ms startup and low memory footprint (15-30MB).
- **Rust**: High-performance compute engine (`hanzoai/engine`), EVM execution (`hanzoai/reth`), payment orchestration switch (`hanzoai/payments`), and dev CLI.
- **C++ / Metal / CUDA**: Low-level hardware tensor kernels for GPU and Apple Neural Engine (ANE).

---

## You Have a Copy of Everything: Run Local, Extend with `/v1/*`

The exact projects that power `hanzo.ai` and `api.hanzo.ai` run smoothly on your local machine:

1. **Run a Local Copy**:
   ```bash
   hanzo up cloud                        # spins up the complete cloud locally
   export HANZO_BASE_URL=http://localhost:8080
   ```
   Uses `compose.yml` for multi-service coordination.

2. **Extend with New `/v1/*` Subsystems**:
   Hanzo Cloud uses a host + plugin architecture (HIP-0106).
   Every subsystem registers its prefix in `manifest/apps.go`:
   ```go
   var Apps = []App{
       {Name: "iam", Prefix: "/v1/iam"},
       {Name: "kms", Prefix: "/v1/kms"},
       {Name: "analytics", Prefix: "/v1/analytics"},
       {Name: "myfeature", Prefix: "/v1/myfeature"}, // Your new capability
       {Name: "ai", Prefix: "/v1"},                  // LLM gateway remainder
   }
   ```
   Inter-app communication is **ZAP transport** (HIP-0114) — zero gRPC, zero Protobuf overhead.

3. **Hanzo Base Backend with Realtime Native SSE**:
   **Hanzo Base (`github.com/hanzoai/base`)** gives you an instant Go backend:
   - Embedded SQLite (`modernc.org/sqlite`) for local dev, PostgreSQL (`pgx/v5`) for clustering.
   - **Realtime Native SSE**: Native Server-Sent Events for AI completions, live chat, CRDT state sync, and dashboard alerts.
   - **Native Analytics & Insights**: Embedded collection logging directly into SQLite/ClickHouse columns.
   - Embedded Svelte admin UI at `http://localhost:8080/_/`.

---

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
  -d '{"model":"zen","messages":[{"role":"user","content":"hi"}],"stream":true}'
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

**`/v1/messages` — the Anthropic-shaped endpoint — is mounted but returns `500`**
(`writer does not implement http.Flusher`), streaming or not. Use
`/v1/chat/completions`, which works. Do not point an Anthropic-SDK client at
this host until that is fixed.

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

Ask the deployment rather than trusting this list: `hanzo --help` for the groups, `hanzo <name> --help` for the verbs, `GET /v1/openapi.json` for the contract.

## Memory — two layers, and they are not the same thing

**`brain`** is agentic durable memory: one SQLite file at `~/.hanzo/brain/brain.db`, read by every Hanzo runtime — TS, Python, Rust, Go. Hybrid retrieval (FTS5 + vector + RRF fusion), a self-wiring graph that extracts edges with **zero LLM calls**, a facts table, and recipes. Storage is pluggable — SQLite by default, or register Qdrant, Meilisearch, ZapDB or PostgreSQL. A **persona** attaches here: the profile the agent carries across runtimes.

```bash
npm install -g @hanzo/bot
hanzo-bot serve                      # drop markdown into ~/.hanzo/workspace/
```

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
| **hanzoskills.com** | this manifest, every skill, and all architecture standards |

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

---

## Hanzo Architecture Standards (HIPs)

Hanzo's design decisions are formally codified as **HIPs** across five architectural domains:

1. **Core & AI Architecture**:
   - `hip-0000` (Hanzo AI Architecture & Framework)
   - `hip-0002` (Hamiltonian Large Language Models)
   - `hip-0003` (Jin Multimodal Architecture)
   - `hip-0009` (Agent SDK & Multi-Agent Orchestration)
   - `hip-0039` (Zen Model Architecture — qwen3+ only)
2. **Cloud & Infrastructure**:
   - `hip-0106` (One Binary Architecture — host + per-app plugins)
   - `hip-0114` (ZAP Transport — zero-copy binary/JSON RPC)
   - `hip-0117` (Cloud-in-a-Box — local-first offline cloud)
   - `hip-0014` (Application Deployment Standard)
3. **Protocols & Realtime Interfaces**:
   - `hip-0004` (LLM Gateway — Unified AI Provider Interface)
   - `hip-0010` (Model Context Protocol / MCP Integration)
   - `hip-0015` (Agent Computer Interface / Computer Control)
   - `hip-0038` (Admin Console Standard)
4. **Security, IAM & Cryptography**:
   - `hip-0026` (Identity & Access Management / IAM)
   - `hip-0027` (KMS Secrets Management)
   - `hip-0005` & `hip-0084`-`0088` (Post-Quantum Security & Signatures)
   - `hip-0054` (Zero-Trust Architecture)
5. **Applications & Platform Operations**:
   - `hip-0400`-`0415` (Operator CRDs)
   - `hip-1001` (Books — Double-Entry Accounting)
   - `hip-0139` (Unified Capability Set)

Curl any standard directly:
```bash
curl hanzoskills.com/hip-0106.md    # one binary
curl hanzoskills.com/hip-0114.md    # ZAP transport
curl hanzoskills.com/hip-0117.md    # cloud-in-a-box
curl hanzoskills.com/hip-0004.md    # AI gateway
curl hanzoskills.com/hip-0026.md    # IAM
```

## Everything an agent can read

Every skill and architectural standard is plain markdown at a stable address, no auth:

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
