# Hanzo API Skills

How to call Hanzo. One skill per product, generated from the document `api.hanzo.ai` serves.

## Category Overview

**Total Skills**: 121
**Surface**: 185 `/v1` families plus 10 roots outside `/v1`, 2492 operations, one bearer token
**Source**: https://api.hanzo.ai/v1/openapi.json
**Focus**: Reaching Hanzo's own products from code and from an agent

The document partitions its operations by the app that serves them. That partition is this catalogue: a product is an app, a family is a `/v1/<family>` prefix inside it. Regenerate with `python3 scripts/generate_hanzo_api_skills.py`.

## Skills in This Category

### Models and inference (9 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `ai.md` | `/v1/ai`, `/v1/audio`, `/v1/chat`, `/v1/completions`, `/v1/dev-bridge` | 302 | Hanzo AI — the model API on /v1 (/v1/chat/completions, /v1/messages, /v1/models and the… |
| `eval.md` *(not yet written)* | `/v1/evals` | 16 | Scoring a model on your own data, with a judge you choose. |
| `benchmark.md` *(not yet written)* | `/v1/benchmark` | 9 | One honest score for any model, on the tests everyone quotes. |
| `research.md` *(not yet written)* | `/v1/research` | 8 | Every experiment you have ever run, kept and comparable. |
| `experiment.md` *(not yet written)* | `/v1/experiments` | 7 | A/B testing anything: a flag, an ad, a subject line, a model. |
| `ml.md` *(not yet written)* | `/v1/ml` | 7 | Model serving: deploy a model behind an endpoint and call it. |
| `prompt.md` *(not yet written)* | `/v1/prompts` | 6 | Your prompt library, versioned, so nothing changes silently. |
| `engine.md` *(not yet written)* | `/v1/engine` | 4 | Hanzo Engine: which models the serving runtime has loaded, and the GPUs under it. |
| `translate.md` *(not yet written)* | `/v1/translate` | 3 | Text in, the same text out in the language you asked for. |

### Agents, code and workflow (14 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `agent.md` *(not yet written)* | `/v1/agent`, `/v1/agents`, `/v1/coding` | 37 | Autonomous agents for your org: define them, run them, keep every run. |
| `task.md` *(not yet written)* | `/v1/tasks` | 20 | Hanzo Tasks: durable workflows that survive a crash, with every run visible and… |
| `tool.md` *(not yet written)* | `/v1/mcp`, `/v1/plugins`, `/v1/skills`, `/v1/tools` | 19 | Everything your org can call, in one list: connector actions, functions, agents, skills… |
| `sandbox.md` | `/v1/sandboxes` | 19 | The ONE compute primitive: a sandbox is a gVisor pod that runs somebody else's code, and… |
| `automation.md` *(not yet written)* | `/v1/automations` | 18 | Workflows that run themselves, on a schedule or a webhook. |
| `auto.md` *(not yet written)* | `/v1/auto` | 11 | Hanzo Auto: build a flow from triggers and actions, publish it, and watch every run. |
| `bot.md` *(not yet written)* | `/v1/bot`, `/v1/bots` | 8 | A bot doing your work on a real desktop, live, while you watch. |
| `flow.md` *(not yet written)* | `/v1/flow` | 8 | Hanzo Flow: build an agent workflow on a visual canvas, run it, and read every run. |
| `code.md` *(not yet written)* | `/v1/code` | 7 | Search and symbols across your repos, for you and your agents. |
| `marketplace.md` *(not yet written)* | `/v1/marketplace` | 6 | The shop for tools and agents: browse, install into your project, publish your own free… |
| `lsp.md` *(not yet written)* | `/v1/code` | 5 | Offers the candidates a language server has at a position, typed and resolved through the… |
| `exec.md` *(not yet written)* | `/v1/download`, `/v1/exec`, `/v1/files`, `/v1/upload` | 5 | The code interpreter: run a snippet in a sandbox, and move files in and out of the… |
| `node.md` *(not yet written)* | `/v1/bot` | 4 | Your own machines, connected and ready to take a command. |
| `skill.md` *(not yet written)* |  | 2 | The skill catalogue an AI client reads to learn what it can do. |

### Retrieval (8 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `index.md` *(not yet written)* | `/v1/index` | 17 | Fast full-text search over your own data, typos forgiven. |
| `knowledge.md` *(not yet written)* | `/v1/kb` | 9 | Your team's wiki and your agents' memory, searchable by meaning. |
| `websearch.md` *(not yet written)* | `/v1/scrape`, `/v1/websearch` | 7 | A web search and a page fetch your agents can call. |
| `world.md` *(not yet written)* | `/v1/world` | 6 | A live news feed filtered to what your project cares about. |
| `product.md` *(not yet written)* | `/v1/search`, `/v1/vector` | 4 | The read-only inventory of the search and vector backends: /v1/search/{indexes,stats}… |
| `ask.md` *(not yet written)* | `/v1/ask` | 2 | A plain-language question about your business, answered with real numbers. |
| `crawl.md` *(not yet written)* | `/v1/crawl` | 1 | Any web page turned into clean markdown a model can read. |
| `search.md` *(not yet written)* | `/v1/search` | 1 | One ranked result set over everything your org has stored. |

### Identity and access (8 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `iam.md` *(not yet written)* | `/v1/iam` | 206 | Hanzo's identity provider: users, organizations, applications, and the OIDC/OAuth2… |
| `admin.md` *(not yet written)* | `/v1/admin` | 56 | The operator's view of the fleet: orgs, users, roles, spend and system health. |
| `account.md` *(not yet written)* | `/v1/appearance`, `/v1/avatar`, `/v1/csrf`, `/v1/embed`, `/v1/keys`, ` | 10 | Your own account: API keys you mint and revoke, and org onboarding. |
| `plugin.md` *(not yet written)* | `/v1/admin` | 4 | What each host is running, and how to change it: enable, disable, reload, or pin a… |
| `authz.md` *(not yet written)* | `/v1/authz` | 3 | Ask whether a subject may act on an object |
| `entitlement.md` *(not yet written)* | `/v1/entitlements`, `/v1/orgs` | 3 | What your org may run: what the plan grants, and which of those products are switched on. |
| `pref.md` *(not yet written)* | `/v1/prefs` | 2 | Your own settings — theme, density, pinned nav — following you across every Hanzo app. |
| `setting.md` *(not yet written)* | `/v1/settings` | 2 | How an org configures each product it uses, secret fields included. |

### Secrets and assurance (5 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `wallet.md` *(not yet written)* | `/v1/wallets` | 8 | Blockchain key custody: create wallets, rotate their keys, and sign with them. |
| `kms.md` | `/v1/kms` | 7 | Secret custody: your org's secrets sealed at rest, plus threshold signing. |
| `security.md` *(not yet written)* | `/v1/security` | 7 | Secret scanning for your code: submit sources, get findings, masked never raw. |
| `sbom.md` *(not yet written)* | `/v1/sbom` | 3 | What is inside a container image: every component, resolvable by digest or image ref. |
| `audit.md` *(not yet written)* | `/v1/audit` | 1 | Your org's tamper-evident audit trail: every security-relevant event, hash-chained and… |

### Money (14 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `commerce.md` *(not yet written)* | `/v1/billing`, `/v1/cart`, `/v1/catalog`, `/v1/commerce`, `/v1/payment | 219 | Selling: checkout, subscriptions, invoices, spend alerts, payment webhooks and the… |
| `pricing.md` *(not yet written)* | `/v1/admin`, `/v1/enablement`, `/v1/pricing` | 33 | The price list: what every model, provider, GPU tier, tool and hosting plan costs. |
| `book.md` *(not yet written)* | `/v1/books` | 25 | Double-entry accounting: chart of accounts, ledger, bank reconciliation, and the reports… |
| `affiliate.md` *(not yet written)* | `/v1/admin`, `/v1/affiliates` | 17 | A partner program that pays commission on what your referrals spend. |
| `plan.md` *(not yet written)* | `/v1/plans` | 15 | The plan catalog: every tier you can buy, what it costs, and what it grants. |
| `author.md` *(not yet written)* | `/v1/admin`, `/v1/authors` | 11 | A royalty for open-source work: your repo runs, you get paid. |
| `licensing.md` *(not yet written)* | `/v1/licensing` | 11 | Download resolves a release to its artifact, gated on a valid license. |
| `billing.md` *(not yet written)* | `/v1/billing`, `/v1/finance` | 9 | Your org's balance, what it has spent, and the cards it pays with. |
| `treasury.md` *(not yet written)* | `/v1/admin`, `/v1/finance` | 8 | Returns the whole treasury board for a SuperAdmin: the reserve fund report, the recent… |
| `leaderboard.md` *(not yet written)* | `/v1/usage` | 6 | Activity returns the per-day usage series for ONE authorized subject — the points a… |
| `usage.md` *(not yet written)* | `/v1/usage` | 5 | What your org ran and what it cost, broken down per account. |
| `referral.md` *(not yet written)* | `/v1/admin`, `/v1/referrals` | 4 | Referral ATTRIBUTION: who referred whom, and whether that referee ever became a real… |
| `allowance.md` *(not yet written)* | `/v1/allowance` | 1 | How much a plan lets you do without paying, and how much of it you have left today. |
| `x402.md` *(not yet written)* | `/v1/x402` | 1 | Pay-per-request over HTTP 402: quote a price, take the payment, serve the resource. |

### Compute and data plumbing (8 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `visor.md` *(not yet written)* | `/v1/clusters`, `/v1/compute`, `/v1/fleet`, `/v1/gpus`, `/v1/k8s`, `/v | 34 | The compute you rent from Hanzo: machines, GPUs and clusters — launch one, resize it,… |
| `provisioning.md` *(not yet written)* | `/v1/instances` | 28 | One-click data add-ons: a SQL, key-value, document, vector, search or object store, wired… |
| `pubsub.md` *(not yet written)* | `/v1/pubsub` | 18 | Your message bus: publish, subscribe, and durable streams your apps read at their own… |
| `mq.md` *(not yet written)* | `/v1/mq` | 15 | Queue and stream admin for your org: create them, watch them drain, ack what you pulled. |
| `storage.md` *(not yet written)* | `/v1/s3` | 8 | Object storage: your buckets and the files in them, with signed URLs for upload and… |
| `registry.md` *(not yet written)* | `/v1/registry` | 6 | Your container and package registry: push images, pull them back, see what you store. |
| `sync.md` *(not yet written)* | `/v1/sync` | 6 | Data sync: link two endpoints and keep them in step, on a webhook, on a schedule, or on… |
| `venue.md` *(not yet written)* | `/v1/cloud` | 5 | Bring your own cloud: link a DigitalOcean, AWS or GCP account and its clusters show up… |

### Shipping and the front door (16 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `git.md` *(not yet written)* | `/v1/git` | 58 | Git hosting for your org: create repos, clone, push, and see what they cost. |
| `project.md` *(not yet written)* | `/v1/edge`, `/v1/platform`, `/v1/projects`, `/v1/sites`, `/v1/tags` | 58 | Where your sites live: create one, deploy a build, roll back to any release. |
| `platform.md` *(not yet written)* | `/v1/builds`, `/v1/environments`, `/v1/git-webhook`, `/v1/pipelines`,  | 37 | Hanzo PaaS: deploy containers to your own tenant namespace — builds, releases,… |
| `cloudflare.md` *(not yet written)* | `/v1/cloudflare` | 33 | Your Cloudflare account, managed from Hanzo: zones, Pages, Workers, Workers AI, R2, KV… |
| `deploy.md` *(not yet written)* | `/v1/deploy` | 21 | Hanzo CD: see what each app is running, sync it, and roll back a bad release. |
| `ingress.md` *(not yet written)* | `/v1/ingress` | 18 | Your front door: automatic TLS certificates and hostname routing to any backend, changed… |
| `function.md` *(not yet written)* | `/v1/functions` | 11 | Your serverless code: publish it, call it over HTTP, watch every run and what it cost. |
| `domain.md` *(not yet written)* | `/v1/domain` | 7 | Hanzo Domains: search a name, see the price, buy it from your prepaid wallet. |
| `dns.md` *(not yet written)* | `/v1/dns` | 5 | Your DNS records: the zones and records behind every name you point at Hanzo. |
| `template.md` *(not yet written)* | `/v1/templates` | 5 | A gallery of starter kits you can deploy as they come. |
| `zt.md` *(not yet written)* | `/v1/mesh`, `/v1/networks` | 4 | The Hanzo Cloud NETWORKING surface: the tenant's Hanzo Zero Trust footprint — overlay… |
| `blueprint.md` *(not yet written)* | `/v1/blueprint` | 3 | What a template costs to run, worked out before you deploy. |
| `gateway.md` *(not yet written)* | `/v1/gateway` | 3 | Live control of the policy your API applies to every incoming request: CORS, rate limits,… |
| `share.md` *(not yet written)* | `/v1/share` | 2 | A public URL for a service on your own machine, and a list of what you have open. |
| `base.md` *(not yet written)* | `/v1/base` | 1 | Managed Hanzo Base: a hosted backend for your app — collections, records, access rules… |
| `catalog.md` *(not yet written)* | `/v1/catalog` | 1 | One place to browse every project, app and site built here. |

### Telemetry (4 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `o11y.md` *(not yet written)* | `/v1/o11y`, `/v1/sentinel`, `/v1/summary` | 387 | Your logs, metrics and traces: ship them in, query them, chart them. |
| `analytics.md` *(not yet written)* | `/v1/analytics`, `/v1/errors`, `/v1/event`, `/v1/event.js`, `/v1/insig | 12 | Product analytics: send an event, read back who did what. |
| `metric.md` *(not yet written)* | `/v1/logs`, `/v1/metrics`, `/v1/traces` | 11 | How many log records this deployment holds for your org |
| `destination.md` *(not yet written)* | `/v1/destinations` | 5 | Your events forwarded to the ad and analytics tools you use. |

### Reaching people (14 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `integration.md` *(not yet written)* | `/v1/connectors`, `/v1/integrations` | 48 | How your org connects third-party accounts like Slack, and revokes them. |
| `marketing.md` *(not yet written)* | `/v1/marketing` | 35 | Lifecycle email: drip sequences that reach the right people. |
| `crm.md` *(not yet written)* | `/v1/crm` | 20 | Your sales pipeline: the companies, the people, the deals in play. |
| `meet.md` *(not yet written)* | `/v1/meet` | 13 | The virtual office: it decides who may join a room and mints the short-lived token that… |
| `social.md` *(not yet written)* | `/v1/social` | 13 | Posting to every social account you own, now or on a schedule. |
| `campaign.md` *(not yet written)* | `/v1/campaign` | 11 | One go-to-market push across paid, organic and email at once. |
| `link.md` *(not yet written)* | `/v1/links` | 11 | The unified AI login manager's registry: the org+user-scoped record of WHICH provider… |
| `tel.md` *(not yet written)* | `/v1/tel` | 10 | The telecommunications surface: phone numbers, calls and messages, on whatever carrier… |
| `webhook.md` *(not yet written)* | `/v1/webhooks` | 8 | How your app hears about events: register an endpoint, pick the events, get each one… |
| `ad.md` *(not yet written)* | `/v1/ads` | 7 | Your paid ad campaigns, launched and paused from one place. |
| `channel.md` *(not yet written)* | `/v1/channels` | 7 | One inbox for the chat apps you connect — Discord, Slack, Teams, Telegram. |
| `content.md` *(not yet written)* | `/v1/content` | 6 | Marketing content from draft to published, on every channel. |
| `help.md` *(not yet written)* | `/v1/help` | 4 | A support desk: customers file tickets, your team answers them. |
| `notify.md` *(not yet written)* | `/v1/notify` | 4 | Transactional email and SMS, sent through your org's own provider credential. |

### Running a company (17 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `captable.md` *(not yet written)* | `/v1/captable` | 31 | Your cap table: stakeholders, share classes, grants, SAFEs, rounds, and who owns what. |
| `company.md` *(not yet written)* | `/v1/company` | 22 | Incorporation end to end: pick a structure, add founders, pay, file, and e-sign. |
| `team.md` *(not yet written)* | `/v1/team` | 19 | Your org's shared workspace: documents edited together, files, seats, and agents as… |
| `framework.md` *(not yet written)* | `/v1/framework` | 19 | Document types you define: describe a record once, then create, list, submit and cancel… |
| `guide.md` *(not yet written)* | `/v1/guide` | 19 | A step-by-step checklist that gets your business running on AI. |
| `compliance.md` *(not yet written)* | `/v1/compliance` | 17 | Your KYC/KYB onboarding, accreditation records, and the evidence trail behind them. |
| `dataroom.md` *(not yet written)* | `/v1/dataroom` | 17 | A secure document room you share by link and watch page by page. |
| `esign.md` *(not yet written)* | `/v1/esign` | 13 | A document out for signature, signed and filed with an audit trail. |
| `todo.md` *(not yet written)* | `/v1/todo` | 12 | Hanzo Todo: boards, the work items on them, and the filters that make a board. |
| `legal.md` *(not yet written)* | `/v1/legal` | 11 | The paperwork your company needs, drafted, signed and filed. |
| `risk.md` *(not yet written)* | `/v1/risk` | 11 | HANZO RISK's model plane: the per-organisation feature surface and the per-organisation… |
| `flag.md` *(not yet written)* | `/v1/flags` | 8 | Feature flags: ship it dark, then turn it on for who you pick. |
| `dataset.md` *(not yet written)* | `/v1/risk` | 7 | List this org's datasets |
| `label.md` *(not yet written)* | `/v1/risk` | 7 | Read the assertions this tenant has recorded |
| `reference.md` *(not yet written)* | `/v1/risk` | 6 | Lists every set this plane publishes, with its version and how fresh it is. |
| `taxonomy.md` *(not yet written)* | `/v1/taxonomy` | 5 | The product catalogue's shape: which categories exist, what each product is called, which… |
| `admission.md` *(not yet written)* | `/v1/flags` | 1 | Reports whether ONE host is currently gated by the launch waitlist. |

### Chain (3 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `web3.md` *(not yet written)* | `/v1/chains`, `/v1/rpc`, `/v1/tokens` | 4 | The chain-access surface: which chains this deployment can reach, a JSON-RPC door onto… |
| `validator.md` *(not yet written)* | `/v1/validators` | 4 | One-click validator onboarding: prove your Genesis NFT, get a node provisioned, queue its… |
| `explorer.md` *(not yet written)* | `/v1/indexers`, `/v1/oracles` | 2 | Chain data: your block indexers and how far each has caught up, plus the on-chain price… |

### The document itself (1 skills)

| Skill | Serves | Operations | Purpose |
|---|---|---|---|
| `openapi.md` *(not yet written)* | `/v1/commands`, `/v1/openapi.json` | 3 | The API description, at the conventional address |

---

**Last Updated**: 2026-08-20
