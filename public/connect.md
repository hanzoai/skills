---
name: connect
description: "Reach a platform's API — Google Ads, Meta, X, LinkedIn, Analytics, Salesforce, Slack, GitHub and the rest — through Hanzo's own connector plane, so the credential is custodied and never written to disk. Use whenever a skill needs live data or write access to an outside service: 'connect my Google Ads', 'pull my Meta spend', 'post to LinkedIn', 'read Search Console', 'is my account connected', 'refresh the token', 'disconnect'. Every platform skill defers here rather than running an OAuth flow of its own."
---

# Connect

One way to reach an outside platform, for every skill that needs one.

## Why this exists

A platform skill written on its own tends to carry its own OAuth dance, its own
SDK install and its own credential file — `pip install <vendor>`, then a JSON
blob under `.agents/`. That is three problems at once: a long-lived secret in
plaintext on a disk somebody else can read, a token nobody can rotate, and as
many auth implementations as there are skills.

Hanzo already holds the connectors, in Go, per user. They are the one door.

- **The secret never leaves custody.** A credential is verified live, then
  sealed into KMS before any row is written. It appears in no row, no response
  and no log — the single exception is the token read below, which exists
  because a caller has to get one somehow.
- **A connector belongs to a person, not a workspace.** Every read and write is
  bound to `(org, user)`, so another user's connector id is simply not there.
- **Rotation is a request, not a migration.** One POST re-mints.

## The surface

Host is `api.hanzo.ai`. Send the signed-in user's bearer; send no org header —
the org comes from the token.

```
GET    /v1/integration/connectors                                 what this user has connected
GET    /v1/integration/connectors/providers                       what can be connected
GET    /v1/integration/connectors/{id}/token                      a usable access token
POST   /v1/integration/connectors/{provider}/device               begin a device sign-in
POST   /v1/integration/connectors/{provider}/device/{flow}/poll   wait for it to finish
POST   /v1/integration/connectors/{provider}/credential           hand over a token you already hold
POST   /v1/integration/connectors/{id}/refresh                    rotate now
DELETE /v1/integration/connectors/{id}                            forget it, and its secrets
```

A connector's **id is `provider:label`** — so one person can hold two Google Ads
accounts as `google_ads:work` and `google_ads:client`, and a skill that wants
"the" account should ask which when there is more than one.

## Two ways in, one plane behind them

The surface above is HTTP, and an agent with a shell can call it directly. It is
also projected as MCP tools at `https://api.hanzo.ai/v1/mcp`, so an agent with
the Hanzo MCP server attached can reach the same connectors as tools without
composing a request. Either is correct — they are the same routes, typed once in
Go and projected, so a tool and a curl cannot come to disagree about what a
connector is.

Prefer the tools where they are attached: an agent that has them gets argument
validation and does not have to hold a token in a shell variable.

## The flow a skill follows

```
1. GET /v1/integration/connectors
       └─ already there?  → step 3
2. POST /v1/integration/connectors/{provider}/device
       └─ show the user the code, then poll .../device/{flow}/poll until it settles
3. GET /v1/integration/connectors/{id}/token
4. call the platform with that token
       └─ 401 → POST .../{id}/refresh once, then retry
```

Ask for a token when you are about to use it, not at the start of a session: it
is short-lived by design, and holding one across a long conversation is how a
skill ends up refreshing something it never needed.

## Providers

These are the ids the plane actually serves. Naming one it does not is a skill
that fails at the first call with nothing useful to say.

**Marketing and ads** — `google_ads`, `meta_ads`, `adroll`, `x`, `linkedin`,
`tiktok`, `pinterest`, `reddit`, `youtube`, `instagram`, `threads`, `facebook`

**Data and analytics** — `google_analytics`, `google_bigquery`, `google_cloud`,
`google`, `salesforce`

**Messaging** — `slack`, `discord`, `teams`, `telegram`, `whatsapp`, `twitch`

**Work** — `github`, `linear`, `microsoft`

`GET /v1/integration/connectors/providers` is the live answer and outranks this
list, which is a convenience and will age.

### What is not here

**Google Search Console has no provider.** A skill that needs it should say so
and stop, rather than reaching for a vendor SDK and a credentials file — that is
the arrangement this skill replaces. `google_analytics` and `google_bigquery`
answer many of the same questions and are connected.

## Rules

- **Never write a credential to disk.** No `.agents/*credentials*.json`, no
  `.env`, no config file. If a skill's instructions tell you to, they predate
  this plane and the token read above is what they wanted.
- **Never install a vendor SDK to authenticate.** The token is HTTP; so is every
  platform API it opens.
- **Never ask the user to paste a long-lived secret** when the provider supports
  the device flow. Where they already hold one, `POST .../credential` takes it
  and seals it.
- **A refusal is the platform's, and is worth repeating verbatim.** "Not
  connected" and "connected but the scope is missing" have different remedies.
