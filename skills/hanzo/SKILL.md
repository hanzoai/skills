---
name: hanzo
description: "Sign up free with a wallet. Get an API key free. A small free allowance to try it. Top up in USDC or USDT over the API to raise your limits and pay as you go. No human required. Onboard an agent onto Hanzo, the AI operating system for business: get a credential, make a first call to api.hanzo.ai, add the Hanzo MCP server, evaluate Kai, and pay. Load when building with or on Hanzo or calling api.hanzo.ai; when getting a Hanzo API key or token (signing in with a wallet, no human; the hanzo CLI device flow; an agent's own email-code sign-up; minting a key); when adding Hanzo MCP to Claude Code, Codex, Cursor or Zed; when making or evaluating Kai decisions; when giving agents keys with budgets, rate limits or expiry; when drafting an ad or email campaign over the API; or when paying for Hanzo (plans, card checkout, USDC or USDT deposits, wire)."
license: MIT
---

# Hanzo

Sign up free with a wallet. Get an API key free. A small free allowance to try it. Top up in USDC or USDT over the API to raise your limits and pay as you go. No human required.

The AI operating system for business. Building business software since 2014. Techstars '17.

## Start: no human required

Identity is `https://hanzo.id`; the API is `https://api.hanzo.ai/v1`. One request a step, with its answer. In full: https://docs.hanzo.ai/docs/agents/headless

1. Challenge: `GET https://hanzo.id/v1/iam/web3/nonce?chain=evm&address=0x…` -> `{"status":"ok","data":{"domain":"hanzo.id","uri":"https://hanzo.id/login","statement":"…","nonce":"<32 hex>","issuedAt":"…","expirationTime":"…","version":"1"}}`. It lives 10 minutes.
2. Sign: build the CAIP-122 message from `data` (as the file below does) and sign it with EIP-191 `personal_sign`.
3. Verify: `POST https://hanzo.id/v1/iam/web3/verify` `{"clientId":"hanzo-ai","chain":"evm","scheme":"secp256k1-eip191","address":"0x…","message":"…","signature":"0x…","type":"code","redirectUri":"http://localhost:3000/callback","scope":"openid profile email","codeChallenge":"<S256>","codeChallengeMethod":"S256"}` -> `{"status":"ok","data":"<authorization code>"}`. A first-seen wallet gets its own account and organization.
4. Token: `POST https://hanzo.id/v1/iam/oauth/token` (form: `grant_type=authorization_code`, `client_id=hanzo-ai`, `code`, `code_verifier`, `redirect_uri`) -> `{"access_token":"…","id_token":"…","refresh_token":"…","token_type":"Bearer","expires_in":604800}`. Seven days.
5. Key, free: `POST https://api.hanzo.ai/v1/account/keys` `{"name":"agent","type":"secret"}` -> `{"id":"…","name":"agent","type":"secret","key":"sk-live-…","status":"active"}`. `key` is shown once.
6. Limits, read free: `GET /v1/billing/plans` (no key), `GET /v1/billing/tier` (`balance.effectiveAvailable` is the one figure to compare to zero), `GET /v1/allowance` (`limit`, `used`, `resets`, `window`, `spent`).
7. A 429 with `"code":"free_plan_cap"` means the free allowance is used: `{"error":{"type":"rate_limit_error","code":"free_plan_cap","resets_at":"…","topup_url":"…","actions":[…]}}`, plus `Retry-After`. Wait for `resets_at`, or top up.
8. Top up: `GET /v1/billing/crypto/options` -> `{"chains":["base","ethereum","solana"],"tokens":["usdc","usdt"]}`. `POST /v1/billing/crypto/deposit` `{"chain":"base","token":"usdc","amountCents":2000}` -> `{"id":"…","status":"pending","chain":"base","token":"usdc","depositAddress":"0x…","expiresAt":"…"}`. Send exactly to `depositAddress`. Poll `GET /v1/billing/crypto/deposit/{id}`: `pending`, `confirming`, `succeeded`.
9. Balance: `GET /v1/billing/balance` -> `{"balance","holds","available","account","cash","credit"}` in whole USD cents. Calls past the allowance draw on `available`: pay as you go.

The same flow as one file, run end to end:

```ts
// No human required: sign in with a wallet, get a key, call the API, top up in USDC.
// A first-seen wallet gets its own Hanzo account and organization, on the Free plan.
// npm i viem   ·   node headless.mts   (Node 22.18+; set HANZO_WALLET_KEY to use your own wallet)
import { createHash, randomBytes } from 'node:crypto'
import { generatePrivateKey, privateKeyToAccount } from 'viem/accounts'

const ID = 'https://hanzo.id'
const API = 'https://api.hanzo.ai'
const CLIENT = 'hanzo-ai'
const REDIRECT = 'http://localhost:3000/callback'

// Your agent's own wallet, or a fresh one. The key never leaves this process.
const account = privateKeyToAccount((process.env.HANZO_WALLET_KEY as `0x${string}`) ?? generatePrivateKey())

const json = async (r: Response) => {
  const body = await r.json()
  if (!r.ok || body?.status === 'error') throw new Error(`${r.url} ${r.status} ${JSON.stringify(body)}`)
  return body
}

// 1. A one-time challenge for this wallet.
const { data: c } = await json(await fetch(`${ID}/v1/iam/web3/nonce?chain=evm&address=${account.address}`))

// 2. The CAIP-122 message the challenge describes, signed with EIP-191 personal_sign.
const message = [
  `${c.domain} wants you to sign in with your Ethereum account:`,
  account.address,
  '',
  ...(c.statement ? [c.statement, ''] : []),
  `URI: ${c.uri}`,
  `Version: ${c.version}`,
  `Nonce: ${c.nonce}`,
  `Issued At: ${c.issuedAt}`,
  ...(c.expirationTime ? [`Expiration Time: ${c.expirationTime}`] : []),
].join('\n')
const signature = await account.signMessage({ message })

// 3. Verify: a first-seen wallet gets an account and an organization of its own.
const verifier = randomBytes(32).toString('base64url')
const challenge = createHash('sha256').update(verifier).digest('base64url')
const { data: code } = await json(
  await fetch(`${ID}/v1/iam/web3/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      clientId: CLIENT, chain: 'evm', scheme: 'secp256k1-eip191', address: account.address, message, signature,
      type: 'code', redirectUri: REDIRECT, scope: 'openid profile email', codeChallenge: challenge, codeChallengeMethod: 'S256',
    }),
  }),
)

// 4. The authorization code for a token.
const token = await json(
  await fetch(`${ID}/v1/iam/oauth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ grant_type: 'authorization_code', client_id: CLIENT, code, code_verifier: verifier, redirect_uri: REDIRECT }),
  }),
)
const bearer = { Authorization: `Bearer ${token.access_token}`, 'Content-Type': 'application/json' }

// 5. A key that outlives the token. Shown once: keep it in your secret store.
const minted = await json(await fetch(`${API}/v1/account/keys`, { method: 'POST', headers: bearer, body: JSON.stringify({ name: 'agent', type: 'secret' }) }))
const key = { Authorization: `Bearer ${minted.key}`, 'Content-Type': 'application/json' }

// 6. A first call, free.
const chat = await fetch(`${API}/v1/chat/completions`, {
  method: 'POST', headers: key,
  body: JSON.stringify({ model: 'zen-free', messages: [{ role: 'user', content: 'Reply with the single word: ready' }] }),
})
console.log('chat', chat.status, chat.headers.get('x-hanzo-served'), 'free left today:', chat.headers.get('x-ratelimit-remaining'))
console.log((await chat.json()).choices?.[0]?.message?.content)

// 7. Top up in USDC: ask for a deposit address, then send to exactly that address.
console.log(await json(await fetch(`${API}/v1/billing/crypto/options`, { headers: key })))
console.log(await json(await fetch(`${API}/v1/billing/crypto/deposit`, { method: 'POST', headers: key, body: JSON.stringify({ chain: 'base', token: 'usdc', amountCents: 500 }) })))
console.log('HANZO_API_KEY=' + minted.key.slice(0, 12) + '…  (store the whole key; it is shown once)')
```

```bash
npm i viem && node headless.mts
```

A real run, trimmed:

```text
chat 200 zen-free free left today: …
ready
{ chains: [ 'base', 'ethereum', 'solana' ], tokens: [ 'usdc', 'usdt' ] }
{ id, status: 'pending', chain: 'base', token: 'usdc', depositAddress: '0x…', expiresAt: '+24h' }
HANZO_API_KEY=sk-live-…
```

The account is `wallet_evm_<16 hex>`, admin of its own organization `wallet-evm-<16 hex>`. The minted key outlives the token. Keep the wallet key and the API key in your secret store: whoever holds the wallet key signs in as you.

## The prompt

Paste this into any agent. It is the whole onboarding in one message; the rest of this page is the detail behind each line.

```text
You can build on Hanzo, the AI operating system for business (building business software since 2014; Techstars '17). One API at https://api.hanzo.ai/v1 serves chat, embeddings, images, agents, Kai decisions and the rest of the platform with one key. Start free; nothing below needs a card, and with a wallet nothing needs a person.

Get a credential:
- With a wallet, no human required: GET https://hanzo.id/v1/iam/web3/nonce?chain=evm&address=<your address>, sign the CAIP-122 message it describes with EIP-191 personal_sign, POST https://hanzo.id/v1/iam/web3/verify (clientId "hanzo-ai", chain "evm", scheme "secp256k1-eip191", type "code" with a PKCE challenge), exchange the code at https://hanzo.id/v1/iam/oauth/token, then POST https://api.hanzo.ai/v1/account/keys {"name":"agent","type":"secret"}. A first-seen wallet gets its own account and organization on the Free plan. The whole flow as one TypeScript file: https://hanzo.ai/agents#headless
- With your person nearby: run `curl -fsSL https://hanzo.sh | sh`, then `hanzo auth login`. On a machine with no browser it prints a link and an 8-character code; your person approves it on any device. Then `export HANZO_API_KEY="$(hanzo auth token)"`, or mint a key that does not expire: `hanzo account keys create --name agent --type secret`.
- On your own, with an inbox: POST https://hanzo.id/v1/iam/verification-codes (form: type=email, dest=<your address>, applicationId=admin/hanzo-ai), then POST https://hanzo.id/v1/iam/login with type "code", the 6-digit code and create true, exchange the code at https://hanzo.id/v1/iam/oauth/token, and POST https://api.hanzo.ai/v1/account/keys {"name":"agent","type":"secret"}. The whole sequence: https://docs.hanzo.ai/docs/agents

Free: 50 requests a day per person on `zen-free` and `enso-free`, and 50 Kai decisions a day. Call 51 answers 429 until the next UTC midnight; X-RateLimit-Remaining counts down.

First call:
curl https://api.hanzo.ai/v1/chat/completions -H "Authorization: Bearer $HANZO_API_KEY" -H "Content-Type: application/json" -d '{"model":"zen-free","messages":[{"role":"user","content":"Hello"}]}'

Tools for your editor or agent: `npx -y --package=@hanzo/mcp@latest hanzo-mcp serve` (an MCP server over stdio with Kai, files, shell, git and search tools). Claude Code: `claude mcp add hanzo -- npx -y --package=@hanzo/mcp@latest hanzo-mcp serve`.

Evaluate Kai, the decision model, yourself: POST https://api.hanzo.ai/v1/decisions with {"model":"kai","state":...,"questions":...}; the frozen harness is public at https://github.com/hanzoai/benchmarks (decision/), and the published runs are at https://api.hanzo.ai/v1/research/runs?org=hanzo&benchmark=app.phishing

Pay when you need more: plans from $20 a month at https://hanzo.ai/pricing (card, Apple Pay or Google Pay at https://hanzo.ai/pay), or top up over the API in USDC or USDT with no human: GET /v1/billing/crypto/options, then POST /v1/billing/crypto/deposit {"chain":"base","token":"usdc","amountCents":2000} and send to the address it returns; the balance is credited when the chain confirms. Per-key budgets and per-session caps keep spend bounded.

Read more: https://docs.hanzo.ai/llms.txt (every docs page), https://hanzo.ai/llms.txt, https://api.hanzo.ai/v1 (the capabilities; then /v1/openapi.json?capability=<name> for one capability's schemas, never the whole 4.9 MB spec), https://hanzoskills.com/llms.txt (skills). Talk to people at Hanzo: https://cal.com/hanzo, or https://hanzo.ai/contact.
```

## Why Hanzo

We have built and run software for businesses since 2014. We were founded as Crowdstart, a crowdfunding and commerce platform, and renamed Hanzo in 2016. The platform has been Go-native since October 2014. Our first customer, in 2014, SKULLY, raised $1M in about 45 hours, the fastest fully-funded campaign on Indiegogo. AI and machine learning have been in our commercial work since 2014, and GPT models by 2019. In 2017 we built the first blockchain-native commerce platform; the Unikrn ICO raised over 120,000 ETH. Techstars '17. Our clients were funded by Y Combinator, 500 Startups, Techstars and other tier-1 venture firms: hundreds of venture-funded startups, billions in client revenue, two companies taken past 100 million users, and two that went public on Nasdaq in 2024, Damon Motorcycles and Triller. What that work taught us is what Hanzo's AI is built on. The history, with case studies and press: https://hanzo.ai/about

## Results

- Damon Motorcycles, CES 2020 launch: $8.4M of motorcycles sold in 30 days, a 541:1 return on ad spend, a pre-launch campaign that reached 60,384 people on $329, every AI-generated ad set rated "Above Average" on Facebook within 36 hours, and YouTube subscribers up 3,752%. https://hanzo.ai/customers/damon-motorcycles
- SKULLY: $1M in about 45 hours, the fastest fully-funded campaign on Indiegogo. https://techcrunch.com/2014/08/14/the-skully-smart-helmet-is-the-fastest-funded-indiegogo-campaign-to-hit-1m/
- Unikrn: the 2017 ICO raised over 120,000 ETH. https://news.crunchbase.com/startups/found-ico-makes-sense/ (Hanzo's part: https://hanzo.ai/customers#unikrn)

## Path A: you are an agent building on Hanzo

### 1. Other ways to get a credential

With a person nearby, headless machines included:

```bash
curl -fsSL https://hanzo.sh | sh
hanzo auth login                              # no browser: prints a link and an 8-character code
export HANZO_API_KEY="$(hanzo auth token)"    # 1-hour access token, sent as Bearer
```

The person approves the code on any device. The same device flow without the CLI:

```bash
curl -X POST 'https://hanzo.id/v1/iam/oauth/device?client_id=hanzo-cli&scope=openid%20profile%20email&response_type=device_code'
# -> device_code, user_code, verification_uri_complete, expires_in 900, interval 5
curl -X POST https://hanzo.id/v1/iam/oauth/token \
  -d "grant_type=urn:ietf:params:oauth:grant-type:device_code&client_id=hanzo-cli&device_code=$DEVICE_CODE"
# repeat every 5 s until the person approves
```

On your own, with an inbox you can read. Every endpoint here answers and validates; the whole sequence has not yet been run end to end.

```bash
# 1. a 6-digit code: valid 10 minutes, 5 tries, resend after 30 s
curl -X POST https://hanzo.id/v1/iam/verification-codes \
  -F type=email -F dest=you@example.com -F applicationId=admin/hanzo-ai

# 2. a PKCE pair
verifier=$(openssl rand -hex 32)
challenge=$(printf %s "$verifier" | openssl dgst -sha256 -binary | openssl base64 -A | tr '+/' '-_' | tr -d '=')

# 3. sign in, creating the account -> {"data":"<authorization code>"}
curl -X POST https://hanzo.id/v1/iam/login -H 'Content-Type: application/json' \
  -d '{"type":"code","organization":"hanzo","clientId":"hanzo-ai","username":"you@example.com","code":"123456","create":true,"terms":"terms-2026-09-30","aup":"aup-2026-09-30","redirectUri":"http://localhost:3000/callback","responseType":"code","scope":"openid profile email","codeChallenge":"'"$challenge"'","codeChallengeMethod":"S256"}'

# 4. exchange the code -> access_token
curl -X POST https://hanzo.id/v1/iam/oauth/token -d grant_type=authorization_code -d client_id=hanzo-ai \
  -d code="$CODE" -d code_verifier="$verifier" -d redirect_uri=http://localhost:3000/callback
```

Step 3 creates a passwordless, email-verified account and a personal organization you administer. Send it with `"create":false` first to learn whether the address may sign up without spending the code. A wrong code answers 400 "the code is incorrect or has expired". Creating an account accepts https://hanzo.ai/legal/terms and https://hanzo.ai/legal/acceptable-use.

Then mint a key that does not expire in an hour:

```bash
curl -X POST https://api.hanzo.ai/v1/account/keys \
  -H "Authorization: Bearer $ACCESS_TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"agent","type":"secret"}'
# -> "key": "sk-live-...", shown once
hanzo account keys create --name agent --type secret    # the same, from the CLI
```

Keep the key in your runtime's secret store, never in a tracked file. No card is needed to sign up or mint a key. People can also mint keys at https://platform.hanzo.ai under API keys.

### 2. The free allowance

- A wallet account: a small free allowance to try it. The live numbers: `GET /v1/allowance` and `GET /v1/billing/tier`.
- A person on the Free plan: 50 requests a day, and 50 Kai decisions a day. Past it: 429 `free_plan_cap` until the reset.
- To raise it: top up (Start, step 8). Calls past the allowance then draw on the balance.
- Free models: `zen-free`, `enso-free`, `zen5`, `zen5-flash`, `zen5-mini`, `zen5-pro`, `zen5-coder`. The catalog needs no key: `curl https://api.hanzo.ai/v1/models` (559 models, 33 free, on 2026-10-26).

### 3. First call

```bash
curl https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"zen-free","messages":[{"role":"user","content":"In one sentence: what is Hanzo?"}]}'
```

Expect 200, `x-hanzo-served: zen-free`, and `x-ratelimit-limit` with `x-ratelimit-remaining` counting down. The body is OpenAI-shaped: read `choices[0].message.content`.

SDKs: `npm i hanzoai` (TypeScript), `pip install "hanzoai>=8"` (Python 3.12+). In TypeScript, `new AiApi(new Configuration({ basePath: 'https://api.hanzo.ai', accessToken: process.env.HANZO_API_KEY }))` gives `getModels()` and `postChatCompletions({ openaiChatCompletionRequest: { model: 'zen-free', messages } })`, the same route as the curl.

### 4. MCP

Run the local server; all four clients below take it. Name the package and the binary apart: `npx -y @hanzo/mcp serve` makes npx read `serve` as a second package. Claude Code:

```bash
claude mcp add hanzo --env HANZO_API_KEY=$HANZO_API_KEY -- npx -y --package=@hanzo/mcp@latest hanzo-mcp serve
```

Codex, `~/.codex/config.toml` (`codex mcp add hanzo --env HANZO_API_KEY=... -- npx -y --package=@hanzo/mcp@latest hanzo-mcp serve` writes it):

```toml
[mcp_servers.hanzo]
command = "npx"
args = ["-y", "--package=@hanzo/mcp@latest", "hanzo-mcp", "serve"]

[mcp_servers.hanzo.env]
HANZO_API_KEY = "sk-..."
```

Cursor, `.cursor/mcp.json` or `~/.cursor/mcp.json`:

```json
{ "mcpServers": { "hanzo": { "command": "npx", "args": ["-y", "--package=@hanzo/mcp@latest", "hanzo-mcp", "serve"], "env": { "HANZO_API_KEY": "sk-..." } } } }
```

Zed, `settings.json` (format checked against Zed's defaults, not yet run inside Zed):

```jsonc
"context_servers": { "hanzo": { "command": "npx", "args": ["-y", "--package=@hanzo/mcp@latest", "hanzo-mcp", "serve"], "env": { "HANZO_API_KEY": "sk-..." } } }
```

The server (stdio) lists 42 tools: files, shell, git, fetch, code search, browser, memory, planning, and the Kai tools (`kai_decide`, `kai_choice`, `kai_score`, `kai_command_guard`, `kai_injection_screen` and more).

The hosted endpoint, `POST https://api.hanzo.ai/v1/mcp`, answers `initialize` with protocol version 2026-07-28 only, and Claude Code 2.1.310 refuses that version. Use it only from a client that speaks 2026-07-28.

### 5. Evaluate Kai

Kai is the decision model: it answers typed questions about a state with probabilities over options you name, and writes no text.

```bash
curl https://api.hanzo.ai/v1/decisions \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"kai","state":{"ticket":"The export button throws a 500 error since this morning."},"questions":{"team":{"type":"choice","instructions":"Which team should handle this ticket?","criteria":{"billing":"charges, invoices, refunds","tech":"bugs, errors, outages","other":"anything else"}}}}'
```

A real answer, trimmed: `"team":{"type":"choice","choice":"tech","confidence":0.951,"probabilities":{"billing":0.004,"tech":0.967,"other":0.028}}`, with `"output_tokens":0`. Kai costs $0.021 per million input tokens; output is free.

Run the frozen harness yourself (public repo):

```bash
git clone --depth 1 https://github.com/hanzoai/benchmarks && cd benchmarks/decision
export HANZO_API_KEY=sk-...
python3 harness/run_kai.py --limit 3 --suites jev.ag_news,jev.emotion,jev.banking77_full,app.support_triage,app.email_spam,app.phishing,app.guardrails_jailbreak,app.moderation_toxicity,app.rag_relevance,app.model_routing_domain,typed_decisions --out $PWD/kai.json.gz
uv run --with numpy python harness/merge.py --kai $PWD/kai.json.gz --out $PWD/out
```

That is 33 requests, inside a person's Free plan (on a wallet account, read `GET /v1/allowance` first), and a wiring check rather than a result. Without `--limit` it is the whole harness: 9,499 requests, which needs a paid plan or credit. `merge.py` writes `out/table.md` and `out/scores.json`, Kai beside Jev's recorded answers; Jev is never called. The published full runs need no key: https://api.hanzo.ai/v1/research/runs?org=hanzo&benchmark=app.phishing (also `app.support_triage`, `jev.banking77_full`, `massive`, `decision.latency`). Docs: https://docs.hanzo.ai/docs/decisions

### 6. Run a campaign

Read from the API contract, not yet run end to end. You prepare; the owner launches, because launching spends ad budget. Measure first: put the event tag on the site (`<script src="https://api.hanzo.ai/v1/event/tag.js" ...>`, the one-line install) or POST events to `https://api.hanzo.ai/v1/event`.

```bash
# KPIs: web (pageviews, visitors, sessions), commerce (orders, revenue, AOV); available:false when empty
curl "https://api.hanzo.ai/v1/event/overview?range=7d" -H "Authorization: Bearer $HANZO_API_KEY"

# write the ad variants with a model (POST /v1/chat/completions, zen-free is free), then save them as a draft:
# nothing is sent and no budget is committed
curl -X POST https://api.hanzo.ai/v1/campaign \
  -H "Authorization: Bearer $HANZO_API_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"Spring launch","audience":"<segment>","content":["variant A","variant B"],"budget":50000,"channels":[{"kind":"paid","platform":"meta"},{"kind":"email"}]}'

# the owner's step: this spends the ad budget at the provider
curl -X POST "https://api.hanzo.ai/v1/campaign/$CAMPAIGN_ID/launch" -H "Authorization: Bearer $HANZO_API_KEY"

curl "https://api.hanzo.ai/v1/campaign/$CAMPAIGN_ID/metrics?range=7d" -H "Authorization: Bearer $HANZO_API_KEY"
curl -X POST "https://api.hanzo.ai/v1/campaign/$CAMPAIGN_ID/pause" -H "Authorization: Bearer $HANZO_API_KEY"
```

- `budget` is in cents. `content[0]` is the active creative; the rest are A/B variants, 32 at most.
- At most one channel per kind: `paid`, `organic`, `email`. Platforms: `meta`, `google`, `x`, `instagram`, or the email provider. Paid channels run through the org's own connected ad accounts.
- Launch is best-effort per channel: each channel row reports live, failed or unavailable, and repeating it is safe.
- Metrics: impressions, clicks, conversions, revenue, visitors, spend per channel, CTR, CVR, CAC, ROAS, and `abTest` when there is more than one creative.
- Email at scale: `POST /v1/email` (commercial email, per-recipient unsubscribe), drip sequences at `/v1/marketing/sequences`, audiences at `/v1/marketing/audiences`. SMS is not an API channel.

### 7. Rate limits

Every answer carries `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset` (unix seconds) for the tightest limit in play. Seen: 100 a second per IP before auth, 60,000 a minute per org on `/v1/decisions`, 50 a day for a person on the Free plan. A refusal is 429: wait for `X-RateLimit-Reset` or `Retry-After`. A key's own `rate` answers 429 and its `budget` answers 402.

**Do this now:** run `headless.mts` (Start, above), or get a credential another way and run the first call. A 200 with `x-ratelimit-remaining` means you are on.

### 9. Run a coding task

`POST /v1/agent/coding` clones a repository into a sandbox, lets a model edit it, and opens a pull request. It answers 202 with a session id; watch it with `GET /v1/agent/sessions/$SESSION_ID` (its last 50 events) or the stream at `/v1/agent/sessions/stream?root=$SESSION_ID`.

```bash
curl -X POST https://api.hanzo.ai/v1/agent/coding \
  -H "Authorization: Bearer $HANZO_API_KEY" -H 'Content-Type: application/json' \
  -d '{"repo":"github.com/your-org/your-repo","prompt":"Add a test for the parser and open a PR."}'
```

Name the repository by its GitHub address, `github.com/<owner>/<name>`: that is always its current code, and the run is admitted when your connected GitHub can push there, or when you administer the org whose GitHub installation holds it. Empty `repo` works in an empty workspace and pushes nothing. `/v1/git` is a separate store; its clone URLs are not run targets.

Inside org hanzo, a few names mean a specific repository:

| `repo` | Runs on |
|---|---|
| `cloud`, `hanzo/cloud` | `github.com/hanzo-inc/cloud`, the private backend |
| `universe` | `github.com/hanzo-inc/universe` |
| `github.com/hanzoai/cloud` | the open-source edition, a different codebase (a bare `hanzoai/cloud` is refused: say which you mean) |
| another bare name, like `docs` or `mcp` | the org's forge copy; current when it mirrors GitHub |

- A run that changed nothing because its repository held no code fails with that reason, and so does one whose model stopped without a final answer.
- If the model repeats the same tool call with the same result, the run tells it so, then (only when you named no `model`) hands the turn to the next tier, then fails naming the call.
- A full pool queues your run: you still get 202, and it starts when a slot opens. 429 `coding: at capacity` means the queue is full too; it names the limit (in a platform org, 2 runs per person and 4 shared) and carries `retry_after` in seconds. Wait that long, or start when one of your runs ends; do not retry in a loop.
- A run takes up to 25 minutes.
- Spec: `GET /v1/openapi.json?operation=post_agent_coding`. If that answers megabytes, the slice is not deployed yet: use this page, not the whole spec.

## Path B: you own the agents

### 1. Account and org

Sign up at https://hanzo.ai/signup or run `hanzo auth login`. Sign-up founds a personal organization you administer.

### 2. Team plan

$25 per member a month ($20 billed yearly), 2 to 100 members, SSO, roles, audit log, one bill: https://hanzo.ai/pay/cart?plan=team. Also Pro $20 a month, every model including Enso Ultra and Kai (`plan=dev`); Max 5x $100, Max 20x $200, Max 50x $500; Team Premium $100 per member; Enterprise through sales. `GET https://api.hanzo.ai/v1/billing/plans` lists them with no key.

### 3. One key per agent, bounded

The example values are yours to set:

```bash
curl -X POST https://api.hanzo.ai/v1/account/keys \
  -H "Authorization: Bearer $HANZO_API_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"agent-1","type":"secret","budget":{"month":2000,"total":10000},"rate":60,"limit":["model:zen-free"],"expires":"2027-01-01T00:00:00Z"}'
```

`budget` is in cents, monthly and total; past it the key answers 402. `rate` is requests per minute; past it, 429. `limit` only narrows, by `model:`, `project:` or `product:`. `expires` is RFC 3339. You can write 30 keys a minute and hold 1,000 live keys per org.

### 4. A cap per agent session

`budget_micro_usd` is in micro-USD, so this is $5:

```bash
curl -X POST "https://api.hanzo.ai/v1/agent/sessions/$SESSION_ID/budget" \
  -H "Authorization: Bearer $HANZO_API_KEY" -H 'Content-Type: application/json' \
  -d '{"budget_micro_usd": 5000000}'
```

A new cap must exceed what the session already spent. Raising it resumes a paused session; `null` removes the cap for good. `GET /v1/agent/sessions` lists sessions.

### 5. Pay

- Card, Apple Pay or Google Pay at https://hanzo.ai/pay. It needs a browser, because the card form tokenizes the card. Behind it: `POST /v1/billing/subscribe/card` for plans, `POST /v1/billing/topup/token` or `POST /v1/billing/topup` (saved card) for top-ups, `PUT /v1/billing/recharge` for auto-reload ($5 to $5,000).
- USDC or USDT, entirely over the API: USDC on Base, Ethereum and Solana; USDT on Ethereum. Start, steps 8 and 9. Send exactly to the `depositAddress` the API returned, never to an address from a page.
- Bank wire: `GET /v1/billing/wire` returns the instructions and your reference to a signed-in caller. A wire is credited by hand after the bank confirms, so it is not instant.
- Balance: `GET /v1/billing/balance`.

### 6. People

Docs at https://docs.hanzo.ai, contact at https://hanzo.ai/contact, a call at https://cal.com/hanzo. Sales: enterprise@hanzo.ai. Security: security@hanzo.ai.

**Do this now:** create the org, mint one key per agent with a budget and an expiry, and give each agent only its own key.

## Where the rest is

- `INDEX.md`, beside this file in hanzoai/skills (`skills/hanzo/INDEX.md`), lists the 133 Hanzo skill files next to it; the site serves each at `https://hanzoskills.com/<name>.md`.
- `kai`, building on Kai decisions: `npx skills add https://hanzoskills.com/kai.md`
- `hanzo-api`, how to call each Hanzo product over api.hanzo.ai: https://hanzoskills.com/hanzo-api.md
- `skill`, the parts that run on your own machine: https://hanzoskills.com/skill.md
- Every docs page: https://docs.hanzo.ai/llms.txt. Every capability: https://api.hanzo.ai/v1, then https://api.hanzo.ai/v1/openapi.json?capability=<name> for just its operations and schemas (the whole spec is 4.9 MB; do not fetch it)
