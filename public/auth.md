# Hanzo auth.md

For an AI agent that needs a credential for the Hanzo API. The resource server is
`https://api.hanzo.ai` (REST under `/v1`, MCP at `/v1/mcp`). The authorization server
is `https://hanzo.id` (OpenID Connect). Every step below is one HTTPS request.

## Discover

1. `GET https://api.hanzo.ai/.well-known/oauth-protected-resource` (RFC 9728) answers
   `resource: https://api.hanzo.ai`, `authorization_servers: ["https://hanzo.id"]`,
   `bearer_methods_supported: ["header"]`.
2. `GET https://hanzo.id/.well-known/openid-configuration` answers `issuer: https://hanzo.id`,
   `token_endpoint: https://hanzo.id/v1/iam/oauth/token`, PKCE `S256`.

A call with no credential answers `401` with `WWW-Authenticate: Bearer`.

## Pick a method

- No person available: sign in with a wallet.
- A person can approve on any device: the device flow.
- A person already holds a key: they mint one at https://platform.hanzo.ai under API keys.

## Wallet sign-in, no person

1. `GET https://hanzo.id/v1/iam/web3/nonce?chain=evm&address=0x…` returns a challenge
   (`domain`, `uri`, `statement`, `nonce`, `issuedAt`, `expirationTime`, `version`). It lives
   10 minutes.
2. Build the CAIP-122 message from it and sign with EIP-191 `personal_sign`.
3. `POST https://hanzo.id/v1/iam/web3/verify` with
   `{"clientId":"hanzo-ai","chain":"evm","scheme":"secp256k1-eip191","address":"0x…","message":"…","signature":"0x…","type":"code","redirectUri":"http://localhost:3000/callback","scope":"openid profile email","codeChallenge":"<S256>","codeChallengeMethod":"S256"}`
   returns an authorization code. A first-seen wallet gets its own account and organization
   on the Free plan.
4. `POST https://hanzo.id/v1/iam/oauth/token`, form `grant_type=authorization_code`,
   `client_id=hanzo-ai`, `code`, `code_verifier`, `redirect_uri`, returns a Bearer
   `access_token` valid seven days (`expires_in` 604800).

The whole flow as one TypeScript file, run end to end: https://hanzoskills.com/hanzo.md

## Device flow, a person approves

1. `POST https://hanzo.id/v1/iam/oauth/device?client_id=hanzo-cli&scope=openid%20profile%20email&response_type=device_code`
   returns `device_code`, `user_code`, `verification_uri_complete`, `expires_in` 900, `interval` 5.
2. Show the person `verification_uri_complete`. Every 5 seconds,
   `POST https://hanzo.id/v1/iam/oauth/token` with
   `grant_type=urn:ietf:params:oauth:grant-type:device_code`, `client_id=hanzo-cli`,
   `device_code`, until it returns a token.

The CLI does the same: `curl -fsSL https://hanzo.sh | sh`, then `hanzo auth login`.

## Mint a key

`POST https://api.hanzo.ai/v1/account/keys` with the access token and
`{"name":"agent","type":"secret"}` returns `key` (`sk-live-…`), shown once. It outlives the
token. Keep it in a secret store, never in a tracked file. No card is needed.

## Use it

Send `Authorization: Bearer <key or access token>` to `https://api.hanzo.ai/v1/…` and to the
MCP endpoint `https://api.hanzo.ai/v1/mcp` (protocol revision 2026-07-28).

## Revoke

`GET https://api.hanzo.ai/v1/account/keys` lists your keys; `DELETE https://api.hanzo.ai/v1/account/keys/{id}`
revokes one; a request inside the next 60 seconds may still be served. Whoever holds the wallet key can sign in as that account: guard it like the API key.

## Not offered

The auth.md agent-registration protocol (`POST /agent/identity`, ID-JAG identity assertions, the
claim ceremony, an `agent_auth` block in authorization server metadata) is not implemented here.
Use one of the methods above.
