# auth.md

You are an agent that needs a credential for the Hanzo API. Registration here is wallet sign-in,
or a device code a person approves: discover, register, exchange the code for an access_token,
mint an API key, call the API, revoke. The resource server is https://api.hanzo.ai (REST under
`/v1`, MCP at `/v1/mcp`). The authorization server is https://hanzo.id (OpenID Connect).
Every step is one HTTPS request.

## Summary

- Audience: AI agents, acting for themselves or for a person.
- Registration endpoint: POST https://hanzo.id/v1/iam/web3/verify registers a first-seen wallet
  (its own account and organization) and returns an authorization code.
- Supported methods: wallet sign-in (a CAIP-122 message signed with EIP-191) and device
  authorization (RFC 8628, a person approves).
- Token endpoint: POST https://hanzo.id/v1/iam/oauth/token returns the access_token.
- Credential: the Bearer access_token (seven days), or an API key from
  POST https://api.hanzo.ai/v1/account/keys, sent as `Authorization: Bearer`.

## Step 1: Discover

1. GET https://api.hanzo.ai/.well-known/oauth-protected-resource (RFC 9728) answers
   `resource: https://api.hanzo.ai`, `authorization_servers: ["https://hanzo.id"]`,
   `bearer_methods_supported: ["header"]`.
2. GET https://hanzo.id/.well-known/openid-configuration answers `issuer: https://hanzo.id`,
   `token_endpoint: https://hanzo.id/v1/iam/oauth/token`, PKCE `S256`.

## Step 2: Pick a method

- No person available: register with a wallet.
- A person can approve on any device: the device flow.
- A person already holds a key: they mint one at https://platform.hanzo.ai under API keys.

## Step 3: Register

### Wallet, no person

1. GET https://hanzo.id/v1/iam/web3/nonce?chain=evm&address=0x… returns a challenge
   (`domain`, `uri`, `statement`, `nonce`, `issuedAt`, `expirationTime`, `version`). It lives
   10 minutes.
2. Build the CAIP-122 message from it and sign it with EIP-191 `personal_sign`.
3. POST https://hanzo.id/v1/iam/web3/verify with
   `{"clientId":"hanzo-ai","chain":"evm","scheme":"secp256k1-eip191","address":"0x…","message":"…","signature":"0x…","type":"code","redirectUri":"http://localhost:3000/callback","scope":"openid profile email","codeChallenge":"<S256>","codeChallengeMethod":"S256"}`
   returns an authorization code. A first-seen wallet is registered: it gets its own account
   and organization, on the Free plan.

The whole flow as one TypeScript file, run end to end: https://hanzoskills.com/hanzo.md

### Device, a person approves

1. POST https://hanzo.id/v1/iam/oauth/device?client_id=hanzo-cli&scope=openid%20profile%20email&response_type=device_code
   returns `device_code`, `user_code`, `verification_uri_complete`, `expires_in` 900, `interval` 5.
2. Show the person `verification_uri_complete`; they approve on any device.

The CLI does the same: `curl -fsSL https://hanzo.sh | sh`, then `hanzo auth login`.

## Step 4: Exchange for an access_token

POST https://hanzo.id/v1/iam/oauth/token (form encoded):

- Wallet: `grant_type=authorization_code`, `client_id=hanzo-ai`, `code`, `code_verifier`,
  `redirect_uri`. The Bearer `access_token` is valid seven days (`expires_in` 604800).
- Device: `grant_type=urn:ietf:params:oauth:grant-type:device_code`, `client_id=hanzo-cli`,
  `device_code`, every 5 seconds until the person approves.

## Step 5: Mint an API key

POST https://api.hanzo.ai/v1/account/keys with the access_token and
`{"name":"agent","type":"secret"}` returns `key` (`sk-live-…`), shown once. It outlives the
access_token. Keep it in a secret store, never in a tracked file. No card is needed.

## Step 6: Use the access_token or key

Send `Authorization: Bearer <key or access_token>` to the REST API under
https://api.hanzo.ai/v1 and to the MCP endpoint https://api.hanzo.ai/v1/mcp (protocol
revision 2026-07-28).

## Revocation

GET https://api.hanzo.ai/v1/account/keys lists your keys; DELETE
`https://api.hanzo.ai/v1/account/keys/{id}` revokes one, and a request inside the next 60 seconds
may still be served. Whoever holds the wallet key can sign in as that account: guard it like the
API key.

## Not implemented

The auth.md agent-registration protocol is not implemented here: there is no agent identity
endpoint, no identity assertion exchange and no claim ceremony. Use one of the methods above.
