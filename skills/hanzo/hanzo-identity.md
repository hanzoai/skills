# Hanzo Identity - Decentralized Identity (DID) System

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-iam.md`, `hanzo/hanzo-contracts.md`

## Overview

Hanzo Identity (`hanzoai/identity`) is a **decentralized identity (DID) registration system** with Solidity smart contracts and a Next.js frontend. It implements W3C DID identifiers across the Lux/Hanzo/Zoo/Pars chains, with an NFT bound to DID ownership. Core DID contracts live in `@luxfi/standard`; this repo provides the registration UI, deployment scripts, and Foundry build tooling.

### What it actually is

- Solidity smart contracts compiled with Foundry (solc 0.8.31, via_ir, optimizer)
- Core DID contracts imported from `@luxfi/standard/contracts/did/` (Registry.sol, IdentityNFT.sol)
- Next.js 16 frontend (`app/`) with wallet integration (RainbowKit, wagmi, viem)
- Static file server (`server.js`) on port 3100 for contract UI testing
- Deployment scripts in both JavaScript (ethers.js v5) and Solidity (Forge scripting)
- Foundry/Forge for contract compilation, testing, and deployment
- Anvil for local chain testing (port 8545, chain 31337)
- Playwright for end-to-end testing
- Deployed to GitHub Pages
- Includes Hanzo AI soul/values document (`soul.md`)

### DID format

```
did:lux:alice      # W3C canonical format
alice@lux.id       # Display format
alice.lux.id       # Web subdomain
```

### Supported chains

| Chain | ID | DID Method | Display |
|-------|-----|------------|---------|
| Lux | 96369 | `did:lux:` | `@lux.id` |
| Pars | 494949 | `did:pars:` | `@pars.id` |
| Zoo | 200200 | `did:zoo:` | `@zoo.id` |
| Hanzo | 36963 | `did:hanzo:` | `@hanzo.id` |

## When to use

- Registering or managing decentralized identities on Lux/Hanzo chains
- Deploying DID registry contracts to new chains
- Building identity management UIs
- Integrating DID resolution into services
- Testing contract deployments on local Anvil chains

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/identity` |
| Branch | `main` |
| Contracts package | `identity-contracts` v1.0.0 |
| Frontend package | `@hanzo/identity-dapp` v1.0.0 |
| Solidity | 0.8.31 (Foundry) |
| Core DID contracts | `@luxfi/standard/contracts/did/` |
| Frontend | Next.js 16, React 19 |
| Wallet | RainbowKit 2, wagmi 2, viem 2 |
| UI library | `@hanzo/ui` |
| Local chain | Anvil (port 8545, chain 31337) |
| Static server | port 3100 |
| Frontend dev | `cd app && pnpm dev` (port 3000) |
| Test | `node test-all.js` (contracts), `npx playwright test` (e2e) |
| Deploy | GitHub Pages (auto on push to main) |
| License | Proprietary (Hanzo AI Inc.) |

## Project structure

```
hanzoai/identity/
  package.json              # identity-contracts 1.0.0
  foundry.toml              # Foundry config (solc 0.8.31, via_ir)
  server.js                 # Static file server (port 3100)
  index.html                # Contract interaction UI
  test-all.js               # Contract test suite
  soul.md                   # Hanzo AI identity and values
  playwright.config.js      # E2E test config
  script/
    DeployLocal.s.sol        # Forge deployment script (local)
    deploy.sol               # Forge deployment script (production)
  scripts/
    deploy.js                # ethers.js deployment (generic)
    deploy-local.js          # ethers.js deployment (Anvil)
    deploy-final.js          # ethers.js deployment (mainnet)
    deploy-with-cast.sh      # cast-based deployment (shell)
    deploy-with-proxy.js     # Proxy deployment (upgradeable)
    register-test-identities.js  # Register test DIDs
    run-all-tests.sh         # Full test runner
    run-with-ui.sh           # Start Anvil + UI together
  app/                       # Next.js frontend
    package.json             # @hanzo/identity-dapp 1.0.0
    next.config.js
    src/
      app/                   # Next.js app router pages
      components/            # React components
      lib/                   # Contract interaction helpers
      types/                 # TypeScript types
    public/                  # Static assets
  docs/                      # Documentation
  tests/                     # Playwright e2e tests
```

## Dependencies

### Contracts (root `package.json`)
- `@openzeppelin/contracts` ^4.9.0 -- ERC standards, access control
- `@openzeppelin/contracts-upgradeable` ^4.9.0 -- proxy patterns
- `ethers` ^5.8.0 -- contract deployment and interaction

### Frontend (`app/package.json`)
- `next` ^16.0.0, `react` ^19.0.0
- `@hanzo/ui` latest -- shared UI components
- `viem` ^2.0.0, `wagmi` ^2.0.0, `@rainbow-me/rainbowkit` ^2.0.0 -- Web3
- `@tanstack/react-query` ^5.0.0 -- data fetching
- `tailwindcss` ^4.0.0, `clsx`, `tailwind-merge` -- styling

## Quickstart

### Local development (contracts)

```bash
git clone https://github.com/hanzoai/identity.git
cd identity
npm install

# Start Anvil local chain + deploy contracts
npm run dev
# Anvil on port 8545, static server on port 3100

# Or separately:
npm run anvil                          # Local chain
node scripts/deploy-local.js           # Deploy contracts
npm run serve                          # Static UI on 3100
```

### Frontend development

```bash
cd app
pnpm install
cp .env.example .env.local
# Edit .env.local with contract addresses and RPC URL
pnpm dev
# Open http://localhost:3000
```

### Testing

```bash
# Contract tests
npm test                    # node test-all.js

# E2E tests
npm run test:e2e            # Playwright headless
npm run test:e2e:headed     # Playwright with browser

# Full suite
npm run test:all
bash scripts/run-all-tests.sh
```

### Deployment

```bash
# Forge (Solidity scripting)
forge script script/deploy.sol --rpc-url $RPC_URL --broadcast

# cast (shell)
bash scripts/deploy-with-cast.sh

# ethers.js
node scripts/deploy.js

# Upgradeable proxy
node scripts/deploy-with-proxy.js
```

## Foundry configuration

```toml
[profile.default]
src = "contracts/src"
out = "artifacts"
libs = ["node_modules"]
solc = "0.8.31"
via_ir = true
optimizer = true
optimizer_runs = 200
remappings = [
    "@luxfi/standard/=node_modules/@luxfi/standard/contracts/",
    "@openzeppelin/contracts/=node_modules/@openzeppelin/contracts/",
    "@openzeppelin/contracts-upgradeable/=node_modules/@openzeppelin/contracts-upgradeable/"
]
```

## Troubleshooting

- **Forge build fails**: Run `npm install` first -- contracts import from `node_modules/@luxfi/standard`
- **Anvil won't start**: Check port 8545 is free (`lsof -i :8545`)
- **Contract deployment fails**: Ensure Anvil is running and funded accounts available
- **Frontend wallet connection fails**: Configure correct chain ID in app env
- **Missing contract artifacts**: Run `forge build` before deployment scripts

## Related Skills

- `hanzo/hanzo-iam.md` -- Centralized IAM (hanzo.id, Casdoor-based), complementary to DID
- `hanzo/hanzo-id.md` -- Login UI for Hanzo IAM
- `hanzo/hanzo-contracts.md` -- Smart contract patterns
- `hanzo/hanzo-web3.md` -- Web3 integration

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: did, identity, solidity, foundry, web3, nft, lux
**Prerequisites**: Node.js, Foundry (forge/anvil), pnpm (for app/)
