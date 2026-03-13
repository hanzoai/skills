# Hanzo MPC - Multi-Party Computation for Distributed Wallet Management

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-kms.md`, `hanzo/hanzo-vault.md`, `hanzo/hanzo-web3.md`

## Overview

Hanzo MPC is a **threshold signing service** for securely generating and managing cryptographic wallets across distributed nodes without ever exposing the full private key. Written in Go, it implements ECDSA (secp256k1) via CGGMP21 and EdDSA (Ed25519) via FROST protocols. Supports Bitcoin, Ethereum, Lux, XRPL, Solana, TON, and all EVM-compatible chains. Pluggable signer backend for Hanzo KMS. Production-deployed as a 2-of-3 StatefulSet on hanzo-k8s at `mpc.hanzo.ai`.

### Why Hanzo MPC?

- **No single point of compromise**: Private keys are never fully assembled
- **Threshold signing**: Only t-of-n nodes needed to sign (default: 2-of-3)
- **Multi-chain**: ECDSA (Bitcoin, Ethereum, EVM, XRPL, Lux) + EdDSA (Solana, TON)
- **Policy engine**: Fireblocks-style transaction governance (signers, limits, whitelists)
- **FHE integration**: Threshold FHE via luxfi/fhe for encrypted policy evaluation
- **ConsensusKV**: Private BFT blockchain with dual-certificate finality (Ed25519 + ML-DSA-65 post-quantum)
- **Bridge compatible**: Drop-in replacement for Rust-based MPC via HTTP API on port 6000

### Tech Stack

- **Language**: Go 1.26
- **Crypto**: `luxfi/threshold` (CGGMP21 + FROST), `luxfi/crypto`, `luxfi/fhe`
- **Messaging**: NATS (JetStream + pub/sub + P2P)
- **Storage**: BadgerDB (AES-256 encrypted key shares)
- **Discovery**: Consul (legacy), NATS KV, or ConsensusKV
- **Identity**: Ed25519 node identity keys (age-encrypted)
- **HSM**: Pluggable (file, AWS KMS, GCP KMS, Azure HSM, Zymbit, Hanzo KMS)
- **Smart Contracts**: ThresholdPolicy.sol (on-chain policy enforcement)
- **Build**: Make + Go

### OSS Base

Repo: `hanzoai/mpc`. Module: `github.com/hanzoai/mpc`.

## When to use

- Generating distributed crypto wallets without exposing private keys
- Threshold signing for multi-chain transactions
- Building custody solutions with policy-based governance
- Integrating with Hanzo KMS as a signing backend
- Bridge migration from Rust-based MPC to Go

## Hard requirements

1. **Go 1.23+** for building
2. **NATS** server for inter-node messaging
3. **At least 3 nodes** for production (2-of-3 threshold minimum)
4. **BadgerDB** password (32-byte AES-256 key) for encrypted storage

## Quick reference

| Item | Value |
|------|-------|
| API | `https://mpc.hanzo.ai` (port 8080, IAM auth) |
| Repo | `github.com/hanzoai/mpc` |
| Module | `github.com/hanzoai/mpc` |
| Branch | `main` |
| Image | `ghcr.io/hanzoai/mpc:v0.4.3` |
| K8s | StatefulSet `hanzo-mpc` (3 replicas) in `hanzo` namespace |
| Threshold | 2-of-3 (default) |
| Binaries | `hanzo-mpc` (node), `hanzo-mpc-cli` (CLI tools) |

## One-file quickstart

### Build and run

```bash
# Build all binaries
make build

# Generate peer configuration (3 nodes)
hanzo-mpc-cli generate-peers -n 3
hanzo-mpc-cli register-peers
hanzo-mpc-cli generate-initiator
hanzo-mpc-cli generate-identity --node node0

# Start a node
hanzo-mpc start -n node0
```

### Docker Compose (full stack)

```bash
cd deploy
docker-compose up -d
# Starts: 3 MPC nodes, NATS, MinIO (backup), KMS
```

### Go client

```go
import (
    "github.com/hanzoai/mpc/pkg/client"
    "github.com/nats-io/nats.go"
)

func main() {
    natsConn, _ := nats.Connect("nats://localhost:4222")
    defer natsConn.Close()

    mpcClient := client.NewMPCClient(client.Options{
        NatsConn: natsConn,
        KeyPath:  "./event_initiator.key",
    })

    // Create a wallet (distributed key generation)
    walletID := "my-wallet-001"
    mpcClient.CreateWallet(walletID)

    // Listen for results
    mpcClient.OnWalletCreationResult(func(event event.KeygenSuccessEvent) {
        fmt.Println("Wallet created:", event)
    })
}
```

## Core Concepts

### Architecture

```
hanzoai/mpc/
  cmd/
    hanzo-mpc/              Main node binary
    hanzo-mpc-cli/          CLI tools (generate-peers, register, identity)
  pkg/
    mpc/                    TSS implementation (CGGMP21, FROST, LSS)
    kvstore/                BadgerDB storage (AES-256 encrypted)
    messaging/              NATS JetStream (pub/sub + P2P)
    identity/               Ed25519 node identity (age encrypted)
    client/                 Go client library
    api/                    HTTP API handlers
    policy/                 Policy engine (signers, limits, whitelist, FHE)
    threshold/              ThresholdVM (policy enforcement in signing)
    storage/                BadgerDB store + S3 backup client
    eventconsumer/          Event processing
    hsm/                    HSM abstraction (file, AWS, GCP, Azure, Zymbit, KMS)
    kms/                    Hanzo KMS integration
    encryption/             Encryption utilities
    encoding/               Serialization (CBOR for FROST/LSS configs)
    config/                 Configuration management
    infra/                  KV backends (ConsensusKV, NATS KV, Consul)
    keyinfo/                Key metadata management
    protocol/               Protocol message types
    types/                  Shared type definitions
    common/                 Common utilities
    constant/               Constants
    logger/                 Structured logging (zerolog)
    utils/                  Helper functions
  contracts/
    ThresholdPolicy.sol     On-chain policy enforcement contract
  deploy/
    compose.yml             Full stack Docker deployment
    Makefile                Deployment automation
  e2e/                      End-to-end integration tests
  examples/                 Usage examples
  docs/                     Documentation
  identity/                 Identity key templates
  config.yaml               Default configuration
```

### Threshold Signing

Uses a **t-of-n threshold scheme** where `t >= floor(n/2) + 1`:

- **ECDSA (secp256k1)**: CGGMP21 protocol for Bitcoin, Ethereum, EVM, XRPL, Lux
- **ECDSA Taproot**: FROST protocol for Bitcoin Taproot (BIP-340)
- **EdDSA (Ed25519)**: FROST (Taproot mode) for Solana, TON (partial)
- **LSS**: Dynamic resharing (change t-of-n without key reconstruction)

### KV Backend Architecture

All cluster state goes through an abstract `infra.KV` interface:

| Backend | Config | Transport | Use Case |
|---------|--------|-----------|----------|
| `consensus` | `kv_backend: consensus` | NATS + Lux Quasar | Production (dual-cert PQ finality) |
| `nats` | `kv_backend: nats` | NATS JetStream KV | Dev/staging |
| `consul` | `kv_backend: consul` | Consul HTTP | Legacy |

**ConsensusKV** runs a private BFT blockchain with dual-certificate finality:
- Classical: Ed25519 (64-byte signatures)
- Post-quantum: ML-DSA-65 (FIPS 204, NIST Level 3, 3,309-byte signatures)
- Block finalized only when BOTH signature types reach t-of-n threshold

### Policy Engine

Fireblocks/Utila-style transaction governance:
- Signers and roles (ADMIN, SIGNER, VIEWER)
- Spending limits (per-tx, daily, monthly by asset)
- Whitelist/blacklist, time windows, rate limiting
- ThresholdVM: evaluates policies before signature shares
- FHE private policies via `luxfi/fhe` (encrypted amount checks, cumulative spending)

### HSM Configuration

```yaml
hsm:
  provider: file  # file, aws, gcp, azure, zymbit, kms
  file:
    base_path: "."
    hex_encoded: true
  # aws:
  #   region: us-east-1
  #   key_arn: arn:aws:kms:us-east-1:123456:key/abc-def
  # gcp:
  #   project: my-project
  #   location: us-east1
  #   key_ring: mpc-keys
  # kms:
  #   site_url: https://kms.hanzo.ai
  #   client_id: ""
  #   client_secret: ""
  #   project_id: ""
```

### NATS Topics

| Topic | Purpose |
|-------|---------|
| `mpc.keygen_request.<walletID>` | Keygen request |
| `mpc.mpc_keygen_result.<walletID>` | Keygen result |
| `mpc.mpc_signing_result.<walletID>` | Signing result |
| `mpc.consensus.<chainID>.blocks` | Consensus blocks |
| `mpc.consensus.<chainID>.proposals` | Consensus proposals |

### Blockchain Support

| Chain | Curve | Protocol | Status |
|-------|-------|----------|--------|
| Bitcoin (Legacy/SegWit) | secp256k1 | CGGMP21/LSS | Full |
| Bitcoin (Taproot) | secp256k1 | FROST | Full |
| Ethereum/EVM | secp256k1 | CGGMP21/LSS | Full |
| XRPL | secp256k1 | CGGMP21/LSS | Full |
| Lux Network | secp256k1 | CGGMP21/LSS | Full |
| Solana | Ed25519 | FROST (Taproot) | Partial |
| TON | Ed25519 | FROST (Taproot) | Partial |

### Production Deployment

```
K8s Cluster: hanzo-k8s (do-sfo3)
  Namespace: hanzo
  StatefulSet: hanzo-mpc (3 replicas)
  Image: ghcr.io/hanzoai/mpc:v0.4.3
  NATS: nats://nats.hanzo.svc.cluster.local:4222
  KV Backend: consensus (M-Chain)
  Config: ConfigMap hanzo-mpc-config
  Identity: Secret hanzo-mpc-identity
  Key shards: BadgerDB on PVC at /data/mpc/db/hanzo-mpc-{N}/
```

### Deploy Stack (compose.yml)

| Service | Port | Purpose |
|---------|------|---------|
| hanzo-mpc-{0,1,2} | 6000-6002 | MPC nodes |
| hanzo-kms | 8080 | Key management |
| nats | 4222 | Message broker |
| minio | 9000 | S3 backup storage |

## Testing

```bash
make test            # Unit tests
make test-coverage   # With coverage report
make e2e-test        # End-to-end (builds binaries first)
make test-all        # Unit + E2E
```

## Critical Serialization Notes

- Protocol messages MUST use `MarshalBinary/UnmarshalBinary`, NOT JSON
- FROST `TaprootConfig` has no JSON marshalers -- use CBOR via `MarshalFROSTConfig()`
- LSS `lssConfig.Config` same -- use CBOR via `MarshalLSSConfig()`
- Party IDs must be sorted consistently across all nodes

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Handler cannot accept message" | Normal broadcast self-receipt | Ignore (expected behavior) |
| Keygen stuck | Nodes not discovering peers | Check NATS connectivity and peer registration |
| BadgerDB error | Wrong password | Verify `BADGER_PASSWORD` is exactly 32 bytes |
| E2E test failures | Stale binary | Run `go install ./cmd/hanzo-mpc && go install ./cmd/hanzo-mpc-cli` |
| Signing fails | Insufficient threshold | Ensure at least t nodes are healthy |

## Related Skills

- `hanzo/hanzo-kms.md` - Key management service (control plane for MPC)
- `hanzo/hanzo-vault.md` - PCI-compliant card tokenization
- `hanzo/hanzo-web3.md` - Web3 services and gateway
- `hanzo/hanzo-evm.md` - EVM execution engine

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: mpc, threshold-signing, wallet, cryptography, custody
**Prerequisites**: Go 1.23+, NATS, BadgerDB
