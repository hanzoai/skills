# Hanzo KMS - Key & Secret Management

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-id.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-vault.md`

## Overview

Hanzo KMS is a **centralized secret management platform** for the Hanzo ecosystem. Fork of Infisical with Hanzo branding, Universal Auth, and K8s-native secret sync via KMSSecret CRD. Live at `kms.hanzo.ai`.

### Why Hanzo KMS?

- **Zero plaintext secrets**: All secrets encrypted at rest, synced to K8s
- **Universal Auth**: Machine-to-machine authentication for CI/CD
- **RBAC**: Per-org, per-project, per-environment access control
- **Auto-rotation**: Scheduled secret rotation with audit trail
- **Multi-language SDKs**: Go, Node.js, Python clients

### OSS Base

Fork of **Infisical** with 9 ported Vault subsystems. Backend is TypeScript/Node.js. Repo: `hanzoai/kms`.

### Vault Subsystems

1. **Shamir Secret Sharing** — Split master key across N parties (M-of-N reconstruction)
2. **Transit Encryption** — Encrypt/decrypt data without exposing keys (envelope encryption)
3. **Leases** — Time-bound secret access with automatic revocation
4. **Seal/Unseal** — Cold-start protection requiring quorum to activate
5. **ACL Policies** — Fine-grained access control per path/operation
6. **Token Management** — Scoped, renewable auth tokens with TTL
7. **Dynamic Secrets** — On-demand credential generation (DB users, cloud IAM)
8. **TFHE Bridge** — Fully homomorphic encryption for compute-on-encrypted-secrets (in development)
9. **Per-Org Root Key Isolation** — Each organization has its own root encryption key

## When to use

- Storing API keys, database credentials, tokens
- Syncing secrets to K8s workloads
- CI/CD pipeline secret injection
- Multi-org secret isolation
- Auditing secret access

## Hard requirements

1. **KMS instance** at `kms.hanzo.ai` or self-hosted
2. **Universal Auth** credentials (client ID + secret) for machine access
3. **Project slug** minimum 5 characters (e.g., `hanzo-paas` not `paas`)

## Quick reference

| Item | Value |
|------|-------|
| UI | `https://kms.hanzo.ai` |
| API | `https://kms.hanzo.ai/api/v1` |
| Auth | Universal Auth (client ID + secret) |
| K8s CRD | `KMSSecret` |
| Go SDK | `github.com/hanzoai/kms-go-sdk` |
| Node SDK | `@hanzo/kms-node-sdk` |
| Python SDK | `hanzo-kms` (pip) |
| Repo | `github.com/hanzoai/kms` |

## One-file quickstart

### CLI (fetch secrets)

```bash
# Login
export KMS_TOKEN=$(curl -s -X POST https://kms.hanzo.ai/api/v1/auth/universal-auth/login \
  -H "Content-Type: application/json" \
  -d '{"clientId": "'$KMS_CLIENT_ID'", "clientSecret": "'$KMS_CLIENT_SECRET'"}' \
  | jq -r '.accessToken')

# Fetch secrets
curl -s https://kms.hanzo.ai/api/v1/secrets \
  -H "Authorization: Bearer ${KMS_TOKEN}" \
  -G -d "workspaceId=${PROJECT_ID}&environment=production"
```

### Go SDK

```go
import kms "github.com/hanzoai/kms-go-sdk"

client := kms.NewClient(kms.Config{
    SiteURL:      "https://kms.hanzo.ai",
    ClientID:     os.Getenv("KMS_CLIENT_ID"),
    ClientSecret: os.Getenv("KMS_CLIENT_SECRET"),
})

secret, err := client.GetSecret(kms.GetSecretOptions{
    ProjectID:   "hanzo-paas",
    Environment: "production",
    SecretName:  "DATABASE_URL",
})
fmt.Println(secret.Value)
```

### Node.js SDK

```typescript
import { KMSClient } from "@hanzo/kms-node-sdk"

const client = new KMSClient({
  siteUrl: "https://kms.hanzo.ai",
  clientId: process.env.KMS_CLIENT_ID!,
  clientSecret: process.env.KMS_CLIENT_SECRET!,
})

const secret = await client.getSecret({
  projectId: "hanzo-paas",
  environment: "production",
  secretName: "DATABASE_URL",
})
```

### Python SDK

```python
from hanzo_kms import KMSClient

client = KMSClient(
    site_url="https://kms.hanzo.ai",
    client_id=os.environ["KMS_CLIENT_ID"],
    client_secret=os.environ["KMS_CLIENT_SECRET"],
)

secret = client.get_secret(
    project_id="hanzo-paas",
    environment="production",
    secret_name="DATABASE_URL",
)
```

## Core Concepts

### K8s Secret Sync (KMSSecret CRD)

```yaml
apiVersion: kms.hanzo.ai/v1
kind: KMSSecret
metadata:
  name: my-app-secrets
  namespace: default
spec:
  project: hanzo-paas
  environment: production
  syncInterval: 5m
  secretRef:
    name: my-app-secrets  # K8s Secret to create/update
  secrets:
    - DATABASE_URL
    - REDIS_URL
    - HANZO_API_KEY
```

The KMS operator (`hanzoai/kms-operator`) watches KMSSecret resources and auto-syncs to K8s Secrets.

### Helm Chart: Auto-Bootstrap

```yaml
# values.yaml for kms-standalone chart
kms:
  autoBootstrap:
    additionalOrganizations:
      - hanzo
      - lux
      - zoo
    additionalOrganizationAdminEmails:
      - admin@example.com
    additionalOrganizationsTokenSecretKey: token
```

### CI/CD Integration

```yaml
# .github/workflows/deploy.yml
jobs:
  deploy:
    steps:
      - name: Login to KMS
        run: |
          export KMS_TOKEN=$(curl -s -X POST $KMS_URL/api/v1/auth/universal-auth/login \
            -d '{"clientId":"${{ secrets.KMS_CLIENT_ID }}","clientSecret":"${{ secrets.KMS_CLIENT_SECRET }}"}' \
            | jq -r '.accessToken')
          echo "KMS_TOKEN=$KMS_TOKEN" >> $GITHUB_ENV

      - name: Fetch deploy secrets
        run: |
          DOCKERHUB_TOKEN=$(curl -s "$KMS_URL/api/v1/secrets/raw/DOCKERHUB_TOKEN?workspaceId=$PROJECT_ID&environment=production" \
            -H "Authorization: Bearer $KMS_TOKEN" | jq -r '.secret.secretValue')
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Slug error on project creation | Slug < 5 chars | Use `hanzo-paas` not `paas` |
| Universal Auth login fails | Wrong client credentials | Regenerate in KMS UI |
| KMSSecret not syncing | Operator not running | Check `kms-operator` pod |
| Token key mismatch | Custom bootstrap template | Set `additionalOrganizationsTokenSecretKey` |

## Related Skills

- `hanzo/hanzo-id.md` - IAM (KMS uses IAM for user auth)
- `hanzo/hanzo-vault.md` - PCI card tokenization
- `hanzo/hanzo-platform.md` - PaaS (uses KMS for secrets)
- `hanzo/hanzo-universe.md` - Production K8s manifests

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: secrets, kms, infisical, encryption, security
**Prerequisites**: API keys concepts, K8s basics
