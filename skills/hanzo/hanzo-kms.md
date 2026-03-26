<!-- Updated: 2026-03-26T15:03:36Z -->
# Hanzo KMS - Key and Secret Management

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-id.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-vault.md`, `hanzo/hanzo-k8s.md`

## Overview

Hanzo KMS is a **centralized secret management platform** for the Hanzo ecosystem. Fork of Infisical with Hanzo branding, 9 ported Vault subsystems, Universal Auth, and K8s-native secret sync via KMSSecret CRD. Live at `kms.hanzo.ai`. Every secret in the Hanzo stack flows through KMS -- no plaintext credentials in manifests, ever.

## When to use

- Storing API keys, database credentials, tokens for any service
- Syncing secrets to K8s workloads via KMSSecret CRD
- CI/CD pipeline secret injection
- Multi-org secret isolation
- Auditing secret access
- Generating dynamic secrets (database users, cloud IAM)
- Encrypting/decrypting data via Transit encryption

## Hard requirements

1. **KMS for ALL secrets**: Never hardcode in manifests, env files, or git
2. **Universal Auth** credentials (client ID + secret) for machine access
3. **Project slug minimum 5 characters** (e.g., `hanzo-paas` not `paas`)
4. **RBAC enforced** on all key and secret operations
5. **Audit logging** enabled for all operations

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
| Upstream | Infisical |
| Repo | `github.com/hanzoai/kms` |
| K8s manifests | `universe/infra/k8s/kms/` |
| Image | `ghcr.io/hanzoai/kms:latest` |
| Operator | `hanzoai/kms-operator` |
| Backend | Node.js/Fastify/TypeScript |
| Frontend | React 18/Vite |
| Database | PostgreSQL + Redis |

## Vault subsystems (9)

1. **Shamir Secret Sharing** -- Split master key across N parties (M-of-N reconstruction)
2. **Transit Encryption** -- Encrypt/decrypt without exposing keys (envelope encryption)
3. **Leases** -- Time-bound secret access with automatic revocation
4. **Seal/Unseal** -- Cold-start protection requiring quorum to activate
5. **ACL Policies** -- Fine-grained access control per path/operation
6. **Token Management** -- Scoped, renewable auth tokens with TTL
7. **Dynamic Secrets** -- On-demand credential generation (DB users, cloud IAM)
8. **TFHE Bridge** -- Fully homomorphic encryption for compute-on-encrypted-secrets
9. **Per-Org Root Key Isolation** -- Each organization has its own root encryption key

## CLI quickstart

```bash
# Login via Universal Auth
export KMS_TOKEN=$(curl -s -X POST https://kms.hanzo.ai/api/v1/auth/universal-auth/login \
  -H "Content-Type: application/json" \
  -d '{"clientId": "'$KMS_CLIENT_ID'", "clientSecret": "'$KMS_CLIENT_SECRET'"}' \
  | jq -r '.accessToken')

# Fetch secrets
curl -s https://kms.hanzo.ai/api/v1/secrets \
  -H "Authorization: Bearer ${KMS_TOKEN}" \
  -G -d "workspaceId=${PROJECT_ID}&environment=production"
```

## SDK quickstart

### Go

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
```

### Node.js

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

### Python

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

## K8s secret sync (KMSSecret CRD)

The KMS operator watches KMSSecret resources and auto-syncs to K8s Secrets:

```yaml
apiVersion: kms.hanzo.ai/v1
kind: KMSSecret
metadata:
  name: my-app-secrets
  namespace: hanzo
spec:
  project: hanzo-paas
  environment: production
  syncInterval: 5m
  secretRef:
    name: my-app-secrets  # K8s Secret created/updated
  secrets:
    - DATABASE_URL
    - REDIS_URL
    - HANZO_API_KEY
```

Deployments reference the synced secret:

```yaml
spec:
  containers:
    - name: my-app
      envFrom:
        - secretRef:
            name: my-app-secrets
```

## Helm chart: auto-bootstrap

```yaml
# values.yaml for kms-standalone chart
kms:
  autoBootstrap:
    additionalOrganizations:
      - hanzo
      - lux
      - zoo
    additionalOrganizationAdminEmails:
      - z@hanzo.ai
    additionalOrganizationsTokenSecretKey: token
```

## CI/CD integration

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

## Authentication methods

- Universal Auth (API key + secret) -- primary for machines
- Kubernetes Service Account Auth
- AWS IAM Auth
- GCP IAM Auth
- Azure AD Auth
- OIDC Auth
- JWT Auth

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ENCRYPTION_KEY` | Yes | 16-byte hex key for platform encryption |
| `AUTH_SECRET` | Yes | JWT signing secret |
| `DB_CONNECTION_URI` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `SITE_URL` | Yes | Base URL for KMS frontend |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Slug error on project creation | Slug < 5 chars | Use `hanzo-paas` not `paas` |
| Universal Auth login fails | Wrong client credentials | Regenerate in KMS UI |
| KMSSecret not syncing | Operator not running | Check `kms-operator` pod |
| Token key mismatch | Custom bootstrap template | Set `additionalOrganizationsTokenSecretKey` |
| HSM connection failed | Wrong PKCS#11 path | Verify `HSM_LIBRARY_PATH` and slot number |

## Related Skills

- `hanzo/hanzo-id.md` -- IAM (KMS uses IAM for user auth)
- `hanzo/hanzo-vault.md` -- PCI card tokenization
- `hanzo/hanzo-platform.md` -- PaaS (uses KMS for secrets)
- `hanzo/hanzo-k8s.md` -- K8s infrastructure
- `hanzo/hanzo-deploy.md` -- Deployment workflow

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: secrets, kms, infisical, encryption, security, vault
**Prerequisites**: API key concepts, K8s basics
