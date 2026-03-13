# Hanzo Universe - Production Kubernetes Infrastructure

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-stack.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-kms.md`

## Overview

Hanzo Universe is the **production infrastructure repository** containing all Kubernetes manifests, Helm charts, and deployment configurations for the Hanzo ecosystem. Everything runs on DOKS (DigitalOcean Kubernetes).

**NOTE**: This is a private repository. Public details are limited to architecture overview.

## When to use

- Deploying Hanzo services to production
- Modifying K8s manifests for Hanzo services
- Debugging production infrastructure
- Adding new services to the cluster

## Architecture

### Clusters

- **hanzo-k8s**: All core Hanzo services (IAM, KMS, Platform, Cloud, Console, Gateway, Commerce, Chat, Web3)
- **lux-k8s**: Lux blockchain validators, gateway, markets, cloud

### Key Domains

| Domain | Service |
|--------|---------|
| hanzo.id, lux.id, zoo.id | IAM (Casdoor) |
| kms.hanzo.ai | KMS (Infisical) |
| platform.hanzo.ai | PaaS |
| console.hanzo.ai | Console |
| cloud.hanzo.ai | Cloud |
| api.hanzo.ai | API Gateway |
| chat.hanzo.ai | Chat |
| api.lux.network | Lux Gateway |

### Database Strategy

All in-cluster PostgreSQL (no managed databases). Each cluster runs its own PostgreSQL instance serving multiple databases for different services.

**DO App Platform**: DECOMMISSIONED (Feb 2026). Everything is K8s-native.

### Manifest Structure

```
universe/
├── infra/k8s/
│   ├── paas/           # PaaS manifests
│   ├── chat/           # Chat deployment
│   ├── iam/            # IAM/Casdoor
│   ├── kms/            # KMS/Infisical
│   ├── llm/            # LLM Gateway
│   └── ...
```

## Access

This is a **private repository**. Contact the infrastructure team for access.

## Related Skills

- `hanzo/hanzo-stack.md` - Local dev environment (vs production)
- `hanzo/hanzo-platform.md` - PaaS deployments
- `hanzo/hanzo-kms.md` - Secret management

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: kubernetes, production, infrastructure
**Prerequisites**: Kubernetes, kubectl, Helm
