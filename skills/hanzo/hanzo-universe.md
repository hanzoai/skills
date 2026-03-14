# Hanzo Universe - Production Kubernetes Infrastructure

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-operator.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-stack.md`, `hanzo/hanzo-ingress.md`

## Overview

Hanzo Universe (`github.com/hanzoai/universe`) is the **single source of truth** for all production infrastructure across the Hanzo ecosystem. It contains Kustomize manifests for 50+ services, Terraform configs for DigitalOcean provisioning, Helm charts, GitHub Actions CI/CD workflows, Docker Compose stacks, and end-to-end tests. Everything runs on DOKS (DigitalOcean Kubernetes Service).

**Repo**: `github.com/hanzoai/universe` (private)
**Local path**: `~/work/hanzo/universe/`

### Design Principles

- **Kustomize over Helm** for service manifests (Helm used only for IAM identity chart)
- **KMSSecret CRDs** for all secrets (never plaintext in manifests)
- **Single namespace** (`hanzo`) for most services; operators and external stacks get their own
- **Idempotent deployment**: `kubectl kustomize . | kubectl apply -f -`
- **All images**: `ghcr.io/hanzoai/<service>:latest`, always `--platform linux/amd64`

## When to Use

- Deploying or updating any Hanzo service in production
- Adding a new service to the cluster
- Debugging production infrastructure (manifests, secrets, ingress)
- Modifying RBAC, monitoring, or observability
- Provisioning new infrastructure with Terraform
- Running CI/CD pipelines for Hanzo services

## Cluster Topology

### hanzo-k8s (`24.199.76.156`)

Primary cluster. Runs all core Hanzo services. DigitalOcean SFO3 region. 4-node worker pool with `hostNetwork` ingress.

### lux-k8s (`24.144.69.101`)

Lux blockchain cluster. Runs validators (mainnet/testnet/devnet), gateway, explorer, bridge, exchange, and bootnode. Applied separately: `kubectl apply -k lux/ --context do-sfo3-lux-k8s`.

### Deployment

DO App Platform is **decommissioned** (Feb 2026). All infrastructure is K8s-native.

## Repository Structure

```
universe/
  infra/
    k8s/                        # Production Kustomize manifests (main entry point)
      kustomization.yaml         # Root -- includes all services below
      Makefile                   # diff, apply, validate, build, deploy-%
      namespace.yaml             # hanzo namespace
      cluster-issuer.yaml        # Let's Encrypt ClusterIssuer (cert-manager)
      platform-ingress.yaml      # Platform-level ingress rules
      base/                      # Shared deployment/ingress templates
      rbac/                      # Cluster-wide RBAC (readonly, operator, ci roles)
      ci/                        # CI deployer ServiceAccount
      ingress/                   # Hanzo Ingress (Traefik-based L7 proxy, 4 replicas)
      # --- Operators ---
      hanzo-operator/            # HanzoService/Datastore/Gateway/MPC/Network/Ingress/Platform CRDs
      kms-operator/              # KMSSecret/KMSDynamicSecret/KMSPushSecret CRDs
      nchain/                    # Blockchain node operator (nchain-system namespace)
      # --- Data ---
      sql/                       # PostgreSQL 16 + ZAP (StatefulSet, 20Gi PVC)
      kv/                        # Valkey + ZAP (StatefulSet, 2Gi PVC)
      docdb/                     # DocDB -- FerretDB (MongoDB wire protocol over PostgreSQL)
      storage/                   # Hanzo S3 (MinIO, s3.hanzo.ai)
      search/                    # Meilisearch (search.hanzo.ai, backup CronJob)
      vector/                    # Qdrant (vector.hanzo.ai, backup CronJob)
      nats/                      # NATS core messaging
      stream/                    # Kafka wire protocol gateway over PubSub
      cloud-sql/                 # Serverless PostgreSQL (Neon-based: pageserver, safekeeper, compute)
      # --- Identity & Security ---
      iam/                       # Casdoor IAM (hanzo.id, lux.id, zoo.id, pars.id)
      kms/                       # KMS / Infisical (kms.hanzo.ai)
      login/                     # White-label login UI
      mpc/                       # Multi-party computation (threshold signing, StatefulSet)
      cloudflare/                # Cloudflare API credentials
      # --- Zero Trust ---
      zt/                        # ZeroTier overlay mesh (controller, routers, MCP gateway)
      # --- AI & Models ---
      zen/                       # Zen LM Gateway (identity injection + Console tracking)
      models/                    # Model registry API (models.hanzo.ai)
      pricing/                   # Pricing API + CronJob (pricing.hanzo.ai)
      rag-api/                   # RAG API for chat
      crawl/                     # crawl4ai web crawler
      # --- Applications ---
      chat/                      # Hanzo Chat (LibreChat + DocDB + search)
      bot/                       # Hanzo Bot (gateway, hub, docs, agents control plane, playground)
      cloud/                     # Cloud API + agent + site
      commerce/                  # Commerce API (HPA, PDB, service)
      console/                   # Developer console + worker
      billing/                   # Billing center
      gateway/                   # API gateway (api.hanzo.ai)
      flow/                      # Visual AI workflow builder + site
      auto/                      # Visual workflow automation
      visor/                     # Agent orchestration
      vm/                        # Virtual machine management
      vmd/                       # VM daemon
      # --- Serverless ---
      openfaas/                  # OpenFaaS function control plane (controller, gateway, queue-worker)
      # --- Platform ---
      paas/                      # PaaS v2 (platform.hanzo.ai, RBAC, PDB)
      gitops/                    # GitOps sync engine
      registry/                  # Container registry + PVC
      argocd/                    # ArgoCD bootstrap (applications, RBAC, ingress)
      # --- Observability ---
      analytics/                 # Event analytics
      insights/                  # Product analytics
      monitoring/                # Prometheus ServiceMonitors, alerts, Grafana dashboards, OTel collector
      logging/                   # Loki + Promtail log pipeline
      o11y/                      # OTel collector config
      status/                    # Status pages (hanzo, lux, zoo, pars, adnexus)
      sentry/                    # Sentry error tracking
      # --- Sites & Misc ---
      app/                       # hanzo.ai main app
      base/                      # Database admin UI
      bootnode/                  # bootnode.hanzo.ai (API + web)
      bootnode-whitelabel/       # White-label bootnode deployments
      captable/                  # Cap table management
      dataroom/                  # Data room
      dns/                       # DNS service (ConfigMap, PDB)
      lux-build/                 # Lux build service
      preview/                   # PR preview environments (namespace, certs, pull secrets)
      sign/                      # Document signing
      team/                      # Project management (own namespace: team)
      # --- External Stacks (separate clusters) ---
      lux/                       # Lux blockchain (lux-k8s cluster)
        operator/                # Lux operator
        validators/              # 15 validators across 3 networks (mainnet/testnet/devnet)
        gateway/                 # KrakenD API gateway
        explorer/                # Blockscout explorer + frontend
        bridge/                  # Cross-chain bridge
        exchange/                # DEX
        bootnode-lux/            # Lux bootnode
        agnost/                  # Agnost deployments
      adnexus/                   # AdNexus advertising (adnexus-k8s cluster)
  terraform/
    main.tf                      # DigitalOcean VPC, Droplets, LB, Firewall, DNS
    autoscaler.tf                # Docker Swarm autoscaler config
    terraform.tfvars.example     # Variable template
  do/
    autoscaler.py                # DigitalOcean autoscaler script
    compose.autoscaler.yml       # Autoscaler compose
  scripts/
    manager-init.sh              # Swarm manager bootstrap
    worker-init.sh               # Swarm worker bootstrap
  deploy.sh                      # One-shot infra deploy (Terraform + Swarm + Universe)
  k8s-deploy.sh                  # K8s-only deploy script
  README.md                      # Infra documentation
  helm/
    identity/                    # IAM Helm chart (Casdoor)
      Chart.yaml
      values.yaml
      values-production.yaml
      templates/                 # deployment, service, ingress, configmap, secrets, pdb
    hanzo-stack/
      values.production.yaml     # Production values for full stack chart
  deploy/                        # Docker Compose production deployment
    compose.production.yml
    deploy.sh                    # Production deploy script
    kms.sh                       # KMS bootstrap
    dns-setup.sh                 # DNS configuration
    staging/                     # Staging environment
  services/                      # Docker Compose service definitions
    compose.yml                  # Main services compose
    compose-hanzo-*.yml          # Hanzo service variants (complete, dev, minimal, services)
    compose-infra.yml            # Infrastructure compose (postgres, redis)
    compose-lux.yml              # Lux services compose
    compose-nodes.yml            # Node compose
    config/                      # Router configs, IAM configs, MCP servers
  .github/
    workflows/                   # 17 CI/CD workflows
    actions/                     # Reusable actions (KMS auth, KMS action)
  compose.yml                    # Root development compose (29k lines)
  compose.production.yml         # Production compose
  compose.single-node.yml        # Single-node compose
  Makefile                       # Root Makefile (up, down, status, logs, health, db-backup)
  Cargo.toml                     # Rust workspace root
  e2e/                           # End-to-end tests
  docs/                          # Documentation
  scripts/                       # Utility scripts
```

## K8s Deployment Commands

```bash
# From infra/k8s/:

# Preview changes (dry-run diff)
make diff

# Apply all resources idempotently
make apply

# Validate manifests without applying
make validate

# Render manifests to stdout
make build

# Deploy a single service
make deploy-chat
make deploy-iam
make deploy-kms

# Apply Lux cluster separately
kubectl apply -k lux/ --context do-sfo3-lux-k8s
```

## Core Data Services

| Service | Image | Type | Port | Storage |
|---------|-------|------|------|---------|
| **sql** | `ghcr.io/hanzoai/sql:latest` | StatefulSet | 5432 (PG) + 9651 (ZAP) | 20Gi do-block-storage |
| **kv** | `ghcr.io/hanzoai/kv:latest` | StatefulSet | 6379 (Valkey) + 9651 (ZAP) | 2Gi |
| **docdb** | DocDB | StatefulSet | MongoDB wire protocol | Via PostgreSQL |
| **storage** | MinIO | Deployment | S3 API | PVC |
| **search** | Meilisearch | Deployment | HTTP | PVC + backup CronJob |
| **vector** | Qdrant | Deployment | gRPC + HTTP | PVC + backup CronJob |
| **nats** | NATS | - | 4222 | - |
| **stream** | Kafka protocol | Deployment | Kafka wire | - |
| **cloud-sql** | Neon-based | Multi-component | 5432 | S3-backed |

The `sql` and `kv` images are unified containers: PostgreSQL 16 + embedded ZAP server, and Valkey + embedded ZAP server respectively. Both expose dual ports. A backward-compat `postgres` Service alias exists for `sql`.

## Secret Management

All secrets flow through KMS (kms.hanzo.ai) via the **KMS Operator**.

### CRDs (API group: `secrets.lux.network/v1alpha1`)

| CRD | Purpose |
|-----|---------|
| **KMSSecret** | Pulls secrets from KMS and syncs to K8s Secret |
| **KMSDynamicSecret** | Dynamic secrets with TTL and lease management |
| **KMSPushSecret** | Pushes K8s Secrets back to KMS |

### How KMSSecret Works

```yaml
apiVersion: secrets.lux.network/v1alpha1
kind: KMSSecret
metadata:
  name: bot-kms-sync
  namespace: hanzo
spec:
  hostAPI: http://kms.hanzo.svc.cluster.local/api
  resyncInterval: 60
  authentication:
    universalAuth:
      credentialsRef:
        secretName: universal-auth-credentials
        secretNamespace: hanzo
      secretsScope:
        projectSlug: secrets-639-c
        envSlug: prod
        secretsPath: /bot
  managedSecretReference:
    secretName: bot-secrets
    secretNamespace: hanzo
    secretType: Opaque
```

Authentication methods: Universal Auth (credentialsRef), Kubernetes Auth (serviceAccountRef + identityId), Service Token (legacy).

The `universal-auth-credentials` secret (clientId/clientSecret) is created out-of-band via kubectl. Never committed to git.

Services with `kms-secrets.yaml`: bot, bootnode, captable, chat, cloud, commerce, insights, o11y, search, sign, sentry, storage, team, vector, zen, zt.

Services with direct `secret.yaml`/`secrets.yaml` (legacy or bootstrap): analytics, app, auto, cloudflare, flow, gateway, kv, mpc, paas, registry, sql.

## Ingress Architecture

The ingress layer is a **Traefik-based L7 reverse proxy** (`ghcr.io/hanzoai/ingress:latest`):

- **4 replicas** with `hostNetwork: true` (pods bind directly to node ports 80/443)
- **podAntiAffinity** ensures one pod per node across the worker pool
- **TLS**: ACME DNS-01 via Cloudflare + static Cloudflare origin certs for `hanzo.ai`, `lux.network`, `hanzo.bot`
- **Providers**: Kubernetes Ingress (`ingressClass: ingress`), Kubernetes CRD (cross-namespace), file provider (1500+ lines of static routes)
- **Tracing**: OpenTelemetry to `otel-collector.hanzo.svc:4317`
- **Cert resolution**: Let's Encrypt with Cloudflare DNS challenge (`dev@hanzo.ai`)

Cloudflare DNS wildcards (`*.hanzo.ai`) round-robin across all 4 pod node IPs.

## Hanzo Operator CRDs

The Hanzo Operator (`ghcr.io/hanzoai/operator:latest`) runs in `hanzo-operator-system` namespace and manages 7 CRDs under `hanzo.ai/v1alpha1`:

| CRD | Description |
|-----|-------------|
| **HanzoService** | Standard service (Deployment, Service, Ingress, HPA, PDB, NetworkPolicy, KMSSecret) |
| **HanzoDatastore** | Stateful data service (StatefulSet, headless Service, PVC, CronJob backup) |
| **HanzoGateway** | API gateway (KrakenD-based, ConfigMap routing) |
| **HanzoMPC** | Multi-party computation (StatefulSet, dashboard, cache) |
| **HanzoNetwork** | Blockchain validator network (StatefulSet, bootnode, indexer, explorer, bridge) |
| **HanzoIngress** | Multi-host Ingress with cert-manager TLS |
| **HanzoPlatform** | Meta-CRD composing all the above |

34 sample CRs in `infra/k8s/hanzo-operator/crs/` cover every deployed service. PodDisruptionBudgets in `pdbs/` for critical services (cloud-api, console, gateway, hanzo-app, hanzo-login, iam, kms, models, paas, pricing).

## Lux Blockchain Stack (lux-k8s)

Applied separately via `kubectl apply -k lux/ --context do-sfo3-lux-k8s`:

- **3 networks**: mainnet (networkId 1), testnet (networkId 2), devnet (networkId 3)
- **15 validators total**: 5 per network, managed by `lux.network/v1alpha1 LuxNetwork` CRD
- **Image**: `ghcr.io/luxfi/node:v1.23.23`
- **Ports**: mainnet 9630-9631, testnet 9640-9641, devnet 9650-9651
- **Staking keys**: Synced from KMS (`kms.lux.network`) per environment
- **Snapshots**: Hourly to S3 (in-cluster MinIO), 5 retained, RLP exports included
- **Gateway**: KrakenD (`api.lux.network`)
- **Explorer**: Blockscout + frontend (`explorer.lux.network`)
- **Bridge**: Cross-chain bridge service
- **Exchange**: DEX service

## Monitoring and Observability

### Prometheus ServiceMonitors

17 ServiceMonitors scrape metrics from: IAM, commerce, gateway, bot, cloud, KMS, chat, analytics, console, PaaS, insights, auto, flow, base, zen, plus OTel collector.

### Grafana Dashboards

6 ConfigMap-based dashboards: Payment monitoring, Platform health, Web analytics, Product usage, AI operations, Per-organization metrics.

### Alerts (PrometheusRule)

Critical alerts: `PaymentFailureRateHigh` (>5% failure over 5m), `ServiceDown` (any hanzo namespace service unreachable >2m), `PodCrashLooping` (>5 restarts in 15m), `HighErrorRate5xx` (5xx error rate threshold).

### Logging

Loki + Promtail pipeline with Grafana log exploration dashboard.

### Tracing

OpenTelemetry Collector (`otel-collector.hanzo.svc:4317`) receives traces from ingress and application services.

## RBAC Model

Three cluster-wide roles in `rbac/`:

| Role | Scope | Purpose |
|------|-------|---------|
| **readonly** | Cluster | Read-only access for developers |
| **operator** | Cluster | Full management for operators |
| **ci** | Cluster + namespace | Automated deployments from CI |

## CI/CD Workflows

17 GitHub Actions workflows in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR | Lint, test, validate manifests |
| `deploy.yml` | Manual | Deploy specific services |
| `deploy-production.yml` | Push to main | Full production deploy |
| `auto-deploy.yml` | Various | Auto-deploy on changes |
| `e2e-production.yml` | Post-deploy | End-to-end production tests |
| `nightly-backup.yml` | Cron | Database backups |
| `pr-preview.yml` | PR | Preview environments |
| `release.yml` | Tag | Release artifacts |
| `build-zap-images.yml` | Manual | Build sql/kv unified images |
| `reusable-deploy-service.yml` | Called | Reusable service deploy |
| `reusable-deploy-cf-pages.yml` | Called | Cloudflare Pages deploy |
| `reusable-deploy-static.yml` | Called | Static site deploy |
| `reusable-go.yml` | Called | Go build/test pipeline |
| `reusable-python.yml` | Called | Python build/test pipeline |
| `reusable-rust.yml` | Called | Rust build/test pipeline |
| `reusable-typescript.yml` | Called | TypeScript build/test pipeline |
| `reusable-kms-oidc.yml` | Called | KMS OIDC auth for CI |

Custom GitHub Actions in `.github/actions/`: `kms-action` (fetch secrets), `kms-auth` (authenticate to KMS).

## Terraform Infrastructure

`infra/terraform/main.tf` provisions on DigitalOcean:

- **VPC**: `10.100.0.0/16` private network
- **Manager node**: `s-4vcpu-8gb` Ubuntu 24.04 (Docker Swarm manager)
- **Worker nodes**: `s-2vcpu-4gb` x3 initial, autoscale 2-99
- **Load balancer**: HTTPS passthrough to Hanzo Ingress
- **Floating IP**: Stable IP for manager
- **Firewall**: SSH/HTTP/HTTPS public; Swarm ports VPC-only
- **DNS**: A records (root + wildcard) pointing to LB

## Docker Compose (Local Development)

Root `Makefile` orchestrates three compose files:

```bash
make setup    # Create network, start postgres + redis + ClickHouse
make up       # Start all services (services + datastore + IAM)
make down     # Stop everything
make status   # Show service status across all compose files
make health   # Health check all services
make db-backup # Backup PostgreSQL + ClickHouse
```

Individual service targets: `make traefik`, `make iam`, `make cloud`, `make analytics`, `make datastore`, `make platform`, `make infra`.

## Key Domain Mapping

| Domain | Service | Cluster |
|--------|---------|---------|
| hanzo.id, lux.id, zoo.id, pars.id, id.ad.nexus, id.bootno.de | IAM (Casdoor) | hanzo-k8s |
| kms.hanzo.ai | KMS (Infisical) | hanzo-k8s |
| api.hanzo.ai | API Gateway | hanzo-k8s |
| console.hanzo.ai | Developer Console | hanzo-k8s |
| cloud.hanzo.ai | Cloud Dashboard | hanzo-k8s |
| chat.hanzo.ai, hanzo.chat | Hanzo Chat | hanzo-k8s |
| platform.hanzo.ai | PaaS | hanzo-k8s |
| flow.hanzo.ai | Visual Workflow | hanzo-k8s |
| auto.hanzo.ai | Automation | hanzo-k8s |
| commerce.hanzo.ai | Commerce API | hanzo-k8s |
| analytics.hanzo.ai | Event Analytics | hanzo-k8s |
| search.hanzo.ai | Full-text Search | hanzo-k8s |
| vector.hanzo.ai | Vector DB | hanzo-k8s |
| s3.hanzo.ai | Object Storage | hanzo-k8s |
| status.hanzo.ai | Status Page | hanzo-k8s |
| hanzo.team | Project Management | hanzo-k8s |
| api.lux.network | Lux Gateway (KrakenD) | lux-k8s |
| explorer.lux.network | Blockscout Explorer | lux-k8s |

## Database Strategy

All databases are **in-cluster PostgreSQL** (no managed DBaaS). The `sql` StatefulSet serves multiple databases for different services via `init-db.sql`. Valkey (`kv`) handles caching and queues. DocDB provides MongoDB wire protocol compatibility over PostgreSQL via FerretDB.

Cloud SQL (Neon-based) provides serverless PostgreSQL with pageserver, safekeeper, storage-broker, and compute nodes.

## Common Operations

```bash
# Deploy everything to hanzo-k8s
cd ~/work/hanzo/universe/infra/k8s && make apply

# Deploy single service
cd ~/work/hanzo/universe/infra/k8s && make deploy-chat

# Preview changes
cd ~/work/hanzo/universe/infra/k8s && make diff

# Check KMS secret sync status
kubectl get kmssecrets -n hanzo

# View operator logs
kubectl logs -n hanzo-operator-system deploy/hanzo-operator-controller-manager

# View ingress logs
kubectl logs -n hanzo -l app=hanzo-ingress --tail=100

# Apply Lux stack
kubectl apply -k ~/work/hanzo/universe/infra/k8s/lux/ --context do-sfo3-lux-k8s
```

## Related Skills

- `hanzo/hanzo-operator.md` -- Hanzo K8s operator (7 CRDs, reconciliation)
- `hanzo/hanzo-kms.md` -- Secret management (KMS/Infisical)
- `hanzo/hanzo-platform.md` -- PaaS deployments (platform.hanzo.ai)
- `hanzo/hanzo-stack.md` -- Local dev environment (vs production)
- `hanzo/hanzo-ingress.md` -- Traefik ingress layer
- `hanzo/hanzo-sql.md` -- PostgreSQL + ZAP unified image
- `hanzo/hanzo-kv.md` -- Valkey + ZAP cache
- `hanzo/hanzo-iam.md` -- Identity and access management
- `hanzo/hanzo-zt.md` -- Zero Trust mesh overlay

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: kubernetes, k8s, infrastructure, kustomize, terraform, production, deployment, ingress, operators, secrets, monitoring
**Prerequisites**: Kubernetes, kubectl, Kustomize, Terraform (for provisioning), Docker Compose (for local dev)
