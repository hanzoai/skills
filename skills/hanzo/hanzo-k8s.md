# Hanzo K8s - Kubernetes Infrastructure

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-deploy.md`, `hanzo/hanzo-ingress.md`, `hanzo/hanzo-kms.md`

## Overview

Hanzo runs all production services on **DOKS (DigitalOcean Kubernetes Service)** clusters. Two primary clusters: `hanzo-k8s` for all Hanzo services and `lux-k8s` for Lux blockchain. All manifests live in the universe repo. Kustomize for manifest management, KMSSecret CRDs for secret sync, Hanzo Ingress (Traefik) for routing.

## When to use

- Managing or debugging K8s workloads on hanzo-k8s or lux-k8s
- Adding new services to the cluster
- Troubleshooting pod failures, networking, or ingress issues
- Understanding the cluster topology

## Hard requirements

1. **Kustomize, not Helm**: `kubectl kustomize . | kubectl apply -f -`
2. **KMS for all secrets**: KMSSecret CRDs, never hardcode in manifests
3. **In-cluster PostgreSQL**: `postgres.hanzo.svc`, no managed DB services
4. **Hanzo IngressClass**: `ingressClassName: hanzo` on all Ingress resources
5. **GHCR images**: `ghcr.io/hanzoai/<service>:<tag>`, always `--platform linux/amd64`
6. **No DO App Platform**: Fully decommissioned as of Feb 2026

## Quick reference

### hanzo-k8s (production)

| Item | Value |
|------|-------|
| Provider | DigitalOcean Kubernetes (DOKS) |
| Location | SFO3 |
| Context | `do-sfo3-hanzo-k8s` |
| Nodes | 22 (15 workers + 7 runners) |
| LB IP | `165.232.146.176` |
| Namespace | `hanzo` (primary) |
| IngressClass | `hanzo` (Traefik) |
| PostgreSQL | `postgres.hanzo.svc:5432` |
| Redis/Valkey | `redis.hanzo.svc:6379` |
| MongoDB | `mongodb.hanzo.svc:27017` |
| MinIO | `minio.hanzo.svc:9000` |

### lux-k8s (blockchain)

| Item | Value |
|------|-------|
| Provider | DigitalOcean Kubernetes (DOKS) |
| Location | SFO3 |
| Context | `do-sfo3-lux-k8s` |
| LB IP | `24.144.69.101` |
| Services | 15 validators, gateway, markets, lux-cloud-web |
| PostgreSQL | `postgres.hanzo.svc:5432` |

## Cluster services map

### hanzo-k8s domains and services

| Domain | Backend Service | Port |
|--------|-----------------|------|
| `hanzo.ai` | hanzo-app | 3000 |
| `api.hanzo.ai` | hanzo-gateway (KrakenD) | 8000 |
| `llm.hanzo.ai` | hanzo-gateway | 8000 |
| `hanzo.id`, `lux.id`, `zoo.id`, `pars.id` | IAM (Casdoor) | 8000 |
| `kms.hanzo.ai` | KMS (Infisical) | 8080 |
| `platform.hanzo.ai` | Platform (Dokploy) | 3000 |
| `console.hanzo.ai` | Console (Langfuse) | 3000 |
| `cloud.hanzo.ai` | Cloud (Casibase) | 8000 |
| `chat.hanzo.ai` | Chat (LibreChat) | 3080 |
| `hanzo.team` | Team (Huly fork) | 8087 |
| `billing.hanzo.ai` | Billing portal | 80 |
| `o11y.hanzo.ai` | SigNoz | 3301 |

### lux-k8s domains and services

| Domain | Backend Service |
|--------|-----------------|
| `api.lux.network` | KrakenD Gateway |
| `cloud.lux.network` | Lux Cloud |
| `markets.lux.network` | Markets |

## Database topology

### hanzo-k8s databases

```
postgres.hanzo.svc:5432
  |- iam          (Casdoor)
  |- cloud        (Casibase)
  |- console      (Langfuse)
  |- hanzo_cloud  (Cloud API)
  |- kms          (Infisical)
  |- platform     (Dokploy)
```

### lux-k8s databases

```
postgres.hanzo.svc:5432
  |- cloud
  |- commerce
  |- console
  |- gateway
  |- hanzo
  |- kms
```

## Common operations

### Check service status

```bash
# List all pods
kubectl --context do-sfo3-hanzo-k8s -n hanzo get pods

# Check specific service
kubectl --context do-sfo3-hanzo-k8s -n hanzo get pods -l app=<service>

# View logs
kubectl --context do-sfo3-hanzo-k8s -n hanzo logs -l app=<service> --tail=100 -f

# Describe pod (for events/errors)
kubectl --context do-sfo3-hanzo-k8s -n hanzo describe pod <pod-name>
```

### Deploy from universe

```bash
cd ~/work/hanzo/universe/infra/k8s/<service>
kubectl --context do-sfo3-hanzo-k8s kustomize . | kubectl apply -f -
```

### Port forward for debugging

```bash
# Access a service locally
kubectl --context do-sfo3-hanzo-k8s -n hanzo port-forward svc/<service> 8080:8080

# Access PostgreSQL
kubectl --context do-sfo3-hanzo-k8s -n hanzo port-forward svc/postgres 5432:5432
```

### Scale a deployment

```bash
kubectl --context do-sfo3-hanzo-k8s -n hanzo scale deployment/<service> --replicas=3
```

## Manifest patterns

### Standard deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-service
  namespace: hanzo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: my-service
  template:
    metadata:
      labels:
        app: my-service
    spec:
      containers:
        - name: my-service
          image: ghcr.io/hanzoai/my-service:latest
          ports:
            - containerPort: 8080
          envFrom:
            - secretRef:
                name: my-service-secrets  # Synced from KMS
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
```

### Standard ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-service
  namespace: hanzo
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt
spec:
  ingressClassName: hanzo
  tls:
    - hosts:
        - my-service.hanzo.ai
      secretName: my-service-tls
  rules:
    - host: my-service.hanzo.ai
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: my-service
                port:
                  number: 8080
```

### KMSSecret for secret sync

```yaml
apiVersion: kms.hanzo.ai/v1
kind: KMSSecret
metadata:
  name: my-service-secrets
  namespace: hanzo
spec:
  project: my-service
  environment: production
  syncInterval: 5m
  secretRef:
    name: my-service-secrets
  secrets:
    - DATABASE_URL
    - API_KEY
```

## Node pools

| Pool | Count | Size | Purpose |
|------|-------|------|---------|
| worker-pool | 15 | 8vCPU/32GB | Application workloads |
| runner-pool | 7 | 8vCPU/32GB | CI/CD self-hosted runners |

Runner pool is oversized (17-26% utilization). Plan to scale to 2 nodes.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Pod CrashLoopBackOff | Missing secrets | Check KMSSecret CRD is synced: `kubectl get kmssecrets` |
| Ingress 404 | Wrong IngressClass | Ensure `ingressClassName: hanzo` |
| DNS not resolving | Missing Cloudflare record | Add A record pointing to `165.232.146.176` |
| Can't pull image | GHCR auth expired | Check imagePullSecrets on service account |
| DB connection refused | Wrong host | Use `postgres.hanzo.svc` (in-cluster) not external URL |
| Network policy blocking | Missing ingress rule | Add NetworkPolicy allowing ingress from Ingress controller |

## Related Skills

- `hanzo/hanzo-deploy.md` -- Deployment workflow
- `hanzo/hanzo-ingress.md` -- Ingress controller
- `hanzo/hanzo-kms.md` -- Secret management
- `hanzo/hanzo-o11y.md` -- Observability

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: kubernetes, doks, infrastructure, cluster, k8s
**Prerequisites**: kubectl, K8s cluster access, Cloudflare DNS
