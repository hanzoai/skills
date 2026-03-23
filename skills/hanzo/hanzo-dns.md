# Hanzo DNS - Programmable DNS Server

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-ingress.md`, `hanzo/hanzo-k8s.md`, `hanzo/hanzo-deploy.md`

## Overview

Hanzo DNS is a **CoreDNS fork** providing a programmable, plugin-based DNS server for Hanzo infrastructure. Go codebase, ships as a single `coredns` binary. Includes a custom `hanzoapi` plugin for API-driven DNS record management. Supports UDP/TCP, DoT (DNS over TLS), DoH (DNS over HTTPS), DoH3, DoQ (DNS over QUIC), and gRPC. All external DNS is managed through Cloudflare; Hanzo DNS handles internal resolution and service discovery.

## When to use

- Internal DNS for Hanzo K8s clusters
- Programmable DNS with API-driven record management via `hanzoapi` plugin
- Service discovery across Kubernetes, etcd, or Nomad
- DNS-over-HTTPS/QUIC termination
- Custom DNS resolution rules for development

## Hard requirements

1. **Port 53** available for DNS listeners (or alternate via `-dns.port`)
2. **Cloudflare for external DNS** -- Hanzo DNS is for internal resolution only
3. **Corefile** configuration required (or specified with `-conf`)

## Quick reference

| Item | Value |
|------|-------|
| Default port | 53 (UDP/TCP) |
| Go module | `github.com/coredns/coredns` |
| Go version | 1.25+ |
| Upstream | CoreDNS |
| Repo | `github.com/hanzoai/dns` |
| Image | `ghcr.io/hanzoai/dns:latest` |
| K8s manifests | `universe/infra/k8s/dns/` |
| Config file | `Corefile` |
| Custom plugin | `hanzoapi` |
| License | Apache 2.0 |

## Architecture

```
External DNS (Cloudflare)
  |- *.hanzo.ai     -> 165.232.146.176 (hanzo-k8s LB)
  |- *.hanzo.team   -> 165.232.146.176
  |- *.lux.network  -> 24.144.69.101 (lux-k8s LB)

Internal DNS (Hanzo DNS / CoreDNS)
  |- *.hanzo.svc    -> cluster-internal IPs
  |- postgres.hanzo.svc -> PostgreSQL pods
  |- redis.hanzo.svc    -> Redis/Valkey pods
  |- llm.hanzo.svc      -> LLM Gateway pods
```

## Cloudflare DNS records (production)

| Domain | Type | Value | Proxied |
|--------|------|-------|---------|
| `*.hanzo.ai` | A | `165.232.146.176` | Yes |
| `*.hanzo.team` | A | `165.232.146.176` | Yes |
| `hanzo.id` | A | `165.232.146.176` | Yes |
| `lux.id` | A | `165.232.146.176` | Yes |
| `zoo.id` | A | `165.232.146.176` | Yes |
| `pars.id` | A | `165.232.146.176` | Yes |
| `*.lux.network` | A | `24.144.69.101` | Yes |

## Corefile example

```
hanzo.svc:53 {
    kubernetes hanzo.svc {
        endpoint https://kubernetes.default.svc
        kubeconfig /etc/kubernetes/admin.conf
    }
    errors
    log
    cache 30
}

.:53 {
    hanzoapi {
        endpoint https://api.hanzo.ai/dns
        token {$DNS_API_TOKEN}
    }
    forward . 1.1.1.1 8.8.8.8
    errors
    log
    cache 60
}
```

## hanzoapi plugin

Custom plugin for API-driven DNS record management:

```
hanzoapi {
    endpoint https://api.hanzo.ai/dns
    token <api-token>
    fallthrough
    ttl 300
}
```

Queries the Hanzo API for DNS records, allowing dynamic record management without zone file edits.

## Plugin chain (50+)

CoreDNS processes queries through a middleware chain. Key plugins:

| Plugin | Purpose |
|--------|---------|
| `kubernetes` | In-cluster service discovery |
| `hanzoapi` | API-driven records (custom) |
| `forward` | Upstream DNS forwarding |
| `cache` | Response caching |
| `errors` | Error logging |
| `log` | Query logging |
| `tls` | DNS-over-TLS listener |
| `doh` | DNS-over-HTTPS listener |
| `hosts` | Static host file records |
| `rewrite` | Query rewriting |
| `loadbalance` | Round-robin DNS |

## Build from source

```bash
cd ~/work/hanzo/dns

# Build (plugin registration from plugin.cfg)
make build

# Run
./coredns -conf Corefile

# Docker
docker build -t ghcr.io/hanzoai/dns:latest .
```

## K8s deployment

```yaml
# universe/infra/k8s/dns/
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hanzo-dns
  namespace: hanzo
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: dns
          image: ghcr.io/hanzoai/dns:latest
          ports:
            - containerPort: 53
              protocol: UDP
            - containerPort: 53
              protocol: TCP
          volumeMounts:
            - name: config
              mountPath: /etc/coredns
      volumes:
        - name: config
          configMap:
            name: hanzo-dns-config
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Port 53 conflict | System resolver running | Change port with `-dns.port` or stop systemd-resolved |
| Plugin not loading | Not in plugin.cfg | Add to `plugin.cfg` and rebuild |
| API records not resolving | Wrong token | Check `hanzoapi` token configuration |
| Slow resolution | No cache | Add `cache` plugin to Corefile |

## Related Skills

- `hanzo/hanzo-ingress.md` -- Ingress controller (DNS resolves to it)
- `hanzo/hanzo-k8s.md` -- K8s cluster (in-cluster DNS)
- `hanzo/hanzo-deploy.md` -- Deployment workflow

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: dns, coredns, cloudflare, service-discovery, resolution
**Prerequisites**: DNS concepts, Cloudflare account
