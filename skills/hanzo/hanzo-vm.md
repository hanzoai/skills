# Hanzo VM - Cloud OS and VM Management Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-iam.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo VM is an **open-source cloud operating system and virtual machine management platform** built with Go (Beego framework) and React. Go module `github.com/hanzoai/vm`, ships as a single `server` binary with a bundled web frontend. Manages VMs across multiple cloud providers (AWS, Azure, GCP, Alibaba, Tencent, DigitalOcean) and local hypervisors (libvirt, Proxmox). Authenticates via Hanzo IAM (Casdoor). License: Apache-2.0.

### Why Hanzo VM?

- **Multi-cloud**: AWS EC2, Azure, GCP, Alibaba, Tencent Cloud in one dashboard
- **Local hypervisors**: libvirt (KVM/QEMU) and Proxmox VE support
- **Hanzo IAM integration**: SSO via Casdoor, Casbin RBAC authorization
- **RDP/SSH tunneling**: WebSocket-based remote desktop via vmd (Guacamole daemon)
- **Blockchain records**: On-chain VM operation audit trail (`chain/` package)
- **React frontend**: Full web UI for VM lifecycle management

### Tech Stack

- **Backend**: Go 1.26, Beego web framework, XORM (ORM)
- **Frontend**: React, Node.js 18, Yarn
- **Database**: PostgreSQL (default), MySQL, or any XORM-supported DB
- **Auth**: Hanzo IAM (Casdoor Go SDK), Casbin RBAC
- **RDP**: vmd (Guacamole daemon, `ghcr.io/hanzovm/vmd`)
- **Image**: Multi-stage Docker (standard + all-in-one with MariaDB)

### OSS Base

Repo: `hanzoai/vm` (Casvisor fork). Default branch: `master`.

## When to use

- Manage VMs across multiple cloud providers from a single dashboard
- Self-hosted cloud management with IAM integration
- Remote desktop access to VMs via browser (RDP/SSH tunneling)
- Audit trail for VM operations with blockchain records
- Local KVM/libvirt or Proxmox VM management

## Hard requirements

1. **PostgreSQL** (or MySQL) database accessible
2. **Hanzo IAM** (Casdoor) instance for authentication
3. **Go 1.21+** for building from source (or Docker)
4. **Node.js 18** + Yarn for frontend build

## Quick reference

| Item | Value |
|------|-------|
| Default Port | 19000 |
| Go Module | `github.com/hanzoai/vm` |
| Go Version | 1.26 (build: 1.21) |
| Config | `conf/app.conf` |
| Database | PostgreSQL (`hanzo_vm`) |
| IAM Endpoint | `http://iam.hanzo.svc:8000` |
| RDP Daemon | `ghcr.io/hanzovm/vmd` (port 4822) |
| License | Apache-2.0 |
| Repo | `github.com/hanzoai/vm` |
| Default Branch | `master` |

## One-file quickstart

### Build from source

```bash
git clone https://github.com/hanzoai/vm
cd vm

# Backend
go build -o server .

# Frontend
cd web && yarn install --frozen-lockfile && yarn build && cd ..

# Run
./server
```

### Docker

```bash
# Standard (requires external DB)
docker build -t hanzo-vm .
docker run -d -p 19000:19000 hanzo-vm

# All-in-one (includes MariaDB + vmd)
docker build --target ALLINONE -t hanzo-vm-aio .
docker run -d -p 19000:19000 hanzo-vm-aio
```

### Docker Compose

```bash
docker compose up -d
# Starts hanzo-vm + PostgreSQL
# Access at http://localhost:19000
```

### RDP Daemon

```bash
docker run --name vmd -d -p 4822:4822 ghcr.io/hanzovm/vmd
```

## Core Concepts

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        Hanzo VM                             │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐  │
│  │ React UI     │───>│ Beego API    │──>│ Cloud APIs   │  │
│  │ (port 19000) │    │ Controllers  │   │ AWS/Azure/   │  │
│  └──────────────┘    └──────┬───────┘   │ GCP/libvirt  │  │
│                             │           └──────────────┘  │
│                      ┌──────┴───────┐                      │
│                      │ PostgreSQL   │                      │
│                      │ (hanzo_vm)   │                      │
│                      └──────────────┘                      │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐                      │
│  │ Hanzo IAM    │    │ vmd (RDP)    │                      │
│  │ (auth/RBAC)  │    │ (port 4822)  │                      │
│  └──────────────┘    └──────────────┘                      │
└────────────────────────────────────────────────────────────┘
```

### Cloud Providers

| Provider | SDK | Capabilities |
|----------|-----|-------------|
| AWS | `aws-sdk-go-v2/service/ec2` | EC2 instances |
| Azure | `azure-sdk-for-go/sdk/resourcemanager/compute` | Azure VMs |
| GCP | `cloud.google.com/go/compute` | Compute Engine |
| Alibaba | `alibaba-cloud-sdk-go` | ECS instances |
| Tencent | `tencentcloud-sdk-go` | CVM instances |
| libvirt | `digitalocean/go-libvirt` | KVM/QEMU local VMs |
| Proxmox | `luthermonson/go-proxmox` | Proxmox VE |

### Configuration (`conf/app.conf`)

| Key | Default | Description |
|-----|---------|-------------|
| `httpport` | 19000 | Web server port |
| `driverName` | postgres | Database driver |
| `dataSourceName` | `host=localhost port=5432 user=hanzo ...` | DB connection |
| `dbName` | hanzo_vm | Database name |
| `iamEndpoint` | `http://iam.hanzo.svc:8000` | Hanzo IAM URL |
| `clientId` | - | IAM application client ID |
| `clientSecret` | - | IAM application client secret |
| `iamOrganization` | hanzo | IAM organization |
| `iamApplication` | app-hanzo-vm | IAM application name |
| `guacamoleEndpoint` | 127.0.0.1:4822 | vmd RDP endpoint |
| `redisEndpoint` | - | Redis for sessions (optional, falls back to file) |

### Directory Structure

```
vm/
  main.go                  # Entry point (Beego app)
  Dockerfile               # Multi-stage: vmd + Node frontend + Go backend
  docker-compose.yml       # VM + PostgreSQL stack
  build.sh                 # Build script
  go.mod                   # github.com/hanzoai/vm, Go 1.26
  conf/
    app.conf               # Beego configuration
  controllers/
    account.go             # User account management
    asset.go               # Asset CRUD
    machine.go             # VM lifecycle (create/start/stop/delete)
    provider.go            # Cloud provider CRUD
    record.go              # Operation records
    record_chain.go        # Blockchain audit records
    session.go             # Remote desktop sessions (RDP/SSH)
    tunnel.go              # WebSocket tunnel for RDP
    tunnel_handler.go      # Tunnel message handling
    base.go                # Base controller
    util.go                # Controller utilities
  object/
    adapter.go             # XORM database adapter
    asset.go               # Asset model
    machine.go             # Machine model + cloud operations
    machine_cloud.go       # Cloud-specific machine operations
    provider.go            # Provider model
    record.go              # Record model
    record_chain.go        # Blockchain record model
    session.go             # Session model (Guacamole integration)
  authz/
    authz.go               # Casbin RBAC authorization
  chain/
    ...                    # Blockchain audit trail
  routers/
    ...                    # Beego route definitions
  service/
    ...                    # Background services
  task/
    ...                    # Scheduled tasks
  util/
    ...                    # IP geolocation, user-agent parsing
  data/
    ...                    # Static data files
  i18n/
    ...                    # Internationalization
  web/
    ...                    # React frontend (yarn build -> web/build)
  swagger/
    ...                    # API documentation
  k8s/
    ...                    # Kubernetes manifests
  .github/
    ...                    # CI workflows
```

### Module Replacement

The go.mod includes a notable replacement:
```
replace github.com/casdoor/casdoor-go-sdk => github.com/hanzoid/go-sdk v1.44.0
```
This uses Hanzo's fork of the Casdoor SDK for IAM integration.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Login fails | IAM not reachable | Verify `iamEndpoint` in app.conf, check Hanzo IAM is running |
| DB connection error | Wrong dataSourceName | Check PostgreSQL host/port/user in `conf/app.conf` |
| RDP not working | vmd not running | Start vmd container: `docker run -d -p 4822:4822 ghcr.io/hanzovm/vmd` |
| Cloud provider errors | Wrong credentials | Verify cloud provider credentials in provider configuration |
| Frontend 404 | Build not run | Run `cd web && yarn build` to generate `web/build/` |
| Session expired | Redis not configured | Set `redisEndpoint` in app.conf or accept file-based sessions |

## Related Skills

- `hanzo/hanzo-iam.md` - Authentication and RBAC (required dependency)
- `hanzo/hanzo-platform.md` - PaaS deployment
- `hanzo/hanzo-universe.md` - K8s infrastructure
- `hanzo/hanzo-sql.md` - PostgreSQL database

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: vm, cloud, hypervisor, libvirt, proxmox, aws, azure, gcp, rdp, casvisor
**Prerequisites**: PostgreSQL, Hanzo IAM, Go 1.21+, Node.js 18
