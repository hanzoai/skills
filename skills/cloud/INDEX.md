# Cloud Skills

## Category Overview

**Total Skills**: 13 (+2 resource packs)
**Category**: cloud

## Hanzo Reality (read first)

For Hanzo's own cloud, NONE of the vendor skills below apply — the
architecture is the unified one-binary ZAP-native plugin cloud (HIP-0106/
0114/0116/0117): `hanzoai/cloud` on zip, ZAP-only transport, per-tenant
SQLite, Quasar + zapdb, deployed via `cloud serve` / `cloud cluster init` /
helm. AWS/GCP here are for integrating with customer infrastructure (visor
autoscales fleets across ANY provider — BYO provider).

Load: `cat skills/hanzo-cloud-architecture/SKILL.md`

## Skills in This Category

### AWS (`aws/`)

- `aws/aws-api-gateway.md` — API Gateway
- `aws/aws-databases.md` — RDS, DynamoDB
- `aws/aws-ec2-compute.md` — EC2 compute
- `aws/aws-iam-security.md` — IAM and security
- `aws/aws-lambda-functions.md` — Lambda functions
- `aws/aws-networking.md` — VPC, networking
- `aws/aws-storage.md` — S3, EBS

### GCP (`gcp/`)

- `gcp/gcp-compute.md` — Compute Engine
- `gcp/gcp-databases.md` — Cloud SQL, Firestore
- `gcp/gcp-iam-security.md` — IAM and security
- `gcp/gcp-networking.md` — VPC, networking
- `gcp/gcp-serverless.md` — Cloud Run, Functions
- `gcp/gcp-storage.md` — GCS

### Kubernetes

- `../cloud-kubernetes-deployment.md` — generic K8s deployment (with the
  Hanzo-reality preamble)
- `kubernetes-deployment/resources/` — reference pack
- `aws-lambda-deployment/resources/` — reference pack

## Related

- `skills/hanzo-cloud-architecture/SKILL.md` — the Hanzo cloud itself
- `skills/deployment/INDEX.md` — third-party PaaS deployment
