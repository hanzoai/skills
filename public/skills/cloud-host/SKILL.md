---
name: cloud-host
description: Run the Hanzo cloud on your own machine — the local API, the local cluster, ports, keys, and what already exists so you do not rebuild it. Use when self-hosting Hanzo, standing up a development environment, running the cloud from a source checkout, deciding between the local API and a local cluster, wiring KMS or IAM locally, or when someone has written their own launcher scripts because they could not find ours.
---

# Cloud Host

Run the cloud on the machine in front of you.

Everything below is what the tree does today, checked against it. Where a thing
does not work locally this says so rather than describing the intent.

## Start here

From a checkout of `hanzoai/cloud`:

```
make build && CLOUD_DEV_UNENCRYPTED=1 CLOUD_LISTEN=127.0.0.1:8080 CLOUD_DATA_DIR=.dev/data ./cloud
```

That serves the API and the embedded console on `http://127.0.0.1:8080`, and
`/v1/openapi.json` lists every route the process serves. The README's `make dev` is meant to be this command; today it stops with
`cek: this build cannot encrypt a store` (see *What does not work locally yet*).

The CLI has two local modes, and neither builds anything:

- **`hanzo host serve [service]`** runs a cloud binary you already have, in the
  foreground: `HANZO_CLOUD_BIN`, else `hanzo-cloud` or `cloud` on `PATH`.
- **`hanzo host start`** runs the host as a daemon. After
  `hanzo network use local`, every cloud command reaches it over a unix socket in
  the CLI's state directory, with no TCP port. It starts `HANZO_HOST_BIN`, default
  `~/work/hanzo/cloud/bin/host`, which `make host` builds; `hanzoai/cloud` has no
  `make host` target yet.

An outside adopter who found none of this wrote four launcher scripts, each
minting a master key by hand, and shipped every service unauthenticated on
loopback.

### When you want the cluster shape instead

```
hanzo up
```

Boots k3s inside a microVM and deploys the cloud into it as the cluster's first
workload, then hands back a kubeconfig. First boot downloads k3s once, in the
foreground, so the wait is visible; later boots start from a disk checkpoint.
`hanzo up down` is one signal — the supervisor holds the VM's stdin as a leash,
so the guest stops when the supervisor does.

Use this when you are testing something that depends on being in a cluster.
For everything else one process is lighter and binds less.

`hanzo up <service>` forwards to `hanzo host serve` for one release. Do not
write new scripts against that spelling.

## Ports

One number to know: **3690**. That is what `local` means to `hanzo network` —
`hanzo network use local` names `http://localhost:3690`, and `hanzo up` pins the
same port so the two cannot disagree.

`8080` is where the cloud serves when you start it from a checkout. It is not
the local-network port, and a URL built from one while the other is running is a
connection to nothing. If you are mixing the two, that is the first thing to
check.

## Listeners, and reading the right file

The ZAP and health listeners default to loopback, `127.0.0.1:9653` and
`127.0.0.1:9090`, and the reason is specific: the ZAP listener serves the
**identical route surface as HTTP over plaintext TCP**, including arbitrary
process execution. A bare `:9653` binds every interface, and that port has been
reached from another host on the same subnet with no credential.

The HTTP listener is the exception. With neither `CLOUD_LISTEN` nor `PORT` set
it binds `:8080`, every interface, which is why the command in *Start here* sets
`CLOUD_LISTEN`.

Clusters set `CLOUD_ZAP_LISTEN` explicitly and have a network policy in front,
so the safe default costs them nothing. On a laptop you have neither, which is
the case the default has to be safe for.

Two things to know when you check this yourself:

- The environment wins over the flag default. `CLOUD_ZAP_LISTEN`,
  `CLOUD_LISTEN`, `CLOUD_HEALTH_LISTEN` and `CLOUD_ADMIN_LISTEN` are the knobs.
- `CLOUD_ADMIN_LISTEN` still defaults to `:8081`, every interface. Set it to
  loopback yourself until that changes.

**Do not conclude a default is safe from one file.** The flag default in
`cmd/cloud/main.go` and the config default in `config.go` disagreed for a
while, and the flag is what runs. An adopter read the safe file and wrote
"safe" into their deploy notes while the process listened on every interface.
Check what is bound, not what is written:

```
lsof -nP -iTCP -sTCP:LISTEN | grep -E 'cloud|base'
```

When you write a test for this, include a **positive control** — show the probe
can see a listener that is there before you assert one is not. A probe that
sees nothing because it is broken passes every exposure test.

## What does not work locally yet

Say so in your notes rather than working around it quietly.

- **KMS needs an IAM principal, and there is no IAM locally.** `/v1/kms` is
  guarded, correctly, and the guard has nothing to check against on a laptop.
  The in-process client skips the guard, which is why reaching for it is
  tempting and why doing so means your local path and your deployed path are
  different code. Until a local IAM superuser is seeded, keep secrets in
  something you control and do not build on the bypass. With no master key
  set, as in *Start here*, secret operations fail closed as well.
- **The default build cannot encrypt its stores.** On `main`, `make dev` stops
  with `cek: this build cannot encrypt a store — a keyed database was written
  with a plaintext header`. `CLOUD_DEV_UNENCRYPTED=1` does not get past it under
  `make dev`, which always passes a key; it applies only when no key is set,
  which is how *Start here* runs, and the files are then plaintext. One adopter
  also saw `sqlcipher_export: no such function` on a second boot, and every
  `*.db` carrying the plaintext SQLite header while the boot log said encryption
  was on. Trust a claim of encryption only after reading the first sixteen bytes
  of the file.

## Do not rebuild these

Each of these was written from scratch by an adopter who could not find ours.
If you are about to write one, read this row first.

| Before you write | We ship | Why it was missed |
|---|---|---|
| A launcher script | the command in *Start here* | The skills described only the hosted cloud |
| A KMS CLI | `/v1/kms`, `hanzo kms` | Needs IAM, absent locally |
| An SDK generator | `zip/cmd/zipgen` | Not in the zip README |
| An LLM client | `python-sdk`, `hanzo-agent` | The packages failed to import |
| A CDP relay or browser MCP | the bot relay, `extension/packages/browser` | Relay lives inside the gateway; no README |
| Compose or Terraform for one box | nothing, honestly | There is no self-host guide yet |

The last row is the real gap. If you build a single-box deploy, it is worth
upstreaming.

## If something here is wrong

Open a pull request. This stack is open source and the team reviews, helps
finish and merges — you do not need permission first, and you do not need to be
on the team. Report a security issue privately to the repository's security
contact rather than in a public issue.

Write down what you measured rather than what you assumed. The findings that
changed this page came from someone who recorded which command produced which
output, and separated "actually executed" from "written but not executed" in
their own runbooks.

## Related

- **`app-port`** — bringing an existing app onto this stack
- **`ui-port`** — making a rewritten interface match the one it replaces
