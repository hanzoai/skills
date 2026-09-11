---
name: cloud-host
description: Run the Hanzo cloud on your own machine — the local API, the local cluster, ports, keys, and what already exists so you do not rebuild it. Use when self-hosting Hanzo, standing up a development environment, running the cloud from a source checkout, deciding between the local API and a local cluster, wiring KMS or IAM locally, or when someone has written their own launcher scripts because they could not find ours.
---

# Cloud Host

Run the cloud on the machine in front of you.

Everything below is what the tree does today, checked against it. Where a thing
does not work locally this says so rather than describing the intent.

## Start here

```
hanzo host serve
```

That is the local cloud API. It speaks ZAP — the fleet's own transport — and
binds a **unix socket**, not a port on an interface. Where the origin is
loopback it binds `127.0.0.1` and nothing wider. A bound TCP port is not what
"running" means here, which is why the default exposes none.

This command is the answer to "how do I run the cloud locally", and it is the
one an outside adopter did not find: they wrote four launcher scripts, each
minting a master key by hand, and shipped every service unauthenticated on
loopback to get where this gets you.

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
For everything else `hanzo host serve` is lighter and binds less.

`hanzo up <service>` forwards to `hanzo host serve` for one release. Do not
write new scripts against that spelling.

## Ports

One number to know: **3690**. That is what `local` means to `hanzo network` —
`hanzo network use local` names `http://localhost:3690`, and `hanzo up` pins the
same port so the two cannot disagree.

`8080` is the container's default and appears in the cloud README's
`docker run`. It is not the local-network port, and a URL built from one while
the other is running is a connection to nothing. If you are mixing the two, that
is the first thing to check.

The README's quick start is `docker run … :latest`. Prefer a pinned `vX.Y.Z`:
floating tags are against policy everywhere else in this estate, and a `:latest`
you cannot name is a bug you cannot reproduce.

## Listeners, and reading the right file

The cloud's listeners default to loopback, and the reason is specific: the ZAP
listener serves the **identical route surface as HTTP over plaintext TCP**,
including arbitrary process execution. A bare `:9653` binds every interface, and
that port has been reached from another host on the same subnet with no
credential.

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
  something you control and do not build on the bypass.
- **At-rest encryption has been reported as active over plaintext files.** One
  adopter's second boot failed with `sqlcipher_export: no such function`, and
  their notes record every `*.db` carrying the plaintext SQLite header while
  the boot log said encryption was on. **We have not reproduced this.** Treat a
  boot log claiming encryption as unverified until you have read the first
  sixteen bytes of the file yourself.

## Do not rebuild these

Each of these was written from scratch by an adopter who could not find ours.
If you are about to write one, read this row first.

| Before you write | We ship | Why it was missed |
|---|---|---|
| A launcher script | `hanzo host serve` | Not in any README |
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
