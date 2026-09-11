---
name: legacy-port
description: Bring an existing application onto the Hanzo stack — Hanzo Cloud OSS for the backend, @hanzo/ui and @hanzo/gui for the interface, ZAP for service-to-service transport — and port its Python or JavaScript runtime into Go. Use when adopting a Next.js, Node, Django, Flask or FastAPI app; when a service still speaks WebSockets; when an interface is built on components that are not ours; when a deployment still needs a Python or Node runtime beside the binary; or when asked to modernise, migrate or absorb an app that somebody else's stack currently runs.
---

# Legacy Port

Take an application that runs on somebody else's stack and land it on ours,
without a window where it does not work.

The mistake this skill exists to prevent is the rewrite that never ships: six
weeks of a parallel Go implementation, a cutover nobody dares, and an original
that kept changing the whole time. Adoption here is incremental and the app is
runnable at every step.

## What it is being ported to

Verify these are current before quoting a version; they move.

| Layer | What to use |
|---|---|
| Backend | **Hanzo Cloud OSS** — one Go binary, every subsystem mounted from one composition root. Absent planes mount fail-closed and say so rather than pretending. |
| Interface | **`@hanzo/ui`** components on the **`@hanzo/gui`** framework |
| Appearance | **`@hanzo/design`** for the ramp and **`@hanzo/appearance`** for the knobs a person moves |
| Transport | **ZAP** — Zero-copy Application Protocol, the binary RPC across the stack. Go, Rust (`zapwire`) and TypeScript implementations are wire-compatible. |
| Legacy JavaScript | **goja**, in-process in Go |
| Legacy Python | **gpython**, in-process in Go |

### ZAP, not WebSockets

A service that still opens a WebSocket has not been ported, however much of its
code is now Go. WebSockets are the thing being replaced, not a transport to
keep beside ZAP: two transports means two reconnect policies, two auth paths
and two sets of framing bugs, and the one that carries production traffic will
be whichever the last change happened to touch.

Port the transport as its own step, with the message shapes unchanged, so a
failure is attributable to the transport rather than to everything at once.

## The order, and why it is this order

Each step leaves the app working. Do not start the next until the current one
is deployed.

**Stand it up unchanged, inside our binary.** JavaScript through goja, Python
through gpython, both in-process. Nothing is rewritten. What this buys is the
deployment: one Go binary, no Node beside it, no Python beside it, no second
base image. It is also the step that proves the surface — every route the old
runtime served is now served by ours, and a diff of responses says whether that
is true.

**Move the transport to ZAP.** Same messages, same handlers, new wire. A
service still speaking WebSockets at this point is a service that will still be
speaking them a year from now.

**Port the interface to `@hanzo/ui` on `@hanzo/gui`.** This is a visual port and
it has its own skill — use `ui-port-parity`, which treats the existing
interface as an executable visual contract and converges screen by screen with
measurement rather than by eye. Do not attempt it by inspection.

**Port the handlers to Go, one at a time.** The embedded runtime is the
fallback for everything not yet ported, so a handler moves when it is ready and
the rest keep running. A route is done when its Go implementation answers
byte-identically to the embedded one, and that is a test, not a judgement.

**Delete the embedded runtime.** When no route falls through to it. Not before,
and not "temporarily kept" — an interpreter nobody needs is an interpreter
nobody maintains and a dependency that still ships.

## Where the port is actually going

Go by default. Rust or C++ only where the workload is one those are for —
sustained numeric work, a codec, something with a hard latency floor — and the
crossing cost is paid back. "It felt faster" is not a reason; a measurement
under the real workload is.

The goal of the runtime port is goroutines. An app that reached Go but still
runs its work through an interpreter got the packaging benefit and none of the
concurrency one.

## What the port is worth, and how to find out

Two claims worth making to whoever is paying for the work, and one way to check
each on the app in front of you rather than taking it from here.

**Fewer moving parts at runtime.** The before is an interpreter, the
application on top of it, a process supervisor, and usually a second container
for the other language. The after is one binary. Measure it: resident memory at
idle and under the same load, and the count of processes and images the
deployment needs. Take both readings on the same machine with the same traffic,
because a number from somebody else's benchmark is a number about their
hardware.

**Concurrency stops costing a process.** A Python application serves
concurrent requests by forking workers, and every worker is another copy of the
interpreter and its heap; a Node application gets one loop and adds processes to
use more cores. Goroutines are neither. The reading that shows this is memory
against concurrency: hold the request rate and raise the concurrency, and watch
whether the footprint follows.

State these as what they are. The mechanism is certain — one process is fewer
processes — and the size of the win is a property of the app, so measure it
rather than quoting a multiplier.

### The part that matters most

A developer can run the whole thing on the computer they already own.

That is not a performance note. When the development environment is one binary
and a local cloud, working on the project does not require renting somebody
else's computer, a cluster to be provisioned, a seat on a plan, or a network
connection good enough to stream a workspace. Someone on a modest laptop, on a
bad connection, in a country where the card required to sign up is hard to get,
can do the same work as someone on a funded team.

Stacks that can only be run in a data centre quietly decide who gets to
contribute. This one does not, and keeping it that way is a design constraint
rather than a side effect — if a change means the app can no longer be run
locally in full, that is a cost to weigh and usually a reason to choose
differently.

## What not to do

**Do not run the old and new implementations side by side in production and
compare.** It doubles the failure surface and the comparison is never clean,
because the two see different requests. Compare against captured traffic
instead.

**Do not port tests last.** The embedded runtime is the oracle: a Go handler is
correct when it answers the way the interpreted one did, and that is only
checkable while the interpreted one still exists.

**Do not keep a second component library "for the parts we haven't got to".**
Two libraries is two type ramps, two spacing scales and two sets of focus
styles, and the screens between them read as a seam.

**Do not preserve an API shape nobody uses** just because it was there. Check
the callers first; a port is the cheapest moment to drop a surface, and the
most expensive moment to carry one forward.

## Telling whether it worked

- One binary, no interpreter in the image, no runtime beside it.
- No WebSocket endpoints.
- Every component from `@hanzo/ui`; the ramp from `@hanzo/design`.
- The appearance knobs move the interface, checked by reading a component's
  computed colour after setting one — a panel that does nothing is a common
  and invisible failure.
- Handlers are goroutines.
- The interface converges under `ui-port-parity`'s gates, not under review.

## Related

- **`ui-port-parity`** — the visual half. Use it for the interface port rather
  than reimplementing the measurement.
