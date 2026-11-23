---
name: kai
description: "Build on Kai, Hanzo's decision model: it answers typed questions about a state with calibrated probabilities over options you name, and generates no text (POST https://api.hanzo.ai/v1/decisions). Load when code needs a bounded judgment over known options, such as classify, route, gate an action, rank candidates, extract by picking one of the values code found, or verify a claim or field against its source, where the alternative is prompting an LLM and parsing its prose; when writing choice, score or noul questions, confidence thresholds or escalation; when calling Kai over HTTP, from the hanzo-kai (Python) or @hanzo/kai (TypeScript) SDK, or through the kai_* MCP tools; or when moving an integration off TypeSafe's Jev."
license: MIT
---

# Kai

Kai decides; it does not write. Send a `state` (text or JSON) and named questions, and each answer comes back as a probability over options you wrote: which label, whether a statement holds, which level of a scale. No prose, no reasoning, `usage.output_tokens` is 0. Your code reads the numbers and decides what to do.

Kai is for bounded judgments inside a program. Writing text, open questions, arithmetic and fact lookup belong to code, search or a Zen model.

- **Classify**: a choice over the categories the product already has.
- **Route**: a choice over teams, handlers, models or tools.
- **Gate**: allow, ask or deny before an action runs; policy in code has the last word.
- **Rank**: the same score question about each candidate, in one call; sort in code.
- **Extract**: code finds candidate values (dates, amounts, names, spans), a choice picks the right one, code copies it verbatim.
- **Verify**: whether a source supports a claim, whether a field matches its document; fail closed.

## Docs first

The docs outrank this skill: read the pages a task needs before writing code. [llms.txt](https://docs.hanzo.ai/llms.txt) indexes every page, and each page's markdown is at `https://docs.hanzo.ai/llms.mdx/docs/<path>/content.md`. The `/docs/kai` section is new, and some of its pages may not be published yet. [Decisions](https://docs.hanzo.ai/docs/decisions) (the wire) and [Kai](https://docs.hanzo.ai/docs/models/kai) (the model) are live; when a newer page is missing, say so and work from those two.

| Task | Read |
|---|---|
| The map of the section | [overview](https://docs.hanzo.ai/docs/kai) |
| First call: key, curl, Python, TypeScript, MCP | [quickstart](https://docs.hanzo.ai/docs/kai/quickstart) |
| Request, response, limits, errors, price | [decisions](https://docs.hanzo.ai/docs/decisions) |
| What a typed decision is | [concepts](https://docs.hanzo.ai/docs/kai/concepts) |
| What to put in `state` | [state](https://docs.hanzo.ai/docs/kai/state) |
| Choosing and wording questions | [questions](https://docs.hanzo.ai/docs/kai/questions), then [choice](https://docs.hanzo.ai/docs/kai/questions/choice), [score](https://docs.hanzo.ai/docs/kai/questions/score), [noul](https://docs.hanzo.ai/docs/kai/questions/noul), [structure](https://docs.hanzo.ai/docs/kai/questions/structure) |
| Thresholds and escalation | [confidence](https://docs.hanzo.ai/docs/kai/confidence) |
| Breaking a feature into decisions | [build](https://docs.hanzo.ai/docs/kai/build), [patterns](https://docs.hanzo.ai/docs/kai/patterns), [use cases](https://docs.hanzo.ai/docs/kai/use-cases) |
| Worked recipes | [cookbook](https://docs.hanzo.ai/docs/kai/cookbook), [route, triage, gate, loop](https://docs.hanzo.ai/docs/guides/cookbook) |
| Where the model is weak, measured | [limits](https://docs.hanzo.ai/docs/kai/limits) |
| SDKs | [Python](https://docs.hanzo.ai/docs/kai/sdk/python), [TypeScript](https://docs.hanzo.ai/docs/kai/sdk/typescript) |
| This skill and the MCP tools | [agents](https://docs.hanzo.ai/docs/kai/agents) |
| Programs, caches, local runtime | [programs](https://docs.hanzo.ai/docs/kai/programs) |
| Moving from Jev | [migrate from Jev](https://docs.hanzo.ai/docs/guides/migrate/jev) |

## Asking

- **State**: what the judgment depends on and nothing more: the text, plus the facts that change the answer (plan, locale, history, the rule in force) as separate JSON keys. The state and any one question must fit in what the checkpoint reads; a request that does not is refused with 422 `state_too_long`, naming the tokens needed and the limit, and nothing is cut silently. Leave secrets and data you may not share out of it.
- **Instructions**: optional, and worth writing on every question: they are the question Kai answers. Kai never sees the question's name, so the whole question goes in `instructions`.
- **choice**: exactly one of two or more options. `criteria` maps each label to a description of what sets it apart; keep the options mutually exclusive. `/v1/decisions` sets no label cap: a wide choice is narrowed by retrieval over every option, then scored. Each option is read whole up to 512 tokens; a longer one is a 422. Add an `other` or `none` label for cases yours do not cover; without one, Kai must choose among them.
- **score**: a level on an ordered scale. `criteria` lists the levels lowest first, every one described, none null. `score` is the expected level; the most probable level is the largest entry of `probabilities`.
- **noul**: the probability that a statement holds. Write it as a statement and describe both sides in `criteria` (`true` and `false`), or better, ask a `choice` whose labels name the outcomes (`refund`, `other`), never `yes` and `no`. The current checkpoint answers some plain yes/no questions badly, so measure the form you pick.
- **Multi-label**: when several labels can be true together, give each its own noul.
- **Fan out**: put every question about the same state in one request, up to 100. Kai answers each against the state alone, never against another answer, so a question that needs an earlier answer waits for a second call (fetch the evidence it names, then ask). The bill counts the state once and each question once, so an added question costs only its own tokens.

## Acting on answers

- `confidence` on a choice or score is (n·p_max − 1)/(n − 1): 0 when every option is equally likely, 1 when one holds all the mass, so one threshold means the same over 3 options or 30. A noul's `noul` is P(true) and its `confidence` is |2p − 1|; read 0.5 as a coin flip.
- Set thresholds on the user's own labeled cases: take the one that gives the accuracy the action needs, and count how many cases it sends elsewhere. Act above it; below it, escalate to a human reviewer or a Zen model.
- Policy stays in code. Permissions, amounts, compliance and hard limits are deterministic checks. A Kai answer may tighten an outcome (ask, deny); it never loosens what policy forbids.
- Before shipping, run representative cases: real states, real wording, the awkward ones. Print what Kai answered and change the question or the state, never the numbers. Accuracy differs by question kind; the limits page has the measurements.
- Read answers by question name, never by position, and record each decision's `id`, `state_hash` and `routing.checkpoint` beside the action taken.
- Keys stay server-side. A decision takes a secret `sk-` key; a publishable `pk-` key gets 403. Never ship one to a browser.

## Calling

HTTP works today:

```bash
curl https://api.hanzo.ai/v1/decisions \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "kai",
       "state": {"ticket": "The export button throws a 500 error since this morning."},
       "questions": {"team": {"type": "choice",
         "instructions": "Which team should handle this ticket?",
         "criteria": {"billing": "charges, invoices, refunds",
                      "tech": "bugs, errors, outages",
                      "other": "anything else"}}}}'
```

Read `answers.team.choice`, its `confidence`, and `probabilities` over every label. Errors: 400 malformed JSON or unknown model, 401 bad key, 402 no plan or balance, 403 publishable key, 422 a malformed question or a state too long, 429 and 529 (wait `Retry-After`, then retry).

The SDKs and MCP tools are new. If a package or tool is not published where you are, use HTTP. Both SDKs read `HANZO_API_KEY`.

```python
from hanzo_kai import Kai, Choice  # uv add hanzo-kai

teams = {"billing": "charges, invoices, refunds", "tech": "bugs, errors, outages", "other": "anything else"}
with Kai() as kai:
    d = kai.decide(state={"ticket": "The export button throws a 500 error."},
                   questions={"team": Choice(instructions="Which team should handle this ticket?", criteria=teams)})
print(d.choices["team"].choice)
```

```ts
import { Kai, choice } from "@hanzo/kai"; // npm i @hanzo/kai; server-side only

const teams = { billing: "charges, invoices, refunds", tech: "bugs, errors, outages", other: "anything else" };
const d = await new Kai().decide({
  state: { ticket: "The export button throws a 500 error." },
  questions: { team: choice("Which team should handle this ticket?", teams) },
});
console.log(d.answers.team.choice); // typed "billing" | "tech" | "other"
```

MCP: the Hanzo MCP tools `kai_decide`, `kai_choice`, `kai_score` and `kai_noul`.

## From Jev

A Jev integration keeps its `state`, its questions and the shape of its answers, and moves one of two ways:

1. On TypeSafe's own SDK, change two settings, `TYPESAFE_BASE_URL=https://api.hanzo.ai` and `TYPESAFE_DEFAULT_MODEL=kai`, with a Hanzo `sk-` key in `TYPESAFE_API_KEY`. Name `kai`: no Jev model id becomes Kai.
2. Or switch to `hanzo-kai` or `@hanzo/kai`, or call `POST https://api.hanzo.ai/v1/decisions` with `"model": "kai"`.

Either way, describe every score level (a null level is refused), then re-run the labeled cases on Kai and set thresholds again; Jev's do not carry over.

Price: 50% lower input-token list price ($0.021 vs $0.042 per 1M). On the frozen harness Kai a7 is more accurate than Jev on 11 of 12 suites, and Jev leads jailbreak 0.940 to 0.920 ([results](https://github.com/hanzoai/benchmarks/tree/c61361fedc5882b0d83b2f5d5a6a35c56ffeb3c3/decision/results/kai-a7)). The [Jev guide](https://docs.hanzo.ai/docs/guides/migrate/jev) has the field-by-field differences.

## Install

```bash
npx skills add https://hanzoskills.com/kai.md
```

With GitHub access to hanzoai/skills: `npx skills add hanzoai/skills --skill kai`, or in Claude Code `claude plugin marketplace add hanzoai/skills`, then `claude plugin install kai@hanzoai-skills`.
