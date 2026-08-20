# Security Policy

## Supported versions

Nearstack is pre-1.0 and published from this monorepo under the
`@nearstack-dev/` npm scope. Fixes land on the latest released minor only;
there are no long-term support branches yet.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes       |
| < 0.1   | no        |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report privately through GitHub's private vulnerability reporting:

1. Go to <https://github.com/GhostScientist/nearstack/security/advisories/new>
2. Describe the issue, the affected package, and a reproduction if you have one.

If private reporting is unavailable to you, open a regular issue saying only
that you have a security report and asking for a private channel — no details.

Expect an acknowledgement within 7 days. If a report is accepted, a disclosure
timeline will be agreed with you before anything is published.

## Scope notes for a local-first framework

Nearstack has no server component by design: data lives in IndexedDB and
inference runs in the browser over WebGPU or against a local Ollama endpoint.
That shifts the threat model, so the following are **in scope**:

- Cross-site scripting or prototype pollution reachable through `defineModel`
  stores, query helpers, or the React/Svelte bindings.
- Data leaking across origins, or between models in the shared `nearstack`
  IndexedDB database.
- Prompt or tool-call handling in `@nearstack-dev/ai` that lets untrusted model
  output reach a privileged sink.
- Scaffolding in `@nearstack-dev/cli` that writes attacker-controlled paths
  outside the target directory.
- Supply-chain issues in this repository's own build and release workflows.

The following are **out of scope**:

- The content of model responses. A local model producing wrong or harmful text
  is a model concern, not a framework vulnerability.
- Anything requiring an attacker who already has code execution in the page.
  A local-first framework cannot defend the browser against itself.
- Vulnerabilities in Ollama, `@mlc-ai/web-llm`, or a browser's WebGPU stack.
  Report those upstream — though do say so if Nearstack's use of them makes an
  upstream issue meaningfully worse.
