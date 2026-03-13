# Hanzo Form.js - Automatic Form Validation and Submission

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-checkout.md`, `hanzo/hanzo-commerce.md`

## Overview

Form.js is a lightweight **browser library for automatic form validation and submission**. It intercepts native HTML form submit events, emits lifecycle events (init, submit, done), and provides an XHR helper for posting form data as JSON. Zero dependencies beyond `little-emitter`. Written in CoffeeScript, ships as both CommonJS and ES module bundles. Designed as a companion to Hanzo's mailing list and lead capture forms.

### Tech Stack

- **Language**: CoffeeScript (compiled to JS)
- **Runtime dependency**: `little-emitter` (tiny event emitter, ~150 bytes)
- **Build**: Cakefile (via `shortcake` + `cake-bundle` using `requisite`)
- **Output**: CommonJS (`lib/form.cjs.js`) + ES module (`lib/form.es.js`)
- **Tests**: Mocha + Chai + Selenium WebDriver + PhantomJS
- **CI**: Travis CI
- **npm package**: `form.js` v0.2.0

### OSS Base

Repo: `github.com/hanzoai/form.js` (branch: `master`, 16 stars).

## When to use

- Intercepting HTML form submissions to send data via XHR/AJAX instead of page reload
- Adding custom validation logic before form submit
- Capturing mailing list signups or lead generation forms
- Lightweight form handling without a full framework (no jQuery required)

## Hard requirements

1. **Browser environment** with DOM (not Node.js server-side)
2. **HTML `<form>` element** in the page -- the library auto-discovers the last form on the page at load time
3. **Script tag must be inside or after the form** (uses `getElementsByTagName` to find the last `<script>` and `<form>`)

## Quick reference

| Item | Value |
|------|-------|
| npm package | `form.js` |
| Version | 0.2.0 |
| Default branch | `master` |
| License | BSD-3-Clause |
| Language | CoffeeScript |
| CJS entry | `lib/form.cjs.js` |
| ES entry | `lib/form.es.js` |
| Global | `window.Inform` |
| Runtime dep | `little-emitter` only |
| Repo | `github.com/hanzoai/form.js` |

## One-file quickstart

### Script tag integration

```html
<form action="/subscribe" method="POST">
  <input type="email" name="email" placeholder="Email" required />
  <input type="text" name="firstName" placeholder="First Name" />
  <button type="submit">Subscribe</button>

  <script src="https://unpkg.com/form.js@0.2.0/lib/form.cjs.js"></script>
  <script>
    var Inform = window.Inform

    // Listen for form submission (prevents default, gives you control)
    Inform.on('inform-submit', function(done, event) {
      var formData = new FormData(event.target)
      var data = {}
      formData.forEach(function(value, key) { data[key] = value })

      fetch('/api/subscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      })
      .then(function() { done() })  // Call done() to fire the 'done' event
      .catch(function(err) { console.error(err) })
    })

    // Listen for successful submission
    Inform.on('inform-done', function(event) {
      alert('Thanks for subscribing!')
    })
  </script>
</form>
```

### npm install

```bash
npm install form.js
```

```javascript
// ES module
import Inform from 'form.js'

Inform.on('inform-submit', function(done, event) {
  // Handle submission
  done()
})
```

## Core Concepts

### Architecture

```
┌────────────────────────────────────────┐
│  HTML Page                              │
│                                         │
│  <form>                                 │
│    <input name="email" />               │
│    <button type="submit">Go</button>    │
│                                         │
│    <script src="form.js"></script>       │
│  </form>                                │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  Inform (EventEmitter)           │   │
│  │                                  │   │
│  │  Events:                         │   │
│  │  1. init-inform-script  ─────>  script tag ref    │
│  │  2. init-inform-form    ─────>  form element ref  │
│  │  3. init-inform-inputs  ─────>  input elements    │
│  │  4. init-inform-submits ─────>  submit buttons    │
│  │  5. inform-submit       ─────>  (done, event)     │
│  │  6. inform-done         ─────>  completion         │
│  └──────────────────────────────────┘   │
└────────────────────────────────────────┘
```

### Event Lifecycle

1. **DOM ready** -- Library auto-initializes, finds the last `<form>` and `<script>` tags
2. **`init-inform-script`** -- Fires with the script element reference
3. **`init-inform-form`** -- Fires with the form element reference
4. **`init-inform-inputs`** -- Fires with array of input/select elements (excludes submit buttons)
5. **`init-inform-submits`** -- Fires with array of submit buttons (input[type=submit] + button[type=submit])
6. **`inform-submit`** -- Fires on form submit with `(done, event)`. Call `done()` to proceed to `inform-done`. The default form submit is prevented.
7. **`inform-done`** -- Fires after `done()` is called, re-dispatches the native submit event

### Global API

The library exposes `window.Inform` as an event emitter with standard methods:

- `Inform.on(event, callback)` -- Listen for an event
- `Inform.off(event, callback)` -- Remove a listener
- `Inform.emit(event, ...args)` -- Emit an event
- `Inform.Events` -- Map of event name constants
- `Inform.events` -- Same map (alias)

### XHR Helper (src/xhr.coffee)

A minimal XMLHttpRequest wrapper with a `post(url, headers, payload, callback)` method. Used internally for legacy AJAX form submission. Supports IE via `ActiveXObject` fallback.

### Source Files

| File | Purpose |
|------|---------|
| `src/index.coffee` | Main library: auto-init, event lifecycle, form interception |
| `src/xhr.coffee` | XHR POST helper class |

## Directory structure

```
form.js/
  package.json          # npm config, v0.2.0
  Cakefile              # Build/test tasks (shortcake + cake-bundle)
  README.md             # Minimal README
  LICENSE               # BSD-3-Clause
  .travis.yml           # Travis CI config
  src/
    index.coffee        # Main library (event emitter + form interception)
    xhr.coffee          # XMLHttpRequest POST helper
  test/
    test.coffee         # Selenium WebDriver integration tests
    test.html           # Test harness HTML page
    inform.js           # Built test bundle
    util.coffee         # Test utilities
    ci-config.coffee    # Cross-browser Sauce Labs config
    jquery-1.11.3.min.js  # jQuery for test assertions
  lib/                  # Built output (generated)
    form.cjs.js         # CommonJS bundle
    form.es.js          # ES module bundle
```

## Build and Test

```bash
# Install dependencies
npm install

# Build (generates lib/form.cjs.js and lib/form.es.js)
cake build

# Run tests (starts static server + Selenium + Mocha)
cake test

# Clean build artifacts
cake clean
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `window.Inform` is undefined | Script loaded before form.js bundle | Ensure form.js script tag is loaded |
| Form submits normally (page reload) | Script tag not inside/after the form | Place `<script>` inside the `<form>` or after it |
| `inform-submit` not firing | Form has no submit button | Add `<button type="submit">` or `<input type="submit">` |
| Events fire twice | `DOMContentLoaded` and `load` both trigger init | Library guards against double-init internally |
| `done()` not working | Not calling the first argument of `inform-submit` callback | Signature is `function(done, event)` -- call `done()` to proceed |

## Related Skills

- `hanzo/hanzo-checkout.md` - Full checkout widget (uses similar patterns)
- `hanzo/hanzo-commerce.md` - Hanzo Commerce frontend

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: forms, validation, submission, lead-capture, mailing-list
**Prerequisites**: Browser environment, HTML form element
