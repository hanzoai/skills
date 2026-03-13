# Hanzo Checkout.js - Embeddable Payment Widget

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-commerce.md`, `hanzo/hanzo-commerce-api.md`

## Overview

Checkout.js is an **embeddable checkout widget** for taking payments and pre-orders on any website. It renders a modal overlay with payment collection (Stripe and PayPal), shipping address, tax/shipping rate calculation, promo codes, and a thank-you screen with social sharing. Written in CoffeeScript using Riot.js 3 for UI components and Jade templates. Communicates with the Hanzo Commerce API via `hanzo.js` (now `crowdstart.js`).

### Tech Stack

- **Language**: CoffeeScript (compiled to JS)
- **UI Framework**: Riot.js 3.0.7
- **Templates**: Jade (Pug predecessor)
- **Styles**: Stylus (compiled to CSS)
- **Build**: Cakefile (CoffeeScript-based task runner via `shortcake`)
- **Bundler**: `requisite` (CommonJS bundler)
- **Tests**: Mocha + Chai + Selenium WebDriver + PhantomJS
- **CI**: Travis CI
- **npm package**: `checkout.js` v2.1.21

### OSS Base

Repo: `github.com/hanzoai/checkout.js` (branch: `master`, 96 stars).

## When to use

- Adding a drop-in checkout modal to any HTML page
- Collecting payments via Stripe or PayPal without building a checkout flow
- Crowdfunding or pre-order campaigns with product management via Hanzo API
- Legacy Hanzo Commerce API integrations

## Hard requirements

1. **Hanzo API key** (access token for the Commerce API)
2. **Products** created in the Hanzo Commerce API (referenced by ID or slug)
3. **Stripe and/or PayPal** credentials configured in Hanzo dashboard
4. **Browser environment** (not a Node.js server-side library)

## Quick reference

| Item | Value |
|------|-------|
| npm package | `checkout.js` |
| Version | 2.1.21 |
| Default branch | `master` |
| License | BSD-3-Clause |
| Language | CoffeeScript |
| UI framework | Riot.js 3.0.7 |
| API client | `hanzo.js` (aliased as `crowdstart.js`) |
| CDN | `cdn.rawgit.com/hanzoai/checkout.js/v2.1.21/checkout.min.js` |
| Repo | `github.com/hanzoai/checkout.js` |

## One-file quickstart

### Script tag integration

```html
<a class="btn" href="#checkout">Buy Now</a>

<script src="https://cdn.rawgit.com/hanzoai/checkout.js/v2.1.21/checkout.min.js"></script>
<script>
  var Checkout = window.Crowdstart.Checkout

  var checkout = new Checkout('your-hanzo-api-key', {
    config: {
      currency: 'usd',
      processors: { stripe: true, paypal: false },
      showPromoCode: true,
      termsUrl: 'https://example.com/terms'
    },
    order: {
      currency: 'usd',
      items: []
    },
    user: {
      email: '',
      firstName: '',
      lastName: ''
    },
    thankyou: {
      header: 'Thank You!',
      body: 'Check your email for the order confirmation.',
      twitter: 'hanaboroai',
      facebook: 'hanzoai'
    },
    theme: {
      background: 'white',
      calloutBackground: '#27AE60',
      fontFamily: "'Helvetica Neue', Helvetica, Arial, sans-serif",
      borderRadius: 5
    }
  })

  // Add items by product ID or slug
  checkout.setItem('84cRXBYs9jX7w', 1)
  checkout.setItem('doge-shirt', 2)

  // Open the widget
  document.querySelector('.btn').addEventListener('click', function(e) {
    e.preventDefault()
    checkout.open()
  })
</script>
```

### npm install

```bash
npm install checkout.js
```

```javascript
var Checkout = require('checkout.js')

var checkout = new Checkout('your-api-key', {
  config: { currency: 'usd' }
})
checkout.setItem('product-slug', 1)
checkout.open()
```

## Core Concepts

### Architecture

```
┌─────────────────────────────────────────────────┐
│  Browser Page                                    │
│                                                  │
│  ┌─────────────┐    ┌────────────────────────┐  │
│  │ Buy Button  │───>│  Checkout Modal (Riot)  │  │
│  │ href=#checkout   │                        │  │
│  └─────────────┘    │  ┌──────────────────┐  │  │
│                     │  │ Screen Manager   │  │  │
│                     │  │  - Payment       │  │  │
│                     │  │  - Shipping      │  │  │
│                     │  │  - Thank You     │  │  │
│                     │  └──────────────────┘  │  │
│                     └───────────┬────────────┘  │
│                                 │                │
│                     ┌───────────▼────────────┐  │
│                     │  hanzo.js API Client    │  │
│                     │  (Crowdstart SDK)       │  │
│                     └───────────┬────────────┘  │
│                                 │                │
└─────────────────────────────────┼────────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │  Hanzo Commerce API     │
                      │  (api.hanzo.ai)         │
                      └────────────────────────┘
```

### Checkout Flow (Screen Script)

The widget follows a three-step flow defined by `script: ['payment', 'shipping', 'thankyou']`:

1. **Payment Screen** - Credit card entry (Stripe) or PayPal button, email, name
2. **Shipping Screen** - Address form with country/state selectors, tax and shipping rate calculation
3. **Thank You Screen** - Confirmation message with social sharing buttons (Facebook, Twitter, Google+, Pinterest, email)

### Key Classes

- **`Checkout`** - Main entry point. Instantiate with API key + options. Methods: `open()`, `setItem(id, qty)`, `setUser(user)`, `update()`, `on()`, `off()`, `one()`
- **`Widget`** - Root Riot.js tag that mounts the modal
- **`ScreenManager`** - Controls navigation between payment/shipping/thankyou screens
- **`Events`** - Event bus (via `crowdcontrol`) for inter-component communication

### Configuration Objects

- **`config`** - Payment processors, currency, promo code toggle, terms URL, call-to-action text
- **`taxRates`** - Array of `{taxRate, city, state, country}` objects, matched in order (first match wins)
- **`shippingRates`** - Array of `{shippingRate, city, state, country}` objects, same matching logic
- **`theme`** - Colors (background, light, dark, error, progress, spinner), font family, border radius
- **`thankyou`** - Header, body, social sharing handles (facebook, twitter, googlePlus, pinterest, email)
- **`analytics`** - Pixel tracking URLs
- **`test`** - Custom API endpoint, PayPal sandbox toggle

### Dependencies

| Package | Purpose |
|---------|---------|
| `hanzo.js` (crowdstart.js) | Hanzo Commerce API client |
| `riot` 3.0.7 | Reactive UI components |
| `crowdcontrol` | Event system for Riot |
| `card` 2.1.1 | Credit card input formatting |
| `style-inject` | Runtime CSS injection |
| `bebop` | Dev server with live reload |

## Directory structure

```
checkout.js/
  checkout.js           # Built bundle (1.1MB)
  checkout.min.js       # Minified bundle (245KB)
  checkout.css          # Built stylesheet (45KB)
  package.json          # npm config, v2.1.21
  Cakefile              # Build/test tasks
  bebop.coffee          # Dev server config
  src/
    index.coffee        # Main Checkout class
    events.coffee       # Event definitions
    data/
      countries.coffee  # Country list for shipping
      currencies.coffee # Supported currencies
      states.coffee     # US state list
    utils/
      analytics.coffee  # Tracking pixel helper
      country.coffee    # Country lookup
      currency.coffee   # Currency formatting
      input.coffee      # Input validation
      theme.coffee      # Dynamic theme injection
    views/
      widget.coffee     # Root Riot tag
      modal.coffee      # Modal overlay
      header.coffee     # Widget header
      screenmanager.coffee  # Screen navigation
      invoice.coffee    # Order summary
      lineitem.coffee   # Single line item
      confirm.coffee    # Order confirmation
      promo.coffee      # Promo code input
      tabs.coffee       # Payment method tabs
      controls/         # Form input controls
      screens/
        screen.coffee   # Base screen class
        payment.coffee  # Credit card / PayPal entry
        shipping.coffee # Address form
        thankyou.coffee # Post-purchase screen
        choose.coffee   # Payment method selection
        paypal.coffee   # PayPal-specific flow
  templates/            # Jade templates for each view
  css/                  # Stylus source files
  vendor/               # Vendored JS/CSS (select2)
  examples/
    basic/              # Basic HTML integration example
  test/
    test.coffee         # Selenium WebDriver tests
    widget.html         # Test harness HTML
    ci-config.coffee    # Cross-browser CI config
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Widget does not open | Button href not `#checkout` or `open()` not called | Use `checkout.open()` or set button `href="#checkout"` |
| Products not loading | Invalid product ID/slug | Verify product exists in Hanzo Commerce API |
| Stripe not working | Stripe processor disabled in config | Set `config.processors.stripe: true` |
| PayPal sandbox | Test mode not enabled | Set `test.paypal: true` in options |
| Old Node.js version | Requires Node >= 4.0.0 | Upgrade Node.js |
| Build fails | Missing CoffeeScript | Run `npm install` to get devDependencies |

## Related Skills

- `hanzo/hanzo-commerce.md` - Hanzo Commerce frontend
- `hanzo/hanzo-commerce-api.md` - Hanzo Commerce API backend
- `hanzo/hanzo-form.md` - Form validation/submission library (same era)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: checkout, payments, ecommerce, widget, stripe, paypal
**Prerequisites**: Hanzo API key, browser environment
