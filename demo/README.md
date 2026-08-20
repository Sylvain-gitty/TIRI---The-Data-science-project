# TIRI · prototype

Two static pages that share one core:

| | |
|---|---|
| **`landing.html`** | the narrative — nine full-screen slides, what the problem is and what the instrument does |
| **`app.html`** | the app demo — the working loop: define a use case, label papers, train, review, export |

**This folder is self-contained.** Nothing outside it is required, there is no build step,
no package manager, no backend and no API keys. Copy the folder anywhere and it runs.

---

## Running it

Any of these. They are in order of least to most convenient:

**1 · Open the file.** Double-click `landing.html`. It works over `file://` — every script
is a classic `<script src>` rather than an ES module, precisely so that this is true
(modules are CORS-checked and `file://` has no origin).

**2 · Any static server.** From inside this folder:

```bash
python3 -m http.server 8090
```

**3 · The bundled server, if you are editing.**

```bash
python3 serve.py 8090
```

Use this one while you work on the files. `python3 -m http.server` sends `Last-Modified`
and nothing else, so a browser may serve a heuristically-fresh copy without revalidating
— you edit a stylesheet, reload, and see the old one, silently. `serve.py` sends
`Cache-Control: no-store` **and** rewrites each page's subresource URLs in flight to
`?v=<mtime>`, so a changed file is a changed URL and cannot come from cache. The files on
disk stay clean, which is what keeps route 1 honest.

---

## What is in here

```
landing.html landing.css landing.js   the narrative page
app.html     app.css     app.js       the app demo
shared/                               everything both pages use
  base.css      design tokens, element defaults, the icon size rules
  charts.css    every visualisation's styling
  assets.js     the wordmark and portraits, base64 — plus the icon-file resolver
  icons.js      the inline icon set (Lucide contract, 24x24, currentColor)
  data.js       THE FIXTURE: 76 papers, nine use cases, 28 library templates, the
                welcome slides. Change this and both pages change together.
  state.js      the shared mutable state the renderers read
  stats.js      the real maths — tf-idf, PCA, MDS, Rocchio, AUC, the Fbeta sweep
  charts.js     the shared renderers. `lean:true` is the landing page's presentation
                flag; it changes how a figure is PRESENTED and never what it computes
assets/         39 logo marks, two portrait cutouts, the repository QR code, and
                ICON_SOURCES.md — where every mark came from
serve.py        the no-cache dev server described above
README.md       this file
SPLIT.md        the design log — why the prototype is split in two, every bug the
                split found, and a record of each review round
```

### The rule the two pages are built around

Most of the landing page **embeds the real interface** — the same renderers, over the same
fixture, from `shared/*.js`. Not screenshots, not a reimplementation. That is the whole
reason there are two pages sharing a core rather than one page each: a narrative page
carrying its own copy of the scatter plot is a page that disagrees with the product
inside a fortnight, and every number on it becomes a claim nobody can check.

Two figures are declared exceptions and both say so on the page itself: the key-metric
scatter draws an illustrative 720-record sample (the 76-paper fixture moves 22 dots
across the whole precision-to-recall scale, which is the one thing that section exists to
show), and the performance chart draws supplied F2 figures. Everything else is measured
live, in the browser, from the fixture.

---

## The one external dependency

The two pages link **Google Fonts** for Libre Bodoni and Inter:

```html
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Libre+Bodoni…">
```

Offline, or with that host blocked, both fall back — `Georgia, serif` and the system sans
— and everything still lays out. But the serif is the brand voice and the difference is
obvious, so if the pages need to work with no network, vendor the two families into
`assets/` (both are OFL and freely redistributable) and swap the `<link>` in each page's
`<head>` for local `@font-face` rules.

Nothing else on either page touches the network. There is no `fetch`, no XHR, no
analytics, and no CDN script.

---

## If you fork it

- **The fixture is `shared/data.js`.** Both pages read it, so a change lands on both.
- **The repository link** on the closing slide is `REPO` in `landing.js`.
- **The welcome slides' body copy is placeholder** by design — `WSL` in `shared/data.js`,
  where each slide's `f:` field says what its final copy should cover.
- **`assets/*.png` are third-party logo marks.** `assets/ICON_SOURCES.md` lists every one
  with where it came from — most are [Simple Icons](https://github.com/simple-icons/simple-icons)
  exports, the rest are drawn replacements where no square mark existed. That file travels
  with the folder deliberately: attribution that stays behind in the original repository
  is attribution nobody can find. Check the terms before shipping them anywhere public.
- The icon set is the resolver's whole **vocabulary** (`ICON_FILE` in `shared/assets.js`),
  not just what the current slides draw — about sixteen of the thirty-nine are on screen.
  A new tool name will find its mark; deleting the unused ones would make it fail silently.
- The portraits and the wordmark are Fossier & Fauvel's own.
