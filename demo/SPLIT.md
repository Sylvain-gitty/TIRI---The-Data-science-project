# TIRI prototype — the split

**2026-08-19.** `converged.html` was one 4,930-line, 460KB file holding two surfaces:
the narrative landing page and the app demo. It is now `tiri/`, two entry points over
one shared core.

The old file is **deleted**, not kept alongside. It lives at commit `3a37840`
(`design(guided_funnel): converged v8 — when the labels and the use case stop
agreeing`) and `git show 3a37840:design/guided_funnel/prototype/converged.html`
returns it whole. Keeping a copy would have meant two places to make every change
and one of them going stale — which is the exact failure the split exists to prevent.

## What is where

| file | lines | what |
|---|---|---|
| `landing.html` / `.css` / `.js` | 215 / 202 / 270 | the narrative surface: eight sections |
| `app.html` / `.css` / `.js` | 335 / 708 / 1,958 | the loop: use case → search → sort → model → email |
| `shared/base.css` | 167 | tokens, reset, type, buttons, bar, modal + form chrome, icons |
| `shared/charts.css` | 320 | every chart both surfaces draw |
| `shared/assets.js` | 3 consts | the mark and the two photos, as data URIs |
| `shared/icons.js` | 55 | the 35-name Lucide/Feather set and `icon()` |
| `shared/data.js` | 259 | the 76-paper fixture, the nine use cases, the 28-collection library, Benchset v2, the worked description |
| `shared/state.js` | 28 | `S`, and the label counters everything reads |
| `shared/stats.js` | 469 | PCA, tf-idf, Rocchio, AUC, the PR sweep, the five settings, the description space |
| `shared/charts.js` | 790 | pool map, Sankey, bubbles, Compare, radar, the labelling animation, the flow, the 2×2 |

Open either entry point directly, or serve the folder:

```bash
python3 -m http.server 8090 --directory design/guided_funnel/prototype/tiri
```

There is a `tiri-proto` entry in `.claude/launch.json` that does exactly this.

## Why no build step

Plain `<link>` and classic `<script src>`. Both work over `file://`; ES modules do
not (the module loader is CORS-checked and `file://` has no origin). No bundler, no
`make`, nothing to forget to re-run — the "did you rebuild?" failure mode is the one
`make qa` exists to prevent in the real app, and a prototype does not need to invent
its own version of it.

The cost is that the folder must travel together. That is the right trade: the
alternative was a 460KB single file that nobody could find anything in.

## The rule that decides where code goes

**Anything both surfaces draw is shared. Anything only one draws is not.**

The landing page is not a mock. It seeds the shared state to a fully-labelled
position — the fixture's own gold answers plus the same worked description the demo
strip loads — and then calls the app's own renderers. Every count, percentage and dot
position on it is computed in the browser from that fixture.

Two deliberate exceptions, both stated on the page itself:

- **The Baseline/Advanced F2 chart** is drawn in `landing.js` with **supplied** numbers.
  76 papers split 60/20/20 leaves fifteen in the test set; a chart drawn from fifteen
  papers measures noise. Drawing it from the fixture anyway — to keep the
  "everything here is computed" line intact — would have been the worse of the two lies.
  The frame carries a one-line stamp saying where the figures come from.
- **The Key metric section has its own sample: 720 records at 12% prevalence.** The
  fixture cannot carry this section. With 76 papers and 23 relevant, High precision keeps 6
  and High recall keeps 28 — twenty-two dots change across the entire five-stop scale, and
  making that scale legible is the section's whole job. The sample is generated
  deterministically (a fixed integer hash, never `Math.random`), and the precision/recall
  sweep, the five Fβ optima and every number in the 2×2 are computed from it by the **same
  shared functions the app uses on real labels** — `curveFor`, `fPoints`, `confMatrix`,
  generalised to take any `{id,g}` set. Nothing is a number somebody typed. The frame says
  *"illustrative sample"* in its own corner: an illustration that admits it is one is fine,
  an illustration dressed as a measurement is not.
- **The confusion matrix** counts against gold answers. The app's own tune panel cannot and
  says so: nobody knows the answers for the rest of a real pool. The landing page gets the
  full table only because it is teaching what the words mean.

## What the split found

Eight bugs, all of them invisible in the single file. The first two are the same bug twice
— my CSS classifier had no rule for a selector with no class in it, so both landed in
`app.css`, which the landing page does not load:

0. **The webfont `@import` and `h1,h2,h3{font-family:var(--hd)}`.** The landing page had
   no Libre Bodoni and every heading on it rendered in Inter. The fonts are a `<link>` in
   both heads now (a `<link>` cannot be mis-ordered by a stylesheet split), and the element
   defaults — `html,body`, `button`, `h1-h3`, form elements, `::selection`, `kbd` — are in
   `shared/base.css`. A rule with no class in its selector is by definition not about one
   surface.

1. **`GEN_MASK` was an eager IIFE.** It ran at load, when `S.ans` is still `{}`, so
   `compiled()` returned `''` and the mask never removed the analyst's own
   vocabulary — the subtraction the comment described was not happening. Splitting
   the file moved `compiled()` and the load order made it throw instead of quietly
   returning the wrong answer. Now lazy and memoised on the description text. The
   General model's honest AUC is **0.748**, below the 0.782 baseline, and that
   ordering is the finding: your own words beat generic scaffolding.
2. **`.cm.tn`, my confusion-matrix cell, collided with `.tn`** — the tune panel's
   number row, three hundred lines away in the same stylesheet. `display:flex`
   flattened one cell of four and nothing else, which reads as a content bug. Cells
   are `cmC` / `ctp` / `cfp` / `cfn` / `ctn` now.
3. **`svg.ic`, not `.ic`.** Every chart container carries a
   `<container> svg{width:100%}` rule; a descendant selector outranks a plain class,
   so an inline `ⓘ` inside a chart's caption rendered at 500px. Raising the icon's
   specificity once immunises all of them.
4. **`shapeAt` was number-unsafe in one branch of six.** Five branches interpolate
   coordinates into a template, where a pre-formatted `"123.4"` works by accident.
   The new hexagon does arithmetic, so a string `x` turned `x + r*cos` into
   concatenation and the shape vanished.
5. **`.radioRow` set its column count inline.** An inline style beats any stylesheet
   rule, so the narrow-screen media query could never win and five settings stayed in
   five 60px columns on a phone. It is a custom property now.
6. **`CMP_PICK` said `'NER'`** for Named Entity Recognition and selected the library's
   "Don**ner**s 2021 — emicizumab dosing in haemophilia A". A bare substring match on
   a short fragment will eventually find a longer word to hide inside, and the failure
   is silent: the chart renders perfectly with the wrong use case in it. Word-boundary
   matching, and a `console.warn` when a name resolves to nothing.

And one measurement problem the landing page made unavoidable:

**The Compare frame was scoring six use cases in the cement papers' vocabulary.** A
soil-microbiome description had four of its fourteen words in that space and a
photovoltaics one had two, so five of six similarities floored at 0.00 and the layout
was driven by rounding noise. A chart whose headline is *"related questions land near
each other"* cannot be computed in a space where nothing is related to anything. There
is a second tf-idf now — `DESCPROJ`, over all 37 use-case descriptions — and the result
is a real finding: **Low-carbon cement and Carbon Capture score 0.73** and the other
four are genuinely unrelated to everything, which is what those six use cases actually
are. The app never had this problem, because `bge-small-en-v1.5` embeds any text into
one space; the prototype had it because a single HTML file cannot carry a sentence encoder.

## Scroll model

**All eight sections fit one viewport, with their next button visible.** Six of them
(03–08) are `.lsec.full`: a flex column of a tight header, the chart, and the button,
where the chart takes whatever is left.

The one thing that makes that work is worth writing down, because it is invisible and
unavoidable. An `<svg>` with a viewBox is a **replaced element with an intrinsic aspect
ratio**: stretched to its container's width as a flex item, it hands a height back, and
`height:100%` on it makes the whole thing circular. The browser resolves the loop from
the aspect ratio, so every full-bleed section came out 50–500px taller than the viewport
and pushed its button off screen. `min-height:0` does not help — the svg is not the flex
item being asked to shrink, it is the content inside one. The fix is to take the svg out
of flow entirely: the stage is `position:relative`, the svg is `position:absolute;inset:0`.
Then the stage is sized by its flex parent and the svg fits whatever it is given.

Snapping is `y proximity` rather than `mandatory`, and the next buttons are in the flow
rather than pinned at `bottom:44px` — a pinned button belongs to a section that fits, and
these did not until the svg fix landed.

## The presentation flag

The shared renderers take `lean:true`. The app hosts each of them in a modal, where prose
has room and a table is the precise read; the landing page gives each a whole screen and
needs the chart to be the only thing on it. So `lean` means fill the frame, big labels, no
table, no explanatory block, no control the section does not need.

**`lean` changes nothing that is computed.** Same numbers, presented for a different
reading distance. One flag rather than six — a flag per removed element would have been a
config surface, and the intent is one decision, not six.

`scope` is separate from `lean`, because it is a DATA choice: the landing page totals six
use cases (`SIX` in data.js) rather than all nine, and the Sankey carries the same six.
The three left out are this project scoring itself and two archived stubs.

## Found by giving the charts a whole screen

- **The Sankey's scale ignored its own label floors.** Every small source gets a slot at
  least one label tall regardless of its bar, so the source column's real height is one big
  bar plus five floors — 529 units inside a 360-unit canvas. **arXiv and CORE were being
  drawn below the bottom edge of the viewBox**: not clipped, absent. Present in the app too,
  just less. The scale is now solved by iteration (which items are at the floor depends on
  the scale, and the scale depends on which items are at the floor), and CORE rides along as
  a zero-value item so the layout reserves its slot instead of a magic `+30` offset.
- **The EDA plot stacked four of six clusters.** Four of the six use cases have a measured
  similarity of exactly 0 — which is the truthful answer — so MDS had nothing to separate
  them with and four cluster names printed on top of each other. Pairwise repulsion did not
  fix it: six points at a 0.30 minimum inside a 0.76×0.68 box is near the packing limit, so
  points pin against the walls and the passes fight the clamp (measured separation 0.077
  against a 0.30 target). It is a 3×2 slot assignment now, and the plot says *"relaxed for
  legibility · read the grouping, not the distances"* rather than leaving the reader to
  measure similarity off a position that no longer encodes it.
- **The stakeholder flow's "Industrial R&D" label collided with "Longer timelines"** — but
  only in the *bad* view, where the label is two words longer. A collision that appears in
  one of two states is one a single screenshot never catches.
- **`.cfStamp` printed over the performance chart's own axis sublabel.** Both wanted the
  bottom-right corner.

## Serving it — and why there is a `serve.py`

```bash
python3 design/guided_funnel/prototype/tiri/serve.py
```

Or open `landing.html` / `app.html` straight off the filesystem; both work over `file://`.

**The server exists because staleness cost real time three separate times.** A font fix
measured as still-broken for three reloads; a JS change looked like it had done nothing
when it had simply not been fetched. `python -m http.server` sends `Last-Modified` and
nothing else, so a browser may serve a heuristically-fresh copy without revalidating —
silently, with no error and no visual cue, which is the worst way for a prototype to be
wrong.

Two mechanisms, because one was not enough:

1. `Cache-Control: no-store` on every response. Necessary, and on its own it **failed** —
   an entry cached earlier under permissive headers survived a server stop, a restart, a
   reload and a brand-new tab. `no-store` on a response the browser never requests cannot
   evict anything.
2. So the HTML is **rewritten in flight**: `src="app.js"` becomes `src="app.js?v=<mtime>"`.
   A changed file is a changed URL and cannot come from cache; an unchanged file keeps its
   URL and still can. The files on disk stay plain, which matters — a permanent `?v=` in
   the markup would be a lie when the page is opened over `file://`.

## Round 11 — the charts at full screen

**Responsive, by height not width.** The Stakeholders section broke on a 13-inch
MacBook, and the reason generalises: an SVG label placed with `text-anchor:start`
occupies however many units its glyphs need, and that width scales with the frame.
At 1.0× "OVER DECADES" cleared "INDUSTRIAL R&D"; at 1.4× it did not. There is no
measurement available at build time, so the flow is divided into four **zones** with
a guaranteed gutter and every label is anchored inside its own — the widest label a
zone can hold is ~230 units and no two anchors are closer than 280, so the collision
cannot happen at any scale rather than not happening at the one that was checked.
The viewBox went 1000 → 1200 to buy the room. The two non-full sections (hero,
business goal) size from their content and overflowed a 700px-tall window, so they
have height-relative caps — `min(px, vw, vh)` rather than a breakpoint ladder, so
there is no size at which the page is between two rules and correct at neither.

**The two mistakes now have two colours.** A false positive and a false negative
cost completely different things — one wastes an afternoon, the other loses the
paper — and drawing both in `--no` said they were the same error twice. `--no`
(orange) is the one you read for nothing; `--miss` (red) is the one you never see.
Red is reserved for that: `--rust` is the conflict warning and is deliberately a
muted brick, not this.

**The 2×2 is keyed by dots, not by type colour.** Tinting each cell's number and
name in its outcome's colour made the table its own legend by re-colouring text —
the weakest way to carry a key: it fails for a red-green deficiency, it drags the
type off the page's ink colour, and it cannot be matched to a mark on a scatter plot
without the reader doing the mapping. A swatch in the mark's own colour beside a name
in ordinary ink is the standard annotation and maps one-to-one onto the plot.

**Solutions used has its own screen.** Sharing one with the Sankey meant neither got
its height — the diagram was squeezed to a third of the section and the stack list
still had to lose its subtitles to fit. Nine sections now.

**The EDA layout, third attempt, and this one is right.** The job is to place N use
cases so distance means dissimilarity.

1. *PCA of the description vectors.* Wrong instrument. PCA finds directions of
   greatest variance in the feature space; over 37 short documents and a 52-word
   shared vocabulary most of that variance is which rare word each one happens to
   use, so four unrelated use cases all landed near the origin and their labels
   stacked.
2. *Snapping to a 3×2 grid.* Separation guaranteed — and distance stopped meaning
   anything, which threw away the one thing the figure is for. A grid can say "here
   are six use cases"; it cannot say "these two are the same question".
3. **Stress majorization on the similarity matrix**, which is what MDS actually is.
   Target distance for a pair is `1 − cosine`: the 0.73 pair wants to be 0.27 apart
   and overlaps, the 0.00 pairs want a full 1.0 and take opposite corners. Measured
   result: 0.73 → 0.11 apart, 0.20 → 0.36, and every 0.00 pair 0.42–0.74. Distance
   means dissimilarity, and it fills the frame for free.

Positions being truthful again put the labels back in each other's way, so the
**labels** move instead: eight bearings at three radii, scored against every label
already placed and every other centroid, first clear slot wins, leader line back to
the mark when it moved far enough to need one. A label is free to be anywhere; a
centroid is not. And the palette is ordered so the pair that *overlaps* gets the
widest colour separation in the set — ochre against blue — since they are the two
that most need telling apart.

**Also:** the bubble centres are equidistant and the group is centred (they were
packed, which put a 22-unit disc hard against a 190-unit one and listed the whole
group 32 units to the top); the disc labels are one sans family with hierarchy from
size, weight and tone rather than three typefaces in one block; the Sankey's viewBox
now matches the aspect of its own frame, because `meet` scales the labels along with
the diagram and a mismatched viewBox shrinks them.

**Missing:** `assets/` holds only `F&F - Dark.png` and the two portraits, so the
stack section has no logo marks to draw from. The slot exists and is empty —
`SOL_ICON` in `landing.js`, one line per mark, files in `assets/stack/`. Drawing
approximations of other projects' brand marks would be a small forgery, so the rows
are names until the files arrive.

## The welcome modal, and a regression I introduced

Splitting the prototype forced the first-run flag from an in-memory `let` into
`localStorage`, because the landing page became a separate document and the flag has to
cross a navigation. That fixed the double-fire and **made the modal permanently
unreachable**: once dismissed, on any machine, there was no way back to it.

That is a defect in two places at once. The prototype could no longer show its own
first-run state, and a product whose single explanation of itself can be seen once, by
accident, only on a machine that has never opened it, does not explain itself.

Three ways past the flag now:

- **"How it works"** in the workspace header — the product answer. An explanation is
  something you go back to.
- **"welcome slides"** on the prototype strip.
- **reset** on the strip clears the flag, so reset means reset rather than "reload with
  yesterday's storage still in place".

`maybeWelcome()` still respects the flag; `openWelcome()` deliberately does not — asking
for it is the answer.

## Round 13 — the deck read at a distance

Almost every item this round was the same instruction applied to a different section:
*make it bigger and take the words out.* Worth recording is what that cost each time,
because "remove microcopy" is not free and twice it was the thing holding a figure up.

**The `fRadio` strip is now two genuinely different controls.** In the app each card
carries a gloss ("only what it is surest of") and a cost ("42 papers · 50% right"),
because an analyst is *choosing a setting* and the numbers are the choice. On the
landing page the reader is being shown a **scale**, and five names in a row ordered
precision → recall *is* the scale — the costs are already drawn full-screen above the
strip as the sample changes colour, so printing them again at 13px made the strip the
thing being read. One conditional line stays: when two stops resolve to the same
cut-off the card still says so, because presenting two identical settings as different
choices is the failure this control was rebuilt to avoid.

**A key can also be a frame too many.** The 2×2 sat in a `.cfPanel` with its own border
and 11px of padding, around a grid that already draws a border and fills its cells
opaquely — a frame around a framed thing, which at full screen reads as chrome. The
two mistake cells lost their tint at the same time: the swatch carries the colour, and
two coloured cells beside two plain ones read as two *rows of importance* rather than
four counts of equal standing. Both changes are scoped to `lean`; the app's panel keeps
its tints, where highlighting what a decision cost is the whole point.

**`height:100%` on a flex item's child is not reliable.** The How-it-works track was
`height:100%` inside `.lfill` (itself `flex:1`), and Chrome resolved the percentage
against an indefinite height — the track sized to its content, so one slide filled a
third of the frame with 470px of paper under it. `flex:1;min-height:0` on the track
instead. Then the slide's `<svg>` had to come out of flow for the reason already
written at `.cfStage`: as a flex item it contributes its aspect-ratio height back and
the section came out 1,043px tall in a 960px window, next button under the fold. That
is the *third* place in this prototype where a viewBox'd SVG in a flex column did
exactly this. If a chart is inside a flex column, put it in a `position:relative` stage
and `position:absolute;inset:0` the SVG — every time, not when it misbehaves.

**A grid row cannot shrink below its content, so `flex:1;min-height:0` does nothing
there.** The Dataset section is the one full-bleed section whose content has a real
minimum height — six bar rows and four dictionary groups, all of it type — and it ran
39px past a 700px window. No flexbox property fixes that; the height media queries give
the padding back instead.

**Removing the centroid marks exposed a hole that had always been there.** The
synthetic clouds jittered at `0.028 + u*0.072`, i.e. an annulus — invisible while a
numbered marker sat in the middle of each one, and unmistakable the moment label mode
took the markers away, because six doughnuts is not what a corpus looks like. Now
`0.088*u^0.8`: a filled disc, denser at the centre (`u^0.5` is exactly uniform, below
that the rim gets denser, above it the middle does). The fix improves the app's modal
too, where the artifact was merely hidden rather than absent.

**Label mode names nothing.** Colouring by label while leaving per-use-case names and
per-use-case centroid rings on the plot gives it two contradictory keys at once — a
green dot inside a purple ring. Names and rings go; a two-swatch key comes in.

**Measured, not eyeballed:** the two performance label boxes were checked by comparing
each `<text>`'s `getBBox()` to its `<rect>` (Advanced's longest line ends at 964.3 in a
box running to 975.9); the welcome diagram's "NOT FOR ME" label was 65 units wide in a
62-unit box — a 1-unit bleed in the modal and five on a full-screen slide — so the box
went to 70; the field of 1,000 squares in the Stakeholders flow had its centre 45 units
above the `CY` that the coins, the decade axis and the plant all sit on, and its top is
now *solved* from the row count and pitch rather than typed.

**Two team cards, one width.** Each card was sized by its own longest URL —
`linkedin.com/in/sylvain-fossier` is 233px, `linkedin.com/in/warrenof` is 181px — so a
deliberately mirrored pair sat 52px off centre from each other and the symmetry read as
a mistake. A `min-width` floor on the caption fixes it; the URLs replaced the two social
glyphs because an icon cannot be typed into a phone from the back of a room.

**Note on the closing slide:** `Feature engineering` is one of its eight bullets and has
no section of its own. It is the finding that sits under EDA and Performance both. The
bullets are no longer numbered, which is what makes that legitimate — a numbered list
would have been pointing at a slide that is not there.

## Round 14 — three sizes and a footer

**The EDA key moved inside its own plot.** Same argument that put the 2×2 inside the
operating-point frame: under a full-screen scatter a key sitting two hundred pixels
below the marks it explains is a second reading, and the eye does not make the trip.
Bottom-left, over the emptiest corner of the six-cluster layout, at 19px with round
swatches that match the marks. Still only in label mode — colour-by-use-case names
every cluster on the plot, so a key of six identical circles beside six names would
just be the names again.

**One size expression, two slides.** The closing mark now uses the *same* `min(340px,
58vw)` as the hero rather than a value of its own, and deliberately nothing else: the
two height media queries at the foot of `landing.css` re-size `.lff` on short windows,
they have equal specificity and they come later in the file, so they govern the closing
mark too. Measured 340/340 at 960px tall and 269/269 at 790. A second expression would
have matched at the one viewport it was checked at and drifted at every other.

**`Title: point`, on one line.** Eight bullets as a bold heading with a line beneath
read as eight little sections; run together they read as eight statements, which is
what a summary is — and the vertical room the second line was using paid for 14px →
16.5px.

**Also:** the dictionary column stepped up to 16.5px (15.5/14.5 on the two short-window
rungs), and the slide has a whitepaper footer — a document mark and the address, set as
a link rather than a button because it leaves the deck. `WHITEPAPER` in `landing.js` is
a **placeholder URL**, named as a constant so there is one line to change and so the
fact that it is a placeholder is written next to it.

## Round 15 — a mark per point, and one thing asking to be clicked

**Eight marks, and the test they are meant to fail.** Each point on the closing slide
takes a glyph for its SUBJECT rather than for its sentence — `sliders` for feature
engineering because the point is that the description is the lever, `trend` for
performance because the point is a line going the right way. Strip the words and
nobody could reconstruct the summary from the icons, which is the rule at the top of
`shared/icons.js`: a mark sits beside a heading and never carries meaning alone.

**Filled, not outlined.** An outlined 15px link next to two large photographs reads as
a caption. The whitepaper is the one thing on the slide asking to be clicked, so it is
filled in the brand at 20px with the address in mono a shade behind it — the sentence
carries across a room, the URL carries at a keyboard. The icon's stroke goes up with it:
1.75 is drawn for a 20px mark on paper and thins into a wireframe at 27px on a dark
ground.

**The closing slide is now the height-critical one.** It carries a hero-sized mark, a
headline, eight summary rows, two photographs and a footer — more blocks than any other
section — so it hit the viewport first: 9px over at 700px tall and 11px over at 790. It
has its own step on both height rungs now. That is the pattern for this page rather than
an exception: `.lsec.full` sections are governed by their chart, and the two that size
from content are governed by whichever has the most blocks in it.

## Round 16 — the closing slide, rebuilt

**The two "-Transparent.png" files were not transparent.** PNG colour type 02, no alpha
channel, with the transparency CHECKERBOARD rasterised in as pixels — the border samples
alternate ~233 grey and ~254 white. So the alpha had to be recovered, and two things had
to be got right:

1. *Keying by colour alone would erase Warren's barn owl*, whose face is near-white. The
   key is CONNECTED-COMPONENT based instead: a pixel is background only if it is
   checkerboard-coloured **and** reachable from the image border through other such
   pixels. The owl's face is enclosed by the owl, so it survives. Sylvain's clear
   glasses survive for the same reason.
2. *The first attempt banded the luminance tightly around 233 and 254* and left
   rectangular patches of checkerboard in the top-left of both images. The seams between
   squares are anti-aliased to values in neither band, which severed connectivity and
   left whole squares as islands with no path to the border. One open band above 224
   keeps them joined; connectivity is still what protects the owl.

Plus a one-pixel erosion of the subject — its outermost ring is where the original
semi-transparent edge was blended with the checkerboard, so it carried a grey fringe —
and a 0.8px blur on the alpha to soften the cut. Verified by compositing both over the
page's own background colour and looking at the result, not by trusting the numbers.
Exported as **WebP**: 161KB and 107KB against 1.7MB and 1.6MB as PNG, alpha intact. It
is the only WebP in the prototype and it earns the exception — everything else in
`assets/` is a small flat-colour mark where PNG costs nothing.

**A `<canvas>` is a replaced element, so `inset:0` does not size it.** The confetti
surface stayed 300x150 in the corner: for an absolutely positioned replaced element the
spec resolves `width:auto` from the intrinsic size and ignores the over-constrained
`right`. Worse, the burst reads its own box to size the backing store, so every click
multiplied the last measurement by the device pixel ratio — 300 → 600 → 1200. Same
family as an `<svg>` in a flex column, which this file has now recorded four times.
`width:100%;height:100%` alongside the `inset`.

**One number governs the portraits and the summary's width.** `--folk` is the share of
each portrait hidden off its own edge. It drives the transform *and* the width the
summary is allowed, because each visible half is `(1 - folk)` of the section's height —
the photographs are square — so the clear middle is `100vw - 2*(1-folk)*100vh`. Written
as two independent values they drift the moment either changes, and the failure mode is
the summary running onto a dark shoulder where it cannot be read. `max(430px, …)` puts a
floor under it: on a very tall narrow window the clear middle goes negative and `min()`
alone resolves the list to zero width.

At 1180x700 the clear middle was 432px, which put every summary point on three lines and
the section 63px past the window. The lever is *widening* the allowance, not shrinking
the type — two columns of 194px cannot hold a 61-character sentence at any size worth
reading — so `--folk` goes to .62 on that rung and the allowance follows. At 960 and 790
tall nothing moves.

**Confetti behind the type, not in front.** `--folk` portraits at z-index 0, the canvas
at 1, the content at 2. In front is the more usual look for a celebration; behind keeps
the summary readable through the whole burst, and this slide is on screen while
questions are being answered. Under `prefers-reduced-motion` it does nothing at all — a
"reduced-motion confetti burst" is still a confetti burst, and the mark carries no other
function, so a silent no-op costs nobody anything.

**Also:** "Industry" capitalised in the headline, the bar and the `<title>`; the four
profile URLs are gone from slide 01 (nothing there is meant to be clicked yet — the deck
has not made its case) and the names went to 27px serif to fill the space they left; the
business goal's headline carries an asterisk and the definition it stands for as an
italic serif footnote, because it qualifies the HEADLINE rather than the solution
column; the How-it-works step titles went sans, which demotes them under the section
headline without shrinking them; the tool marks are 60px in equal full-width columns
(`auto-fit` + `minmax(0,1fr)`, and the 0 is what lets a long name shrink its column
instead of overflowing the grid).

**One placeholder is gone and none replaced it:** the whitepaper URL was a placeholder,
the repository URL is real.

## Round 17 — the duotone on a cutout, and a QR worth raising a phone at

**The same duotone as slide 01, which on a cutout takes two layers.** `.lpimg` gets it
for free: a green box with an OPAQUE photograph over it in `mix-blend-mode:luminosity`,
so the photo supplies the light and the box supplies the hue. A cutout cannot do that —
a green box behind it shows as a green rectangle everywhere the photograph is
transparent, and `background-blend-mode` fails the same way (the bottom layer is opaque
green, so a transparent top pixel resolves to green).

So the green is **masked to the cutout's own alpha** — a `::before` using the same file
as the `<img>` — and the photograph blends over that. `isolation:isolate` on the wrapper
is the part that matters: it confines the blend, so `luminosity` sees the masked green
as its backdrop rather than the page, the summary, or the other portrait. And
`aspect-ratio:1`, because a mask box needs a width and `height:100%;width:auto` gives it
none; these photographs are square, so 1 is the honest number rather than a convenient
one. The file is named twice and fetched once.

**One token for the clear middle.** `--clear` is now derived from `--folk` and read by
BOTH the summary and the repository block. The 170px QR was what forced it: at 1440x790
the block's left edge landed 88px inside Sylvain's shoulder while the summary — bound by
its own copy of the same arithmetic — sat comfortably clear. Two elements obeying the
same rule from two places is one element eventually not obeying it.

**The QR gives way before the section does.** 92px was a decoration nobody in a room
raises a phone at, so it takes the height the slide has spare — but sized in vh as well
as px, and stepped down on both short rungs (20vh / 14vh / 13vh). Dropping the printed
address, which the brief allowed, would not have helped: the QR sets the block's HEIGHT
and the address only its width. So the address stays and the code scales.

**A URL breaks where a path breaks.** Capped to the clear middle the address has to wrap,
and `word-break:break-all` wrapped it mid-name — `…-science-proj / ect`, which reads as a
rendering fault rather than a line break. It is now split in the renderer at the last
separator (owner on one line, repository on the next, both intact) with
`overflow-wrap:anywhere` left as the net for a window narrower than the repository name
itself. `break-all` breaks eagerly; `anywhere` breaks only when it must.

## Round 18 — the filter was identical; the tonal weight was not

Asked again to apply slide 01's colour filter to the closing slide's background, the
first thing to do was check whether it was applied. It was, byte for byte: both compute
to `mix-blend-mode:luminosity` with `contrast(1.04) brightness(1.22)` over
`rgb(36,81,55)`. Read off the computed styles, not inferred from the stylesheet.

**What differs is what each one is showing.** Slide 01 crops to the faces with
`object-fit:cover`, so its average luminance is mid and the tile reads as a mid-tone
green. A full-height cutout is mostly dark shirt, so the same filter produces a heavy
dark mass. Identical treatment, opposite impression — which is worth recording because
"apply the same filter" and "make it look the same" are different instructions and only
one of them was actually satisfiable by touching the filter.

So the correction is an **opacity**, not another filter: at `.72` the pale page shows
through the darks, the pair sits at the tonal weight of slide 01's cards, and the faces
and the owls stay readable — which `.6` gave up. It also puts them back in the role the
radar wash used to play on this slide: background, at the page's own value.

## Round 19 — packaged to travel

Asked for a single folder that runs independently in another repo. The honest answer was
that **`tiri/` almost already was one** — so this round moved two things in and wrote
nothing new to build. Checked rather than assumed:

- **No path escapes the folder.** The only `../` in the tree is inside a comment.
- **No modules, no `fetch`, no XHR.** Every script is a classic `<script src>`, which is
  what makes `file://` work at all (modules are CORS-checked and `file://` has no
  origin). `localStorage` is wrapped in `try/catch` with an in-memory fallback in both
  directions, which is the other thing `file://` can break.
- **`app.html` does not load `shared/assets.js` and does not need to** — its only
  consumers are `landing.js` and the `typeof ICO==='function'`-guarded branch of
  `plantArt`, which is a landing-only path. Verified rather than left as a suspicion.
- **The only external host is Google Fonts**, and both families fall back to real stacks
  (`Georgia, serif`, system sans). Recorded in the README with what it costs and how to
  vendor them, because losing Libre Bodoni has already been mistaken for a bug once.

What moved: **`serve.py` came in from the parent** and now serves its own directory, so
nothing above the folder is needed (`.claude/launch.json` repointed). **`ICON_SOURCES.md`
was copied into `assets/`** — attribution that stays behind in the original repository is
attribution nobody can find. And both pages got an **inline data-URI favicon**, which is
the only 404 either page has ever produced.

`README.md` is now the front door: the three ways to run it, what each file is, the rule
the two pages are built around, the one external host, and the five things a fork needs
to change. `SPLIT.md` stays what it is — the log.

1.2MB, 60 files, no build step.

## Still open

- The welcome slides' sub-copy is placeholder, by request.
- ④ label drift is deliberately not built — see the v8 notes in `app.js`.
- Iteration 1g has not been written: `DESIGN.md`, `NUMBERS.md` and `UX_TRACK.md` are
  four commits behind the prototype.
