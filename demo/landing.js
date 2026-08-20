/* ══════════════════════════════════════════════════════════════════════════
   TIRI · LANDING · behaviour
   design/guided_funnel/prototype/tiri/landing.js

   Everything on this page that is not chrome comes out of shared/*.js. What is
   here is (a) the seeding that puts the shared state into a fully-labelled
   position, (b) the graphics only this page draws, and (c) the mounting.

   WHY SEED AT ALL. The charts read `S.lab` and `S.ans`, because in the app those
   are the analyst's work. On a landing page there is no analyst, so the honest
   choice is to show the interface at a state that actually occurred: the
   fixture's own gold answers, and the same worked description the app's demo
   strip loads. The alternative — hard-coding plausible numbers into the page —
   is the ADR 0009 failure, a fabricated measurement that looks exactly like a
   real one.

   THE PRESENTATION RULE. Five sections give a chart the whole screen, so the
   shared renderers are called with `lean:true` — fill the frame, big labels, no
   table, no prose, no control the section does not need. `lean` changes nothing
   that is COMPUTED. Same numbers, presented for a different reading distance.
   ══════════════════════════════════════════════════════════════════════════ */

/* ── the seed ──────────────────────────────────────────────────────────────
   Every paper that has an abstract gets its gold answer. `nab` papers are
   skipped for the same reason the app never deals them: a paper with nothing to
   read was never a decision, and counting it as one would inflate every
   denominator on this page. */
S.ans={...DEMO_ANS};
PAPERS.forEach(p=>{ if(!p.nab) S.lab[p.id]=p.g });

/* the page's own controls. `flow` is which ending section 03 shows; `set` is
   which of the five operating points section 04 sits at. */
const LS={ flow:'good', set:'f1' };

/* ══════════════════════════════════════════════════════════════════════════
   03 · THE TWO ENDINGS
   ══════════════════════════════════════════════════════════════════════════ */
function setFlow(k){ LS.flow=k; renderFlowSec() }
function renderFlowSec(){
  document.getElementById('flowSegs').innerHTML=
    [['good','When it works'],['bad','When it does not']].map(([k,n])=>
      `<button class="${LS.flow===k?'on':''}" onclick="setFlow('${k}')">${n}</button>`).join('');
  document.getElementById('lflow').innerHTML=flowSVG(LS.flow==='bad');
}

/* ══════════════════════════════════════════════════════════════════════════
   04 · KEY METRIC
   The precision/recall trade-off as one full-screen figure: the sample, the 2x2
   as its key, and the five settings as a strip — all three INSIDE the chart
   frame, because the numbers and the picture are one reading and a caption two
   hundred pixels below the plot is a second one.

   ─────────────────────────────────────────────────────────────────────────
   WHY THIS SECTION HAS ITS OWN SAMPLE, AND WHY THAT IS NOT A FABRICATION
   ─────────────────────────────────────────────────────────────────────────
   Every other figure on this page is the real fixture. This one cannot be. The
   fixture is 76 papers with 23 relevant, so High precision keeps 6 and High
   recall keeps 28 — twenty-two dots change across the entire five-stop scale,
   and the section's whole job is to make that scale legible.

   So: a declared SAMPLE of 720 records at 13% prevalence, positions and scores
   drawn from two overlapping normal distributions. It is generated
   deterministically — a fixed integer hash, never Math.random — so the picture
   is identical on every load and can be reasoned about.

   What is real about it: the precision/recall sweep, the five Fβ optima, and
   every number in the 2x2 are computed from those scores by the SAME shared
   functions the app uses on real labels (`curveFor`, `fPoints`, `confMatrix`).
   Nothing here is a number somebody typed because it looked convincing. What is
   not real is the sample itself, and the frame says "illustrative sample" in
   its own corner rather than in a paragraph — which is the concise labelling
   this section was asked for, and the ADR 0009 line either way: an illustration
   that admits it is one is fine; an illustration dressed as a measurement is not.
   ══════════════════════════════════════════════════════════════════════════ */
const WS=(()=>{
  const N=720, PREV=0.13;
  /* one integer hash, two uniforms per draw, Box-Muller for the normals. Fixed
     seed arithmetic: no Math.random anywhere, so this is the same 720 points in
     every browser on every load. */
  let s=0x2f6e2b1;
  const u=()=>{ s=(s*1103515245+12345)&0x7fffffff; return (s>>>8)/0x7fffff };
  const gauss=()=>{ const a=Math.max(1e-9,u()), b=u();
    return Math.sqrt(-2*Math.log(a))*Math.cos(2*Math.PI*b) };
  const items=[];
  for(let i=0;i<N;i++){
    const pos = u() < PREV;
    /* the separating axis. Positives sit higher on it and the two clouds OVERLAP
       — without overlap there is no trade-off to show and the five settings would
       all land on the same cut-off, which is the failure mode this section is
       about. */
    const a = pos ? 0.62+gauss()*0.30 : -0.18+gauss()*0.34;
    const b = gauss()*0.42;
    /* the score is the axis plus noise, so the model is good but not an oracle:
       some positives score below some negatives, which is what produces the
       false positives and false negatives the 2x2 counts. */
    items.push({id:i, g:pos?'pos':'neg', a, b, sc:a+gauss()*0.24});
  }
  /* Screen coordinates, normalised on the 2nd–98th PERCENTILE rather than on
     min/max. Two normal distributions have long thin tails, so scaling to the
     extremes packed 96% of the points into the middle 60% of the frame and left a
     third of the plot empty. Clamping to the percentiles fills the frame and the
     handful of true outliers sit on the edge, which is where an outlier belongs. */
  const q=(arr,p)=>{ const s=[...arr].sort((a,b)=>a-b);
    return s[Math.min(s.length-1,Math.max(0,Math.round(p*(s.length-1))))] };
  const ax=items.map(p=>p.a), bx=items.map(p=>p.b);
  const lo=q(ax,0.02), hi=q(ax,0.98), lo2=q(bx,0.02), hi2=q(bx,0.98);
  /* THE TAILS ARE SQUASHED, NOT CLAMPED. Clamping to a bound — any bound — puts every
     point beyond it at exactly the same coordinate, so the far tail stacked into a
     straight vertical line at the plot's edge and read as a boundary in the data.
     There is no boundary; there are six points a long way out.
     An asymptotic squash maps everything outside [0,1] into a thin margin while
     preserving the ORDER, so those six fan out along the edge instead of coinciding.

     THE BOUND HAS TO MATCH THE PADDING. The first version used `d/(1+4d)`, bounded by
     0.25 — five times the 0.048 of margin the plot's padding actually provides, so the
     far points were laid out past the edge of the viewBox and silently clipped away
     entirely. Worse than a stack: a stack is visible. `MARG*d/(1+d)` is bounded by
     MARG by construction, and MARG is the margin in the same units the caller pads in.
     Asserted below, because "it looks fine" is what the last two versions looked. */
  const MARG=0.042;
  const squash=d=>MARG*d/(1+d);
  const cl=v=> v>1 ? 1+squash(v-1) : v<0 ? -squash(-v) : v;
  items.forEach(p=>{
    p.x=cl((p.a-lo)/(hi-lo||1));
    p.y=1-cl((p.b-lo2)/(hi2-lo2||1));
  });
  const sc=new Float64Array(N); items.forEach(p=>sc[p.id]=p.sc);
  /* every point inside the drawable box, or the plot is quietly lying about its size */
  const off=items.filter(p=>p.x<-MARG-1e-9||p.x>1+MARG+1e-9||p.y<-MARG-1e-9||p.y>1+MARG+1e-9);
  if(off.length) console.error('WS: '+off.length+' points outside the plot box');
  return {items, sc, N, nPos:items.filter(p=>p.g==='pos').length, MARG};
})();

function setWs(k){ LS.set=k; renderWeak() }
function renderWeak(){
  const pts=curveFor(WS.sc,WS.items), F=fPoints(pts);
  if(!(LS.set in F)) LS.set='f1';
  const at=pts[F[LS.set]], cut=new Set(at.order.slice(0,at.n));
  document.getElementById('wsBody').innerHTML=`
    <div class="chartFrame">
      <div class="cfStage">${wsScatter(cut)}
        <div class="cfPanel">${confMatrix(cut,WS.items,{lean:true})}</div>
      </div>
      <div class="cfCtl">${fRadio(pts,F,LS.set,"setWs('%k')",{fold:false})}</div>
    </div>`;
}

/* The scatter is drawn here rather than by `poolMap`, and the reason is that
   these are two different figures wearing the same colours. `poolMap` REPORTS on
   the analyst's own 76 papers inside a modal; this EXPLAINS a concept at full
   screen over a declared sample. Parameterising one renderer to do both would
   have meant a flag for the point source, a flag for the frame, and a flag for
   the embedded panels — six flags to share forty lines. What they do share is the
   part that matters for consistency: the four outcome classes and their colours
   come from shared/charts.css, so a false positive is the same orange in both. */
function wsScatter(cut){
  /* the padding has to cover the tails' margin: MARG is a FRACTION of the span, so
     the pixels it needs depend on the span, which depends on the padding. Solved
     rather than guessed — pad = MARG*(W-2*pad) rearranges to this. */
  const W=1000, H=560;
  const pad=Math.ceil(WS.MARG*W/(1+2*WS.MARG))+8;
  const dots=WS.items.map(p=>{
    const inCut=cut.has(p.id), rel=p.g==='pos';
    const cls = inCut ? (rel?'dTP':'dFP') : (rel?'dFN':'dTN');
    const r = cls==='dTN' ? 3.4 : cls==='dTP' ? 5.4 : 6;
    return `<circle class="${cls}" cx="${(pad+p.x*(W-2*pad)).toFixed(1)}"
      cy="${(pad+p.y*(H-2*pad)).toFixed(1)}" r="${r}"/>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img"
      aria-label="the sample, coloured by whether the model was right about each record">
    <style>.dTP{fill:hsl(145,38%,23%);opacity:.92}.dTN{fill:hsl(150,10%,52%);opacity:.26}
      .dFP{fill:hsl(26,72%,45%);opacity:.95}
      .dFN{fill:hsl(2,66%,46%);opacity:.95}</style>
    ${dots}</svg>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   05 · THE DATASET, AS TWO COLUMNS
   80/20: what has been labelled, and what a row of it actually contains. The
   proportional-area bubbles are gone — they answered "how much is there against
   what exists", which is a comparison against Benchset, and this section is now
   about what WE have. A horizontal bar per use case with its own split is the
   right shape for six numbers of the same kind.

   The dictionary column is not decoration: the most common question about a
   labelled corpus is what a row of it is, and the answer is short enough to print.
   ══════════════════════════════════════════════════════════════════════════ */
/* ── THE DICTIONARY, AS FOUR LINES ──────────────────────────────────────────
   Each group is an icon, a name and its columns on ONE comma-separated line. It
   used to be a stack of one-word rows, which spent seven lines of type on seven
   words and made the reader's eye travel further than the content did. The counts
   collapse the same way: "Embeddings: 385" is the whole fact, and "385 columns" in
   a column list was saying "columns" twice.

   The icons are from the inline set, and they follow that file's own rule — a mark
   beside a heading, never meaning on its own. Every one of these four is legible
   with its word removed only by accident; the word is what carries it. */
const DICT=[
  ['file',   'Paper metadata', 'DOI, Title, Abstract, Venue, Citations, Year, Authors'],
  ['layers', 'Embeddings',     '385'],
  ['search', 'Sources',        '7'],
  ['tag',    'Label',          'Interesting / Not interesting']
];
function renderDataset(){
  const us=sixUcs();
  /* one scale across all six bars, or the lengths mean nothing against each other */
  const max=Math.max(...us.map(u=>u.pos+u.neg));
  const rows=us.map(u=>{
    const tot=u.pos+u.neg, w=tot/max*100, pp=u.pos/tot*100;
    return `<div class="dsRow">
      <div class="dsNm">${u.nm}</div>
      <div class="dsBarWrap">
        <div class="dsBar" style="width:${w.toFixed(1)}%">
          <span class="dsPos" style="width:${pp.toFixed(1)}%"></span>
          <span class="dsNeg" style="width:${(100-pp).toFixed(1)}%"></span>
        </div>
      </div>
    </div>`;
  }).join('');
  const tot=us.reduce((a,u)=>a+u.pos+u.neg,0), pos=us.reduce((a,u)=>a+u.pos,0);
  /* ONE SUB-HEADING PER COLUMN, IN THE SAME STYLE, because the two columns answer the
     two halves of one question: how many rows, and how many columns. Written as
     "Rows:" and "Columns:" rather than as a title and a total, so the pair reads as a
     shape — 1,852 x 400 — rather than as two unrelated facts. The row figure is
     COMPUTED from the six use cases below it; the column figure is the dictionary's
     own sum, rounded down to the "400+" the deck says out loud. */
  document.getElementById('dsBody').innerHTML=`
    <div class="dsGrid">
      <div class="dsChart">
        <div class="dsHd"><b>Rows:</b> ${pos.toLocaleString()} of ${tot.toLocaleString()} labelled interesting</div>
        <div class="dsRows">${rows}</div>
        <div class="dsKey">
          <span><i class="lg" style="background:var(--yes)"></i>Interesting</span>
          <span><i class="lg" style="background:var(--no)"></i>Not interesting</span>
        </div>
      </div>
      <div class="dsDict">
        <div class="dsHd"><b>Columns:</b> 400+ features</div>
        ${DICT.map(([ic,h,v])=>`<div class="dsDictGrp">
          <span class="dsDictIc">${icon(ic)}</span>
          <span class="dsDictTx"><b>${h}:</b> ${v}</span></div>`).join('')}
      </div>
    </div>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   06 · HOW IT WORKS, AS A SIDE-SCROLL
   The welcome modal's four slides, laid out horizontally, each carrying the two to
   four tools that step actually uses. The Sankey it replaces was a beautiful
   accounting of one search run and answered a question nobody in the room had asked;
   these four answer "what does it do", which is the one they had.

   The slides come from `WSL` in shared/data.js — the same four the app's welcome
   modal shows, so a change to one changes both. The tool lists are per-slide because
   a flat stack list makes the reader map thirty names onto four steps themselves.
   ══════════════════════════════════════════════════════════════════════════ */
const SLIDE_TOOLS=[
  ['OpenRouter','Claude Sonnet 4.6','Versioned prompts'],
  ['OpenAlex','Semantic Scholar','Crossref','pgvector'],
  ['scikit-learn','MiniLM-L12-v2','Rocchio centroid','Leave-one-out'],
  ['recall@k','WSS@95','FastAPI','React']
];
function renderHow(){
  /* ONE SLIDE, FULL FRAME. Four 420px panels side by side meant four small diagrams
     and the reader deciding where to look; one slide at a time makes the diagram the
     size of the section. The paragraph went with the same edit — a slide read from
     across a room is a title, a picture and the tools under it, and three lines of
     13px prose on it are three lines nobody reads standing up. The prose still exists
     where it is read sitting down, in the app's own welcome modal, from the same WSL. */
  document.getElementById('howTrack').innerHTML=WSL.map((s,i)=>`
    <article class="hwSlide">
      <div class="hwHd"><span class="hwN">0${i+1}</span><h3>${s.t}</h3></div>
      <div class="hwDia">${s.d}</div>
      <div class="hwTools">${(SLIDE_TOOLS[i]||[]).map(nm=>{
        const src=ICO(nm);
        return `<span class="hwTool">${src?`<img src="${src}" alt="">`:''}${nm}</span>`}).join('')}</div>
    </article>`).join('');
  document.getElementById('hwPrev').innerHTML=icon('chevRight');
  document.getElementById('hwNext').innerHTML=icon('chevRight');
  hwSync();
}
/* THE INDEX IS READ OFF THE SCROLLER, never held in a variable — the same rule the
   section counter follows. The track is still a snapped horizontal scroller, so a
   trackpad swipe works exactly as before; the arrows are a second way to do the one
   thing, and if they kept their own idea of "where we are" the two would disagree the
   first time somebody swiped. */
const hwAt=()=>{ const t=document.getElementById('howTrack');
  return Math.round(t.scrollLeft/(t.clientWidth||1)) };
function hwGo(d){
  const t=document.getElementById('howTrack'), w=t.clientWidth||1;
  const i=Math.max(0,Math.min(WSL.length-1, hwAt()+d));
  t.scrollTo({left:i*w, behavior:'smooth'});
}
/* the dots and the two arrows' disabled state, from that same read */
function hwSync(){
  const i=hwAt();
  document.getElementById('hwPrev').disabled = i<=0;
  document.getElementById('hwNext').disabled = i>=WSL.length-1;
  document.getElementById('hwDots').innerHTML=WSL.map((s,k)=>
    `<i class="${k===i?'on':''}" title="${s.t}"></i>`).join('');
}

/* ══════════════════════════════════════════════════════════════════════════
   09 · BASELINE AGAINST ADVANCED
   The one chart on this page whose numbers are SUPPLIED rather than computed. 76
   papers split 60/20/20 leaves fifteen in the test set; drawing it from the
   fixture anyway — to keep the "everything here is computed" line intact — would
   be the worse of the two lies. The frame carries a one-line stamp saying so.
   ══════════════════════════════════════════════════════════════════════════ */
const PERF={
  splits:['Train','Validate','Test'],
  share:[60,20,20],
  /* the names carry what each one IS. "Baseline" and "Advanced" are labels for a
     legend; "Baseline (LogReg)" and "Advanced (CatBoost, Random Forest, LogReg)" are
     the answer to the question anybody in the room is about to ask. */
  series:[ {k:'adv',  nm:'Advanced', sub:'CatBoost · Random Forest · LogReg', v:[0.958,0.906,0.892]},
           {k:'base', nm:'Baseline', sub:'LogReg', v:[0.841,0.820,0.790]} ]
};
function perfChart(){
  /* R is 250 rather than 112: the series labels sit at the end of their own lines
     and each is now a boxed two-line block, so the plot has to stop well short of
     the frame's right edge or the box runs off it. L and B grew for the same reason
     on the other two axes — the axis type went up and had no room. */
  const W=1000, H=520, L=140, R=345, T=62, B=110;
  const iw=W-L-R, ih=H-T-B;
  /* the y-axis starts at 0.70, not 0. A zeroed axis on two series that both sit
     above 0.78 compresses the entire difference into a tenth of the plot; the
     axis is labelled with its own floor so nobody reads the gap as a ratio. */
  const y0=0.70, y1=1.0;
  const x=i=>L+iw*(i/(PERF.splits.length-1));
  const y=v=>T+ih*(1-(v-y0)/(y1-y0));
  const grid=[0.70,0.75,0.80,0.85,0.90,0.95,1.0].map(g=>`
    <line class="pc-g" x1="${L}" y1="${y(g).toFixed(1)}" x2="${W-R}" y2="${y(g).toFixed(1)}"/>
    <text class="pc-ax" x="${L-34}" y="${(y(g)+7).toFixed(1)}" text-anchor="end">${g.toFixed(2)}</text>`).join('');
  const cols=PERF.splits.map((s,i)=>`
    <text class="pc-x" x="${x(i).toFixed(1)}" y="${H-B+50}" text-anchor="middle">${s}</text>
    <text class="pc-xs" x="${x(i).toFixed(1)}" y="${H-B+78}" text-anchor="middle">${PERF.share[i]}% of the data</text>`).join('');
  const lines=PERF.series.map(s=>{
    const d=s.v.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
    const dots=s.v.map((v,i)=>`<circle class="pc-d ${s.k}" cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="7"/>
      <text class="pc-v ${s.k}" x="${x(i).toFixed(1)}" y="${(y(v)-20).toFixed(1)}"
        text-anchor="middle">${v.toFixed(3)}</text>`).join('');
    /* THE SERIES NAME IN A BOX AT THE END OF ITS OWN LINE, which removes the legend
       and removes the reader's job of matching a swatch to a stroke. Two lines: the
       name, then what it is made of. The box is filled in the series' own colour at
       low opacity with a solid border, so it reads as a label belonging to that line
       rather than as floating text that happens to be nearby.
       Boxed in SVG means a <rect> at a hand-computed size — there is no text metric
       available here, so the width comes from the longer of the two strings at a
       measured per-character cost. It is generous by design: too wide is invisible,
       too narrow clips. */
    const bx=x(2)+42, by=y(s.v[2]);
    /* the box is sized from the LONGER of the two lines at that line's own
       per-character cost — the name is 20px sans at ~0.52em, the subtitle 12.5px mono
       at ~0.60em. Two different costs because they are two different fonts, and using
       one for both is how the subtitle came to hang 50 units outside its own box.
       Verified by comparing the text's getBBox against the rect's, not by eye. */
    const PADX=19, PADY=17;                    /* the box's own inside padding */
    const wName=s.nm.length*10.6+PADX*2, wSub=s.sub.length*7.3+PADX*2;
    const bw=Math.min(Math.max(wName,wSub), W-bx-6), bh=72;
    const lbl=`<g>
      <rect class="pc-box ${s.k}" x="${bx.toFixed(1)}" y="${(by-bh/2).toFixed(1)}"
        width="${bw.toFixed(1)}" height="${bh}"/>
      <text class="pc-nm ${s.k}" x="${(bx+PADX).toFixed(1)}" y="${(by-6).toFixed(1)}">${s.nm}</text>
      <text class="pc-sub ${s.k}" x="${(bx+PADX).toFixed(1)}" y="${(by+18).toFixed(1)}">${s.sub}</text></g>`;
    return `<path class="pc-l ${s.k}" d="${d}"/>${dots}${lbl}`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img"
      aria-label="F2 for the baseline and advanced models across train, validate and test">
    ${grid}${cols}
    <text class="pc-ax pc-unit" x="${L-34}" y="${T-22}" text-anchor="end">F2</text>
    ${lines}
  </svg>`;
}
/* No corner statistic. "+10.2 F2 points" was the difference between the two rightmost
   dots, which are eighty pixels apart on the plot and labelled with their own values —
   so the number was a third telling of a thing the chart says twice, and it sat in the
   one corner the reader's eye reaches last. */
function renderPerf(){
  document.getElementById('perfBody').innerHTML=`
    <div class="chartFrame">
      <div class="cfStage plain">${perfChart()}</div>
    </div>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   MOUNTING
   ══════════════════════════════════════════════════════════════════════════ */
/* The same six use cases the bubble chart and the Sankey total, resolved from the
   one `SIX` list in data.js by id. This used to be six name fragments matched as
   substrings, and `'NER'` selected the library's "Don-NER-s 2021 — emicizumab
   dosing in haemophilia A" — a silent failure that renders perfectly with the
   wrong use case in it. Ids cannot do that. */
const cmpPreselect=()=>SIX.map(id=>'uc'+id);

/* ══════════════════════════════════════════════════════════════════════════
   09 · THANK YOU
   The hero again, with one line per section that came before it. A closing slide
   whose bullets are written by hand drifts from the deck the first time a section
   moves; these are a list beside the section they summarise, and the section
   NUMBER is printed with each one so a mismatch is visible rather than latent.
   ══════════════════════════════════════════════════════════════════════════ */
/* Written by hand, one line per section, and NOT numbered. The numbers were there to
   make a drift visible — a bullet whose "05" no longer matched section 05 — but they
   also made an eight-item list read as an eight-step process, which is the wrong shape
   for a summary. `Feature engineering` has no slide of its own; it is the finding that
   sits under EDA and Performance both, and it is here because it is the thing worth
   remembering, not because there is a slide to point at. */
/* A MARK PER POINT, and each one is the mark for the SUBJECT rather than for the
   sentence — `sliders` for feature engineering because the point is that the use-case
   description is the lever, `trend` for performance because the point is a line going
   the right way. Eight distinct glyphs from the inline set, and they follow that file's
   own rule: a mark beside a heading, never meaning on its own. Take the words away and
   nobody could reconstruct these; that is the test, and it is meant to fail. */
const THANKS=[
  ['target',  'Business goal','Read fewer papers, but more relevant science'],
  ['users',   'Stakeholders','Industrial R&D investing $mm for decades'],
  ['gauge',   'Key metric','F2 to catch weak signals earlier'],
  ['archive', 'Dataset','6 use cases, 1800 rows (1k interesting), 400 columns'],
  ['flow',    'How it works','Use case \u2192 Label \u2192 Predict \u2192 Alert'],
  ['shapes',  'EDA','Embeddings are key, but not enough'],
  ['sliders', 'Feature engineering','Use case design is the biggest lever'],
  ['trend',   'Performance','Advanced model >0.1 better on test set']
];
function renderThanks(){
  /* ONE LINE PER POINT — `Title: point`, not a heading with a line under it. Two
     stacked lines made eight bullets read as eight little sections; run together they
     read as eight statements, which is what a summary is, and the same vertical room
     then pays for bigger type. */
  document.getElementById('tyList').innerHTML=THANKS.map(([ic,h,c])=>
    `<li><span class="tyIc">${icon(ic)}</span><span><b>${h}:</b> ${c}</span></li>`).join('');
  /* THE REPOSITORY, TWO WAYS AT ONCE. The QR is for the room — phones up, nobody
     retypes forty characters of `TIRI---The-Data-science-project` correctly — and the
     printed address is for whoever reads this on their own screen, where a QR code is
     useless. Both halves sit inside one <a>, so either one is the same click. */
  /* THE ADDRESS BREAKS WHERE A PATH BREAKS. Capped to the slide's clear middle it has to
     wrap, and `word-break:break-all` wrapped it mid-word — `…-science-proj / ect`, which
     reads as a rendering fault. Split at the last separator instead: the owner on one
     line, the repository on the next, both intact. */
  const addr=REPO.replace(/^https?:\/\//,''), cut=addr.lastIndexOf('/');
  document.getElementById('tyFoot').innerHTML=
    `<a class="tyRepo" href="${REPO}" target="_blank" rel="noopener">
      <img class="tyQR" src="assets/repo-qr.svg" alt="QR code for the repository">
      <span class="tyRepoTx">
        <svg class="ghMark" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">${GH_MARK}</svg>
        <b>${addr.slice(0,cut+1)}<br>${addr.slice(cut+1)}</b></span></a>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   THE TEAM
   Rendered in two places — the hero and the closing slide — from one list, because
   two copies of a pair of names, roles and four URLs is two places for a typo.
   The links open in a new tab with `rel="noopener"`: without it the opened page gets
   a handle on this one through `window.opener`, which is a real if small hole and
   costs one attribute to close.
   ══════════════════════════════════════════════════════════════════════════ */
const TEAM=[
  {n:'Sylvain Fossier', r:'Scientist',    img:()=>IMG_TEAM.sylvain},
  {n:'Warren Fauvel',   r:'Technologist', img:()=>IMG_TEAM.warren}
];
/* THE LINKS ARE GONE FROM THE CARDS. Four URLs under two names on the opening slide
   were four things competing with the two names, and nothing on slide 01 is meant to be
   clicked — the deck has not made its case yet. The one address that survives is the
   repository, on the closing slide, where asking for a click is the point.
   THE FIRST CARD IS MIRRORED so the two portraits meet in the middle rather than each
   sitting left of its own caption with a column of type between them. */
function teamHTML(){
  return TEAM.map((p,i)=>`<figure class="lp${i===0?' flip':''}">
    <span class="lpimg"><img src="${p.img()}" alt="${p.n}"></span>
    <figcaption><b>${p.n}</b><span>${p.r}</span></figcaption></figure>`).join('');
}

/* ══════════════════════════════════════════════════════════════════════════
   09 · THE REPOSITORY, AND THE CONFETTI
   ══════════════════════════════════════════════════════════════════════════ */
const REPO='https://github.com/Sylvain-gitty/TIRI---The-Data-science-project';
/* The GitHub mark is a FILLED brand glyph and cannot come from `shared/icons.js`, which
   is a stroked 24x24 Lucide-contract set — redrawing it in that style would be a worse
   forgery than not drawing it. One path, filled, from the mark's own published outline. */
const GH_MARK='<path d="M12 2.4a9.6 9.6 0 0 0-3 18.7c.5.1.6-.2.6-.5v-1.7c-2.6.6-3.2-1.2-3.2-1.2'+
  '-.4-1.1-1-1.4-1-1.4-.9-.6 0-.6 0-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.3 1.1 2.9.8.1-.6.4-1.1.7-1.4'+
  '-2.1-.2-4.3-1-4.3-4.6 0-1 .4-1.9 1-2.5-.1-.3-.4-1.3.1-2.6 0 0 .8-.3 2.7 1a9.2 9.2 0 0 1 4.9 0'+
  'c1.9-1.3 2.7-1 2.7-1 .5 1.3.2 2.3.1 2.6.6.6 1 1.5 1 2.5 0 3.6-2.2 4.4-4.3 4.6.4.4.7 1.1.7 2.2'+
  'v3.2c0 .3.1.6.7.5A9.6 9.6 0 0 0 12 2.4z"/>';

/* ── the confetti ───────────────────────────────────────────────────────────
   Click the mark and the slide throws a handful of the brand's own colours. Forty lines
   of canvas rather than a dependency: it is one gesture on one slide, and a confetti
   library is 20KB plus a supply chain for it.

   Deterministic is NOT wanted here. Every other random-looking thing in this prototype
   is a fixed hash because it has to be reproducible evidence; this is the one thing on
   the page that is meant to be different every time you press it.

   Under `prefers-reduced-motion` it does nothing at all. A silent no-op is the honest
   answer — the mark carries no other function, so nobody loses anything, and a
   "reduced-motion confetti burst" is still a confetti burst. */
const CONF_COL=['hsl(145,38%,23%)','hsl(145,45%,15%)','hsl(96,40%,34%)',
                'hsl(44,86%,52%)','hsl(40,74%,42%)','hsl(52,74%,60%)'];
let confParts=[], confRAF=null;
function confetti(){
  if(matchMedia('(prefers-reduced-motion:reduce)').matches) return;
  const cv=document.getElementById('tyConfetti'), mk=document.getElementById('tyMarkImg');
  const box=cv.getBoundingClientRect(), m=mk.getBoundingClientRect();
  /* sized in DEVICE pixels and scaled back with a transform, or every edge is soft on a
     retina display */
  const dpr=Math.min(2,window.devicePixelRatio||1);
  cv.width=box.width*dpr; cv.height=box.height*dpr;
  const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  /* burst from the middle of the mark, in the canvas's own coordinates */
  const ox=m.left-box.left+m.width/2, oy=m.top-box.top+m.height/2;
  for(let i=0;i<150;i++){
    const a=Math.random()*Math.PI*2, sp=3+Math.random()*11;
    confParts.push({x:ox, y:oy, vx:Math.cos(a)*sp, vy:Math.sin(a)*sp-4,
      w:5+Math.random()*7, h:3+Math.random()*5, rot:Math.random()*6.3,
      vr:(Math.random()-0.5)*0.4, life:1,
      col:CONF_COL[(Math.random()*CONF_COL.length)|0]});
  }
  if(confRAF) return;                      /* a second click feeds the loop already running */
  const step=()=>{
    ctx.clearRect(0,0,box.width,box.height);
    confParts=confParts.filter(q=>{
      q.vy+=0.22;                          /* gravity */
      q.vx*=0.99; q.vy*=0.99;              /* air */
      q.x+=q.vx; q.y+=q.vy; q.rot+=q.vr;
      q.life-=0.008;
      if(q.life<=0 || q.y>box.height+30) return false;
      ctx.save();
      ctx.translate(q.x,q.y); ctx.rotate(q.rot);
      ctx.globalAlpha=Math.max(0,Math.min(1,q.life*1.6));
      ctx.fillStyle=q.col;
      ctx.fillRect(-q.w/2,-q.h/2,q.w,q.h);
      ctx.restore();
      return true;
    });
    if(confParts.length) confRAF=requestAnimationFrame(step);
    else { confRAF=null; ctx.clearRect(0,0,box.width,box.height) }
  };
  confRAF=requestAnimationFrame(step);
}

function renderLanding(){
  document.getElementById('lmark').insertAdjacentHTML('afterbegin',icon('radar','lg')+' ');
  document.querySelectorAll('.lnext .nx').forEach(e=>e.innerHTML=icon('arrowDown'));

  document.getElementById('lmarkImg').src=IMG_MARK;
  document.getElementById('lteam').innerHTML=teamHTML();
  document.getElementById('radarWrap').innerHTML=radarSVG();
  /* the closing slide has no team cards and no radar any more: the two of them ARE the
     background, cut out and half off each edge, which says what the cards said with
     none of their height. The mark is the confetti's trigger. */
  document.getElementById('tyMarkImg').src=IMG_MARK;
  renderThanks();

  /* 02 — the same radar as the hero, at a fraction of the opacity. Same function,
     so the two backgrounds cannot drift into being two different diagrams. */
  document.getElementById('radarBg2').innerHTML=radarSVG();
  document.getElementById('lofi').innerHTML=lofiHTML();

  /* 03 – 08 */
  renderFlowSec();
  renderWeak();
  renderDataset();
  renderHow();
  CMP.sel=cmpPreselect();
  renderCompare('caBody',{lean:true, enc:'colour'});
  renderPerf();
}

function lgo(i){ document.getElementById('ls'+i).scrollIntoView({behavior:'smooth'}) }
/* The app is a separate document now, so this is a navigation rather than a
   screen change. The welcome modal used to be triggered from here; it is now
   triggered on app.html's first load from the same localStorage flag, so a
   visitor who has already dismissed it does not meet it twice. */
function launch(){ location.href='app.html' }

renderLanding();

/* The section counter reads the scroller rather than the last button pressed, so
   a trackpad and the buttons can never disagree about where you are. */
const NSEC=document.querySelectorAll('.lsec').length;
document.getElementById('lscroll').addEventListener('scroll',e=>{
  const h=e.target.clientHeight||1, n=Math.min(NSEC,Math.round(e.target.scrollTop/h)+1);
  document.getElementById('lsecn').textContent=('0'+n).slice(-2)+' / '+('0'+NSEC).slice(-2);
});
