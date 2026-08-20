/* ── A SPACE THE DESCRIPTIONS ACTUALLY LIVE IN ────────────────────────────
   The Compare frame used to place and score every use case in PROJ — the
   vocabulary of the fixture's 76 CEMENT papers. That is fine for the one use
   case those papers belong to and near-useless for the rest: a soil-microbiome
   description had four of its fourteen words in the space and a photovoltaics one
   had two, so five of six similarities floored at 0.00 and the layout was being
   driven by rounding noise. A chart whose headline says "related questions land
   near each other" cannot be computed in a space where nothing is related to
   anything.

   So: a second tf-idf, over the use-case DESCRIPTIONS themselves — all nine live
   ones and all twenty-eight from the library, 37 short documents. Every use case
   has full coverage by construction, and the similarity that comes out is a real
   measurement over real text rather than an artefact of which corpus happened to
   be loaded.

   This is closer to the product, not further from it: `bge-small-en-v1.5` embeds
   any string into one space, so the app never had this problem. The prototype had
   it because a single HTML file cannot carry a sentence encoder.

   WHAT IT STILL CANNOT DO, and the section says so: tf-idf over 37 short texts
   sees shared WORDS, not shared meaning. Two descriptions that say the same thing
   in different vocabulary score zero here and would not under a real encoder.
   ══════════════════════════════════════════════════════════════════════════ */
/* LAZY. It reads `cmpAll()` (defined in charts.js, which loads after this file)
   and `stem` (a const arrow further down this one, so not hoisted). An eager IIFE
   here throws on both counts — and the previous version of exactly this mistake,
   the eager GEN_MASK below, silently produced a wrong answer instead of throwing.
   Built once on first use and kept. */
let _dp=null;
const DESCPROJ=()=>_dp||(_dp=(()=>{
  const docs=cmpAll().map(a=>a.txt);
  const N=docs.length;
  const tok=s=>new Set((String(s).toLowerCase().match(/[a-z][a-z-]{3,}/g)||[])
    .map(stem).filter(w=>!STOP.has(w)));
  const sets=docs.map(tok);
  const df={}; sets.forEach(s=>s.forEach(w=>df[w]=(df[w]||0)+1));
  /* df>=2 only. A word appearing in exactly one description can never contribute
     to a similarity — it just inflates that document's norm and pushes every
     cosine it takes part in down, which is how "no shared vocabulary" and "not
     alike" got confused in the first version. */
  const vocab=Object.keys(df).filter(w=>df[w]>=2).sort();
  const idx={}; vocab.forEach((w,j)=>idx[w]=j);
  const D=vocab.length;
  const vec=s=>{ const v=new Float64Array(D); let hit=0, tot=0;
    tok(s).forEach(w=>{ tot++; if(w in idx){ v[idx[w]]=Math.log(N/df[w]); hit++ } });
    let nr=0; for(let j=0;j<D;j++) nr+=v[j]*v[j]; nr=Math.sqrt(nr)||1;
    for(let j=0;j<D;j++) v[j]/=nr;
    return {v,hit,tot};
  };
  return {dim:D, N, vocab, raw:vec, vec:s=>vec(s).v};
})());

/* ── LAYING THE SELECTION OUT ───────────────────────────────────────────────
   The job: place N use cases so that DISTANCE MEANS DISSIMILARITY. Two that ask
   related questions should sit on top of each other; two that ask nothing alike
   should sit as far apart as the frame allows.

   THREE ATTEMPTS, AND WHY THE THIRD IS THE RIGHT ONE.

   (1) PCA of the description vectors, projected to two components. This is what a
       "map of the pool" does and it is wrong here. PCA finds the directions of
       greatest variance in the FEATURE space; with 37 short documents over a
       52-word shared vocabulary, most of that variance is which rare word each one
       happens to use, so four unrelated use cases all landed near the origin —
       indistinguishable from each other and from the mean. Their labels stacked.

   (2) Snapping the PCA output to a 3x2 grid. Separation guaranteed, labels legible
       — and distance stopped meaning anything at all, which threw away the one
       thing the figure is for. A grid says "here are six use cases"; it cannot say
       "these two are the same question".

   (3) STRESS MAJORIZATION on the similarity matrix itself, which is what MDS
       actually is. Target distance for a pair is 1 - cosine: a pair at 0.73
       similarity wants to be 0.27 apart and overlap; a pair at 0.00 wants to be a
       full 1.0 apart and takes opposite corners. So the layout is driven by the
       numbers the reader is being shown, not by a side effect of the feature space,
       and it fills the frame for free — the zero-similarity pairs push each other
       to the edges.

   Deterministic throughout: the initial configuration is a fixed circle, never
   Math.random, so the same selection always produces the same picture. */
function cmpLayout(sel){
  const n=sel.length, DP=DESCPROJ();
  if(n===1) return [[0.5,0.5]];
  const V=sel.map(c=>DP.raw(c.txt).v);
  const cos=(a,b)=>{ let d=0; for(let j=0;j<a.length;j++) d+=a[j]*b[j]; return d };
  /* the target distances, straight off the similarity the table would print */
  const D=[];
  for(let a=0;a<n;a++){ D.push([]);
    for(let b=0;b<n;b++) D[a].push(a===b?0:1-Math.max(0,Math.min(1,cos(V[a],V[b]))));
  }
  if(n===2) return [[0.5-D[0][1]/2*0.74,0.5],[0.5+D[0][1]/2*0.74,0.5]];

  /* fixed circular start: evenly spaced, so no pair begins coincident and the
     iteration never has to invent a direction to separate one */
  let P=[];
  for(let i=0;i<n;i++){ const a=2*Math.PI*i/n;
    P.push([0.5+0.32*Math.cos(a), 0.5+0.32*Math.sin(a)]) }

  /* SMACOF: each point moves to the weighted mean of where every other point
     "wants" it, along the current bearing at the target distance. Converges
     monotonically on the stress, which is why it needs no step size. */
  for(let it=0;it<300;it++){
    const Q=[];
    for(let a=0;a<n;a++){
      let sx=0, sy=0, w=0;
      for(let b=0;b<n;b++){
        if(a===b) continue;
        let dx=P[a][0]-P[b][0], dy=P[a][1]-P[b][1];
        let d=Math.hypot(dx,dy);
        if(d<1e-9){ dx=Math.cos(a*2.399+b); dy=Math.sin(a*2.399+b); d=1 }
        /* where b wants a to be: b's position plus the target distance along the
           current bearing from b to a */
        sx += P[b][0] + D[a][b]*dx/d;
        sy += P[b][1] + D[a][b]*dy/d;
        w++;
      }
      Q.push([sx/w, sy/w]);
    }
    P=Q;
  }
  /* Fit the result into the frame on its own bounding box, independently per axis.
     Per-axis is the "transform the axes so the space is used" step: an MDS solution
     is only defined up to rotation and scale, so stretching each axis to the frame
     costs nothing true and stops a solution that happens to come out long and thin
     from using a fifth of the plot. The ORDERING of distances survives; their
     absolute ratios do not, which is why the plot says to read the grouping. */
  const fit=(vals,lo,hi)=>{ const a=Math.min(...vals), b=Math.max(...vals), sp=(b-a)||1;
    return vals.map(v=>lo+(v-a)/sp*(hi-lo)) };
  const xs=fit(P.map(p=>p[0]),0.13,0.87), ys=fit(P.map(p=>p[1]),0.16,0.84);
  return xs.map((x,i)=>[x,ys[i]]);
}
const FILLER=new Set(['data','analysis','results','study','using','used','approach','method','novel','system']);
const STOP=new Set(('the a an and or of for in on to with by from is are we our this that these those it its as at be'+
 ' can could will would should may might have has had do does did not no new using used study studies paper research'+
 ' work results between their there here than then when where which who what how why all any both each few more most'+
 ' other some such only own same so me my you your us are that').split(/\s+/));
const SCAFFOLD=new Set(('across reported studied measured observed compared presented obtained showed found shown given'+
 ' tested evaluated investigated performed conducted between within during higher lower total authors study studies'+
 ' paper papers review reviews three four five six seven eight nine eleven twelve fourteen sixteen twenty thirty forty'+
 ' factors capacity content effects effect analysis results method methods against their which these those there'+
 ' remains largely rather often widely several because therefore however whose existing sizes size model models'+
 ' value values range ranges level levels case cases type types source sources system systems process processes'+
 ' condition conditions property properties material materials sample samples specimen specimens'+
 ' below above roughly relative approximately nearly almost about across within throughout'+
 ' control controls beyond primary published reported reporting current currently recent'+
 ' available limited scarce dominant governing leading common commonly widely largely').split(/\s+/));
/* Crude singularisation, so `plant` and `plants` are not offered as two separate
   judgements — asking the same question twice is the fastest way to teach someone
   the suggestions are not worth reading. */
const stem=w=>w.endsWith('ies')?w.slice(0,-3)+'y':(w.endsWith('sses')?w.slice(0,-2):(w.endsWith('s')&&!w.endsWith('ss')?w.slice(0,-1):w));
const tokOf=id=>new Set(((PAPERS[id].t+' '+PAPERS[id].ab).toLowerCase().match(/[a-z][a-z-]{4,}/g)||[]).map(stem));
/* Words most of the POOL contains cannot tell any part of it from another — this is
   the guard that catches `cement`, which an earlier version offered because the
   description says "clinker" and "binders" but never "cement". */
const POOLDF=(()=>{ const d={}; PAPERS.forEach(p=>tokOf(p.id).forEach(w=>d[w]=(d[w]||0)+1));
  return new Set(Object.entries(d).filter(([,n])=>n/PAPERS.length>0.14).map(([w])=>w)); })();
/* ── A REAL PROJECTION, NOT A HASH ──────────────────────────────────────────
   v5 placed every dot with a deterministic hash of the title, nudged toward one
   of two clouds BY ITS GOLD LABEL, and the caption admitted it. Honest, but
   circular: a map drawn from the answer key can only ever show structure that
   was put there on purpose, and structure the analyst did NOT put there is the
   only thing this panel exists to show. Two clean clusters would have been the
   fixture agreeing with itself.

   So this is an actual two-component PCA over the fixture's own text:

     tf-idf bag of words  →  mean-centre  →  first two principal components by
     POWER ITERATION with deflation  →  project every paper onto them.

   Power iteration is the whole trick and it is four lines: repeatedly apply
   XᵀX to a vector and renormalise, and it converges on the direction of
   greatest variance — which IS the first principal component. Strip that
   direction out of every row ("deflate") and the same loop finds the second.
   No eigendecomposition, no library, ~40 lines.

   Deterministic on purpose: the start vector is a fixed function of the index,
   never Math.random, so the map is pixel-identical on every load. A map that
   reshuffles itself between visits is one nobody can learn to read.

   In the app the input matrix is the 384-d embedding block the ranker already
   holds, not word counts. The maths below does not change — which is exactly
   why it was worth doing properly here rather than faking it again. */
const PROJ=(()=>{
  const docs=PAPERS.map(p=>tokOf(p.id));
  const N=docs.length;
  /* Vocabulary. A word in only one paper can describe a point but never a
     direction; a word in most of them separates nothing. Both are dropped. */
  const df={}; docs.forEach(d=>d.forEach(w=>df[w]=(df[w]||0)+1));
  const vocab=Object.keys(df).filter(w=>df[w]>=2&&df[w]<N*0.55&&!STOP.has(w)).sort();
  const idx={}; vocab.forEach((w,j)=>idx[w]=j);
  const D=vocab.length;
  /* Rows are idf-weighted and L2-normalised, so a long abstract does not simply
     land further from the origin than a short one saying the same thing. */
  const X=docs.map(d=>{
    const v=new Float64Array(D);
    d.forEach(w=>{ if(w in idx) v[idx[w]]=Math.log(N/df[w]) });
    let nr=0; for(let j=0;j<D;j++) nr+=v[j]*v[j];
    nr=Math.sqrt(nr)||1; for(let j=0;j<D;j++) v[j]/=nr;
    return v;
  });
  const mean=new Float64Array(D);
  X.forEach(v=>{ for(let j=0;j<D;j++) mean[j]+=v[j]/N });
  X.forEach(v=>{ for(let j=0;j<D;j++) v[j]-=mean[j] });
  const comp=()=>{
    let v=new Float64Array(D);
    /* a fixed, arbitrary, non-degenerate start — the classic hash-sine, used
       here precisely because it is NOT random */
    for(let j=0;j<D;j++) v[j]=(Math.sin(j*12.9898)*43758.5453)%1;
    for(let it=0;it<80;it++){
      const w=new Float64Array(D);
      X.forEach(r=>{ let dot=0; for(let j=0;j<D;j++) dot+=r[j]*v[j];
                     for(let j=0;j<D;j++) w[j]+=r[j]*dot });
      let nr=0; for(let j=0;j<D;j++) nr+=w[j]*w[j];
      nr=Math.sqrt(nr)||1; for(let j=0;j<D;j++) v[j]=w[j]/nr;
    }
    return v;
  };
  const pc1=comp();
  /* Score on pc1 BEFORE deflating — deflation destroys the rows it reads. */
  const s1=X.map(r=>{ let d=0; for(let j=0;j<D;j++) d+=r[j]*pc1[j]; return d });
  X.forEach(r=>{ let d=0; for(let j=0;j<D;j++) d+=r[j]*pc1[j];
                 for(let j=0;j<D;j++) r[j]-=d*pc1[j] });
  const pc2=comp();
  const s2=X.map(r=>{ let d=0; for(let j=0;j<D;j++) d+=r[j]*pc2[j]; return d });
  /* Rescale each axis to the panel. This is a linear stretch, so it preserves
     every ratio a reader could legitimately take off the picture. */
  const norm=a=>{ const lo=Math.min(...a), hi=Math.max(...a), sp=(hi-lo)||1;
                  return v=>(v-lo)/sp };
  const n1=norm(s1), n2=norm(s2), m={};
  PAPERS.forEach((p,i)=>{ m[p.id]=[0.05+n1(s1[i])*0.90, 0.07+n2(s2[i])*0.86] });
  /* Exposed for the landing page's radar, which plots the same coordinates in
     polar form — the blips on the marketing page are the fixture's real papers.

     `embed(text)` is the other half, and it is what makes Compare honest: any
     string can be pushed through the SAME vocabulary, the SAME idf weights, the
     SAME mean-centring and the SAME two components, so a use case's description
     lands in the frame the papers already live in. Two collections sitting near
     each other on that plot are near each other because their own words are. */
  m._vocab=vocab.length;
  /* The RAW normalised vector, plus how much of the string this space can even
     see. Similarity is measured on this, not on the centred version: centring is
     what PCA needs, and cosine between mean-centred sparse vectors is not text
     similarity — it is similarity-to-the-mean, which floors and reorders every
     pair. Coverage rides along because a 0.02 between two unrelated topics and a
     0.02 between two related ones the vocabulary cannot see are different facts,
     and only one of them is about the topics. */
  m.raw=txt=>{
    const toks=new Set((String(txt).toLowerCase().match(/[a-z][a-z-]{4,}/g)||[]).map(stem));
    const v=new Float64Array(D); let hit=0;
    toks.forEach(w=>{ if(w in idx){ v[idx[w]]=Math.log(N/df[w]); hit++ } });
    let nr=0; for(let j=0;j<D;j++) nr+=v[j]*v[j]; nr=Math.sqrt(nr)||1;
    for(let j=0;j<D;j++) v[j]/=nr;
    return {v,hit,tot:toks.size};
  };
  /* the centred vector for any string, in the papers' own feature space */
  m.vec=txt=>{
    const toks=new Set((String(txt).toLowerCase().match(/[a-z][a-z-]{4,}/g)||[]).map(stem));
    const v=new Float64Array(D);
    toks.forEach(w=>{ if(w in idx) v[idx[w]]=Math.log(N/df[w]) });
    let nr=0; for(let j=0;j<D;j++) nr+=v[j]*v[j]; nr=Math.sqrt(nr)||1;
    for(let j=0;j<D;j++) v[j]=v[j]/nr-mean[j];
    return v;
  };
  m.embed=txt=>{
    const v=m.vec(txt);
    let a=0,b=0; for(let j=0;j<D;j++){ a+=v[j]*pc1[j]; b+=v[j]*pc2[j] }
    return [0.05+n1(a)*0.90, 0.07+n2(b)*0.86];
  };
  m.dim=D;
  return m;
})();
const XY=PROJ;
/* the tf-idf rows again, kept for the model scores in the tune panel */
const VEC=(()=>{
  const docs=PAPERS.map(p=>tokOf(p.id)), N=docs.length;
  const df={}; docs.forEach(d=>d.forEach(w=>df[w]=(df[w]||0)+1));
  const vocab=Object.keys(df).filter(w=>df[w]>=2&&!STOP.has(w)).sort();
  const idx={}; vocab.forEach((w,j)=>idx[w]=j);
  const rows=docs.map(d=>{ const v=new Float64Array(vocab.length);
    d.forEach(w=>{ if(w in idx) v[idx[w]]=Math.log(N/df[w]) });
    let nr=0; for(let j=0;j<v.length;j++) nr+=v[j]*v[j]; nr=Math.sqrt(nr)||1;
    for(let j=0;j<v.length;j++) v[j]/=nr; return v });
  return {rows,idx,dim:vocab.length};
})();
const cosv=(a,b)=>{ let d=0; for(let j=0;j<a.length;j++) d+=a[j]*b[j]; return d };
const MODELS=[
 ['base','Baseline','no labels at all — every paper scored against your description, which is the order the pool arrived in'],
 ['gen','General','stands in for a model fitted on the 28 public screening projects: it knows what a substantive paper looks like and nothing about your topic'],
 ['you','Your Model','fitted on your labels and nothing else, and scored by leaving each labelled paper out of the model that judges it']];

function vnorm(c){ let nr=0; for(let j=0;j<c.length;j++) nr+=c[j]*c[j];
  nr=Math.sqrt(nr)||1; for(let j=0;j<c.length;j++) c[j]/=nr; return c }
function rocchio(posIds,negIds){
  const D=VEC.dim, c=new Float64Array(D);
  posIds.forEach(id=>{ const v=VEC.rows[id]; for(let j=0;j<D;j++) c[j]+=v[j]/posIds.length });
  /* negatives pull the centroid away at a fraction of the weight — the classic
     Rocchio asymmetry, and it matches model.py's ladder: one positive is worth
     far more than one rejection to a centroid. */
  if(negIds.length) negIds.forEach(id=>{ const v=VEC.rows[id];
    for(let j=0;j<D;j++) c[j]-=0.45*v[j]/negIds.length });
  return vnorm(c);
}
function qvec(){
  const D=VEC.dim, c=new Float64Array(D);
  let ws=(compiled().toLowerCase().match(/[a-z][a-z-]{4,}/g)||[]).map(stem).filter(w=>w in VEC.idx);
  /* If the form was never filled — the demo strip can jump straight here — fall
     back to the use case's own objective, which is still label-free. */
  if(ws.length<3) ws=(UCS[0].ob.toLowerCase().match(/[a-z][a-z-]{4,}/g)||[]).map(stem).filter(w=>w in VEC.idx);
  ws.forEach(w=>c[VEC.idx[w]]+=1);
  return vnorm(c);
}
/* ── THE DESCRIPTION, AS ONE STRING ───────────────────────────────────────
   Shared rather than app-only because the description is the INPUT to the
   baseline model: `qvec()` embeds it, and the landing page's weak-signals
   section scores against it. It is a pure function of `S.ans` — it does not
   read the question list — so it travels with the maths, not with the form. */
function compiled(){
  const a=S.ans, sys=(a.systems||[]).join(', '), out=(a.outcome||[]).join(', ');
  const num=Array.isArray(a.numbers)? a.numbers.join(', ') : (a.numbers||'');
  const near=Array.isArray(a.near)? a.near.join(', ') : (a.near||'');
  if(!sys&&!out&&!num) return '';
  let s='';
  if(sys) s+=`${sys.charAt(0).toUpperCase()+sys.slice(1)} for structural application`;
  if(a.kind==='A solution to a problem') s+=', as routes to lower embodied carbon';
  s+='. ';
  if(out) s+=`${out.charAt(0).toUpperCase()+out.slice(1)} are the properties that matter. `;
  if(num) s+=`${num.charAt(0).toUpperCase()+num.slice(1)}. `;
  if(near) s+=`Not: ${near}.`;
  return s.trim();
}

/* ── the library model ──────────────────────────────────────────────────
   A model fitted on 28 other people's screening projects cannot be fitted inside
   a single-file prototype, so this is a STAND-IN — and it is built to behave the
   way the real thing would rather than to a target AUC.

   The construction: a centroid fitted on a fixed half of the fixture, then
   restricted to the vocabulary COMMON TO THE WHOLE POOL and stripped of every
   word the analyst named. What is left is scaffolding — `tested`, `performance`,
   `across`, `effect`. So it can tell a substantive empirical paper from a thin
   one and it does not know what a binder is, which is exactly the shape of a
   model trained on somebody else's question. Its AUC is then MEASURED from those
   scores, not chosen. */
const GEN_TRAIN=PAPERS.filter(p=>p.id%2===0);
/* LAZY, AND THE SPLIT IS WHAT FOUND IT. This was an eager IIFE. It ran at load,
   when `S.ans` is still {} and `compiled()` therefore returns '' — so `mine` was
   always the empty set and the mask never actually removed the analyst's own
   vocabulary. The comment above described a subtraction the code was not doing:
   the same species of bug as a fabricated measurement, because the number it
   produced was real, it just was not the number the comment claimed.
   Memoised on the description text, so it recomputes exactly when that changes
   and no more often. */
const DF_ALL=(()=>{ const d={}; PAPERS.forEach(p=>tokOf(p.id).forEach(w=>d[w]=(d[w]||0)+1)); return d })();
let _genMask={key:null,idx:null};
function genMask(){
  const desc=compiled();
  if(_genMask.key===desc) return _genMask.idx;
  const mine=new Set((desc.toLowerCase().match(/[a-z][a-z-]{4,}/g)||[]).map(stem));
  _genMask={key:desc, idx:Object.entries(DF_ALL)
    .filter(([w,n])=>n/PAPERS.length>=0.20 && !mine.has(w) && (w in VEC.idx))
    .map(([w])=>VEC.idx[w])};
  return _genMask.idx;
}
function scoresFor(m){
  const out=new Float64Array(PAPERS.length);
  if(m==='base'){ const c=qvec(); PAPERS.forEach(p=>out[p.id]=cosv(VEC.rows[p.id],c)); return out }
  if(m==='gen'){
    const full=rocchio(GEN_TRAIN.filter(p=>p.g==='pos').map(p=>p.id),
                       GEN_TRAIN.filter(p=>p.g==='neg').map(p=>p.id));
    const c=new Float64Array(VEC.dim); genMask().forEach(j=>c[j]=full[j]); vnorm(c);
    PAPERS.forEach(p=>out[p.id]=cosv(VEC.rows[p.id],c)); return out;
  }
  const pos=Object.keys(S.lab).filter(i=>S.lab[i]==='pos').map(Number);
  const neg=Object.keys(S.lab).filter(i=>S.lab[i]==='neg').map(Number);
  if(pos.length<3){ const r=scoresFor('gen'); r.fell='you'; return r }
  /* LEAVE ONE OUT, and this is not a nicety. Scoring a paper with a centroid
     that contains that paper is in-sample: on this fixture it reads 0.99 where
     the honest number is 0.86, and a panel whose whole job is choosing an
     operating point cannot be drawn on the optimistic curve. Every LABELLED
     paper is therefore scored by a centroid refitted without it; unlabelled
     papers have nothing to leak and use the full fit. That is one score vector,
     which matters — the map, the slider and the numbers must all be reading the
     same one or the highlighted dots stop matching the count beside them. */
  const full=rocchio(pos,neg);
  PAPERS.forEach(p=>{
    if(p.id in S.lab){
      const c=rocchio(pos.filter(i=>i!==p.id), neg.filter(i=>i!==p.id));
      out[p.id]=cosv(VEC.rows[p.id],c);
    } else out[p.id]=cosv(VEC.rows[p.id],full);
  });
  return out;
}
/* Mann-Whitney AUC with tied ranks averaged. Ties matter: a sparse bag of words
   gives a lot of papers a score of exactly zero, and counting those as wins
   would inflate every number on the panel. */
function aucOf(sc){
  const ids=PAPERS.map(p=>p.id).sort((a,b)=>sc[a]-sc[b]);
  const rank={}; let i=0;
  while(i<ids.length){ let j=i;
    while(j+1<ids.length&&sc[ids[j+1]]===sc[ids[i]]) j++;
    const r=(i+j)/2+1; for(let t=i;t<=j;t++) rank[ids[t]]=r; i=j+1 }
  const np=PAPERS.filter(p=>p.g==='pos').length, nn=PAPERS.length-np;
  const sum=PAPERS.filter(p=>p.g==='pos').reduce((a,p)=>a+rank[p.id],0);
  return (sum-np*(np+1)/2)/(np*nn);
}
/* `items` defaults to the fixture, which is every caller in the app. The landing
   page's Key metric section passes its own sample — same sweep, same Fβ optima,
   same clamping, over a different set of {id,g} records. Generalising the MATHS
   rather than copying it is the whole point: a second precision/recall sweep with
   its own rounding is how two figures on one page start disagreeing. */
function curveFor(sc,items){
  const IT=items||PAPERS;
  const order=IT.map(p=>p.id).sort((a,b)=>sc[b]-sc[a]);
  const byId={}; IT.forEach(p=>byId[p.id]=p);
  const P=IT.filter(p=>p.g==='pos').length, pts=[]; let tp=0;
  order.forEach((id,i)=>{
    if(byId[id].g==='pos') tp++;
    const n=i+1, pr=tp/n, rc=tp/P;
    pts.push({n,pr,rc,order,
      f1:(pr+rc)?2*pr*rc/(pr+rc):0,
      f2:(4*pr+rc)?5*pr*rc/(4*pr+rc):0});
  });
  return pts;
}
/* Ties broken in each metric's own direction: on a plateau F1 takes the tightest
   point that achieves the maximum and F2 the loosest, because that is what the
   two of them mean. Without it a flat stretch hands both marks to whichever end
   the reduce happened to reach first. */
const argmax=(pts,k,last)=>pts.reduce((b,p,i)=>(last? p[k]>=pts[b][k] : p[k]>pts[b][k])?i:b,0);

/* ── THE FIVE SETTINGS ON THE MODEL SCREEN ───────────────────────────────
   A range input asks the analyst to choose a number on a curve they cannot see
   the shape of. These are the five points on that curve worth having, each one
   swept from the data rather than placed:

     High precision   the tightest cut that still finds a quarter of what matters
     F0.5             precision counted double
     F1               a wasted read and a miss cost the same
     F2               recall counted double
     High recall      the least reading that reaches 95% — WSS@95's point

   ⚠️ Warren's brief calls the precision-weighted one "F0 (Precision x 2)". F0 is
   precision alone (β=0 drops recall from the formula entirely); the setting that
   weights precision twice as heavily is F0.5. The label uses F0.5 with the plain
   words beside it, because the number in the name has to match the number in the
   formula or the tooltip teaches something false.

   Fβ = (1+β²)·P·R / (β²·P + R) — one function, five values of β, no special cases. */
/* NAMED FOR THE SCALE THEY SIT ON. "High precision" and "High recall" were the ends
   of a five-stop range whose middle three are Fβ scores — so the row read as two
   adjectives and three formulas rather than as one series. */
const FSET=[
 ['hp','Precision','only what it is surest of',null],
 ['f05','F0.5','precision counts double',0.5],
 ['f1','F1','balanced',1],
 ['f2','F2','recall counts double',2],
 ['hr','Recall','miss as little as possible',null]];
function fbeta(p,b){ const B=b*b; return (B*p.pr+p.rc)? (1+B)*p.pr*p.rc/(B*p.pr+p.rc) : 0 }
/* one sweep, five answers, and the ordering is enforced rather than hoped for */
function fPoints(pts){
  const at={};
  at.hp=(()=>{ const i=pts.findIndex(p=>p.rc>=0.25); return i<0?0:i })();
  [['f05',0.5],['f1',1],['f2',2]].forEach(([k,b])=>{
    at[k]=pts.reduce((best,p,i)=>fbeta(p,b)>fbeta(pts[best],b)?i:best,0) });
  at.hr=(()=>{ const i=pts.findIndex(p=>p.rc>=0.95); return i<0?pts.length-1:i })();
  /* monotone by construction of the metrics, but clamped anyway: a control whose
     "tighter" option reads looser than its neighbour is lying about itself, and
     that has already happened once in this file. */
  let prev=0;
  FSET.forEach(([k])=>{ at[k]=Math.max(at[k],prev); prev=at[k] });
  return at;
}

/* ── THE THREE NAMED OPERATING POINTS (the alert's own control) ────────────────────────────────────
   One definition, used by the tune panel, the pool map's three contours and the
   alert's cut-off — because they are the same choice made in three places, and
   three different vocabularies for one choice is how a product teaches its users
   that the words do not mean anything.

   Each is defined in the metric the product actually ships — recall at a share of
   the pool, per recall.py and metrics.py — never as a probability threshold:
     TIGHT     the least reading that still reaches half of what matters
     BALANCED  the F1 optimum — a wasted read and a miss cost the same
     LOOSE     the least reading that reaches 95%, which is WSS@95's operating
               point and the one review methodology argues about

   All three are swept from the curve; none is placed by hand.

   ⚠️ THE FIRST VERSION OF THIS DEFINED TIGHT BY PRECISION ("9 in 10 of what it
   hands you is worth reading") AND IT COLLIDED WITH BALANCED — both landed on the
   same 20 papers, because a good model's F1 optimum already sits at 95%
   precision. A three-way control with two options that do the same thing is worse
   than a two-way one: it teaches the user that the words do not mean anything.
   Defining all three in ONE metric family fixes it structurally — recall is
   monotone in the cut-off, so a 50% target cannot land past a 95% target. The
   F1 optimum is then clamped INTO that interval rather than the ends being
   clamped towards it, which is what made them collapse. */
function opPoints(){
  const pos=Object.keys(S.lab).filter(i=>S.lab[i]==='pos').length;
  const pts=curveFor(scoresFor(pos>=3?'you':'gen'));
  const firstAt=t=>{ const i=pts.findIndex(p=>p.rc>=t); return i<0? pts.length-1 : i };
  const tight=firstAt(0.50), loose=firstAt(0.95);
  const bal=Math.min(Math.max(argmax(pts,'f1'),tight),loose);
  return {tight:pts[tight],balanced:pts[bal],loose:pts[loose],n:pts.length,
    /* the degenerate case: one card carries the jump from under half to over 95%.
       Then all three ARE the same setting, and the control has to say so rather
       than offer three identical choices. */
    collapsed:pts[tight].n===pts[loose].n};
}
