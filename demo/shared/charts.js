/* ── use case analysis ────────────────────────────────────────────────────
   Two tabs. OVERVIEW is a proportional-AREA comparison of three collections;
   COMPARE places their descriptions in the projection the papers already live in.

   Two things about the overview worth getting right, both of them ways a bubble
   chart normally lies:

   (1) AREA, NOT RADIUS. Radius goes as sqrt(value) or the eye reads a 95x
       difference as a 9,000x one.

   (2) THE CENTRES DO NOT MOVE WHEN YOU TOGGLE. Laying the discs out from the
       CURRENT radii makes every centre shift as soon as the measure changes, so
       the reader tracks travel instead of size — the one comparison the control
       exists to make. Centres are computed from each collection's LARGEST radius
       across BOTH measures, once, so toggling only ever grows and shrinks.

   The slices are positive and negative only. An earlier version also drew
   "passed" and "never opened", which made the workspace's disc a four-way blob
   at the size the papers measure puts it, and mixed two different questions into
   one ring: how much was decided, and how it was decided. */
/* WHERE THESE RENDERERS WRITE.
   The app hosts each of them in a modal; the landing page hosts them in its own
   full-height sections, and shows the Overview and the Compare frame at the SAME
   TIME in two different places. So the host cannot be a single global, and it
   cannot be a plain argument either — every control inside the rendered HTML
   re-renders by calling the function again with no arguments. It is therefore
   remembered: pass it once when you mount, and every re-render finds its way
   home. `tabs` is nullable because the landing page splits the two tabs across
   two sections and has no tab strip at all. */
const UCA={by:'papers', tab:'overview', host:'ucaBody', tabs:'ucaTabs', lean:false, scope:null};
function openUca(){ document.getElementById('ucaModal').classList.add('on'); renderUca() }
/* Counted per use case: a paper in two use cases counts twice, because they are
   two pools and two separate decisions (identity is UNIQUE(use_case_id,
   fingerprint), never global).

   `scope` picks which use cases the first disc totals. The app passes nothing and
   gets all nine, which is what its own workspace holds. The landing page passes
   SIX, because the three it leaves out are this project scoring itself and two
   archived stubs — see the note on SIX in data.js. */
function ucaData(scope){
  const us = scope? scope.map(id=>UCS.find(u=>u.id===id)).filter(Boolean) : UCS;
  const w={nm:'Starting data', n:us.length,
    papers:us.reduce((a,u)=>a+u.papers,0),
    pos:us.reduce((a,u)=>a+u.pos,0), neg:us.reduce((a,u)=>a+u.neg,0),
    lab:us.reduce((a,u)=>a+u.lab,0)};
  /* the library's negatives are real: every record in a screening project got a
     decision, so screened - included IS the number turned down. */
  const l={nm:'Benchset v1', n:TEMPLATES.length,
    papers:TEMPLATES.reduce((a,t)=>a+t[3],0), pos:TEMPLATES.reduce((a,t)=>a+t[4],0)};
  l.neg=l.papers-l.pos; l.lab=l.papers;
  return [w,l,BENCH2];
}
/* `lean` is the LANDING PRESENTATION, one flag rather than six.
   The app hosts these in a modal, where prose has room and a table is the precise
   read. The landing page gives each of them a whole screen and needs the chart to
   be the only thing on it — so lean means: fill the frame, big labels, no table,
   no explanatory block, no controls the section does not need. It changes nothing
   that is COMPUTED, which is the line that matters: the same numbers, presented
   for a different distance. */
function renderUca(host,tabs,opt){
  if(host!==undefined){ UCA.host=host; UCA.tabs=tabs||null }
  if(opt) Object.assign(UCA,opt);
  const lean=!!UCA.lean;
  const tb=UCA.tabs && document.getElementById(UCA.tabs);
  if(tb) tb.innerHTML=
    [['overview','Overview','pie'],['compare','Compare','compare']].map(([k,n,ic])=>
      `<button class="${UCA.tab===k?'on':''}" onclick="UCA.tab='${k}';renderUca()">${icon(ic)}${n}</button>`).join('');
  if(UCA.tab==='compare') return renderCompare(UCA.host);
  /* lean sizes by papers and never offers the toggle: the use-case count moves
     into the label, so one figure carries both numbers instead of hiding one
     behind a control nobody on a landing page will press. */
  const [w,l,b]=ucaData(UCA.scope), by=lean?'papers':UCA.by, cols=[w,l,b];
  const RMAX=lean?190:98, GAP=lean?86:52, PAD=lean?46:30, TOPLBL=lean?92:52;
  const val=(d,mode)=>mode==='papers'? d.papers : d.n;
  const maxOf=mode=>Math.max(...cols.map(d=>val(d,mode)));
  /* A FLOOR, not a fudge. At a 113x range the smallest disc comes out at 11 user
     units and its two pie slices are indistinguishable, so the floor is raised in
     lean — the reader can still see the ratio inside it. The floor is the ONLY
     place the drawing departs from proportional area, it only ever affects the
     smallest disc, and the label above every disc carries the real number. */
  const radAt=(d,mode)=>Math.max(lean?22:8,Math.sqrt(val(d,mode)/maxOf(mode))*RMAX);
  /* THE CENTRES ARE EQUIDISTANT, and the group is centred in the frame.
     They used to be packed — each centre placed one radius past the last disc's
     edge — which put a 22-unit disc hard against a 190-unit one and left the whole
     group hanging off the left of the frame. Even spacing at the widest disc's
     pitch means the eye reads the SIZES against a constant rhythm instead of
     against a varying gap, and the group sits on the frame's own centre line.
     Nothing about the areas changes; only where the discs are put. */
  const rMaxEach=cols.map(d=>Math.max(radAt(d,'papers'),radAt(d,'cases')));
  const rBig=Math.max(...rMaxEach);
  const PITCH=2*rBig+GAP;
  const W=Math.max(PAD*2+PITCH*cols.length-GAP, 620);
  const span=PITCH*(cols.length-1);
  const x0=(W-span)/2;
  const cxs=cols.map((_,i)=>x0+i*PITCH);
  /* SYMMETRIC, so the group of discs sits on the frame's own middle. It was
     +14 above and +46 below, which is a 32-unit list to the top that the eye reads
     as "not quite centred" without being able to say why. */
  const PADV=lean?30:18;
  const CY=TOPLBL+PADV+rBig, H=CY+rBig+PADV;
  const pie=(d,r,cx,cy)=>{
    const parts=[[d.pos,'hsl(145,38%,23%)',.92],[d.neg,'hsl(26,72%,45%)',.80]];
    const tot=d.pos+d.neg||1; let a0=-Math.PI/2, out='';
    parts.forEach(([v,col,op])=>{
      if(!v) return;
      const a1=a0+2*Math.PI*v/tot;
      /* a full-circle slice cannot be an arc — start and end coincide and the
         path collapses to nothing */
      out += (v===tot)
        ? `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}" fill-opacity="${op}"/>`
        : `<path d="M${cx} ${cy} L${(cx+r*Math.cos(a0)).toFixed(1)} ${(cy+r*Math.sin(a0)).toFixed(1)}
             A${r} ${r} 0 ${a1-a0>Math.PI?1:0} 1 ${(cx+r*Math.cos(a1)).toFixed(1)} ${(cy+r*Math.sin(a1)).toFixed(1)} Z"
             fill="${col}" fill-opacity="${op}"/>`;
      a0=a1;
    });
    return out;
  };
  const circles=cols.map((d,i)=>{
    const r=radAt(d,by), x=cxs[i], rate=(d.pos/(d.pos+d.neg)*100);
    return `<g${lean?' class="ucaBig"':''}>
      <text class="uca-nm" x="${x}" y="${lean?26:18}">${d.nm}</text>
      <text class="uca-v" x="${x}" y="${lean?54:34}">${d.papers.toLocaleString()} papers</text>
      <text class="uca-r" x="${x}" y="${lean?76:47}">${d.n} use case${d.n===1?'':'s'} · ${
        rate.toFixed(1)}% interesting</text>
      ${pie(d,r,x,CY)}
      <circle cx="${x}" cy="${CY}" r="${r.toFixed(1)}" fill="none" stroke="hsl(150,35%,10%)" stroke-opacity=".16"/>
    </g>`;
  }).join('');
  const pct=(v,d)=>d? (v/d*100).toFixed(1)+'%' : '—';
  const stage=`<div class="ucaStage"><svg viewBox="0 0 ${W} ${H}" role="img"
      preserveAspectRatio="xMidYMid meet"
      aria-label="proportional area comparison">${circles}</svg></div>`;
  const key=`<div class="ucaKey">
      <span><i class="lg" style="background:var(--yes)"></i>Interesting</span>
      <span><i class="lg" style="background:var(--no);opacity:.8"></i>Not interesting</span>
      <span class="sp" style="flex:1"></span>
      <span style="color:var(--ink3)">area, not radius</span>
    </div>`;
  if(lean){ document.getElementById(UCA.host).innerHTML=stage+key; return }
  document.getElementById(UCA.host).innerHTML=`
    <div class="ucaTop">
      <span class="lbl">Size the circles by</span>
      <span class="segs">${['papers','cases'].map(k=>
        `<button class="${by===k?'on':''}" onclick="UCA.by='${k}';renderUca()">${
          k==='papers'?'Papers':'Use cases'}</button>`).join('')}</span>
      <span class="sp" style="flex:1"></span>
      <span style="font-size:11.5px;color:var(--ink3)">area is proportional to the value, not radius</span>
    </div>
    ${stage}${key}
    <table class="ucaT"><thead><tr><th>Collection</th><th>Use cases</th><th>Papers</th>
      <th>Decided</th><th>Relevant</th><th>Rate</th></tr></thead><tbody>
      ${cols.map(d=>`<tr><td>${d.nm}</td><td>${d.n}</td><td>${d.papers.toLocaleString()}</td>
        <td>${d.lab.toLocaleString()}</td><td><b>${d.pos.toLocaleString()}</b></td>
        <td>${pct(d.pos,d.pos+d.neg)}</td></tr>`).join('')}
    </tbody></table>
    <div class="ucaNote">
      <b>Area, not radius.</b> A collection holding ${(b.papers/w.papers).toFixed(0)}× the papers is
      drawn ${Math.sqrt(b.papers/w.papers).toFixed(1)}× as wide. Setting the radius to the value
      directly would make it ${(b.papers/w.papers).toFixed(0)}× as wide and therefore
      ${(b.papers/w.papers).toFixed(0)}× too big by area — the standard way a bubble chart lies.<br>
      <b>The centres are fixed.</b> Toggling the measure only grows and shrinks the discs; it never
      moves them, so what you are comparing is size rather than travel.<br>
      <b>Counted per use case.</b> A paper in two of your use cases counts twice — two pools, two
      separate decisions, which is also how the database stores it.<br>
      <b>Why the rates matter more than the sizes.</b> ${pct(w.pos,w.pos+w.neg)} of what you have
      decided was relevant, against ${pct(l.pos,l.lab)} of the library and
      ${pct(b.pos,b.lab)} of Benchset v2 (${b.themes}). Your pools are searched, not screened,
      so they start far richer — a model that looks strong on yours has not yet been asked the
      hard question.</div>`;
}

/* ── COMPARE ──────────────────────────────────────────────────────────────
   "Which of these use cases are about the same thing?" answered geometrically.

   WHAT IS REAL HERE, precisely, because the answer differs per collection:
     · Every collection's POSITION is real. Its own description text is pushed
       through PROJ.embed — same vocabulary, same idf, same two components as the
       papers — so collections sit near each other when their words do. The
       similarity matrix underneath is the cosine of those same vectors.
     · "Low-carbon cement binders" has REAL PAPERS: the 75 fixture papers at
       their own projected coordinates, with the labels actually given to them.
     · Every OTHER collection has no papers in this fixture. Its cloud is drawn
       around its real centre with deterministic jitter, at its real positive
       rate. The cloud shows the interaction and the shape; it is not that
       collection's corpus, and the panel says so on screen rather than here.
   Getting this boundary wrong would be the worst version of the bug this file
   already fixed once: a picture that looks like data and is not. */
const CMP={sel:['uc1','tpl0','tpl1'], enc:'colour', host:'ucaBody', lean:false};
/* Six, because the landing page shows six use cases at once. Cross is last on
   purpose: it is the only one drawn as a stroke rather than a fill, so it reads
   as a marker rather than a mass and is the weakest of the six. */
const SHAPES=['circle','triangle','square','diamond','hexagon','cross'];
function cmpAll(){
  const out=[];
  UCS.forEach((u,i)=>out.push({key:'uc'+u.id,mine:true,nm:u.nm,txt:u.nm+' '+u.sub+' '+u.ob,
    papers:u.papers,pos:u.pos,neg:u.neg,live:u.id===1}));
  TEMPLATES.forEach((t,i)=>out.push({key:'tpl'+i,mine:false,nm:t[0],txt:t[0]+' '+t[2]+' '+t[5],
    papers:t[3],pos:t[4],neg:t[3]-t[4]}));
  return out;
}
function cmpToggle(k){
  const i=CMP.sel.indexOf(k);
  if(i>=0){ if(CMP.sel.length>1) CMP.sel.splice(i,1) }
  else if(CMP.sel.length<SHAPES.length) CMP.sel.push(k);
  renderUca();
}
function shapeAt(kind,x,y,r,fill,op){
  /* COERCED, because every other branch interpolates its coordinates into a
     template and a pre-formatted "123.4" works there by accident. The hexagon
     does arithmetic, so a string x turned `x + r*cos` into concatenation and the
     shape vanished. A helper that is number-safe in four of six branches is a
     trap for the fifth. */
  x=+x; y=+y; r=+r;
  const f=`fill="${fill}" fill-opacity="${op}"`;
  if(kind==='circle')   return `<circle cx="${x}" cy="${y}" r="${r}" ${f}/>`;
  if(kind==='square')   return `<rect x="${x-r}" y="${y-r}" width="${2*r}" height="${2*r}" ${f}/>`;
  if(kind==='triangle') return `<path d="M${x} ${y-r*1.15}L${x+r} ${y+r*.8}L${x-r} ${y+r*.8}Z" ${f}/>`;
  if(kind==='diamond')  return `<path d="M${x} ${y-r*1.25}L${x+r*1.25} ${y}L${x} ${y+r*1.25}L${x-r*1.25} ${y}Z" ${f}/>`;
  if(kind==='hexagon'){ const pts=[0,1,2,3,4,5].map(k=>{ const a=Math.PI/6+k*Math.PI/3;
      return `${(x+r*1.1*Math.cos(a)).toFixed(1)} ${(y+r*1.1*Math.sin(a)).toFixed(1)}` }).join('L');
    return `<path d="M${pts}Z" ${f}/>` }
  return `<path d="M${x-r} ${y-r}L${x+r} ${y+r}M${x+r} ${y-r}L${x-r} ${y+r}" fill="none"
    stroke="${fill}" stroke-opacity="${op}" stroke-width="1.9"/>`;
}
/* NO GREEN AND NO RED IN THIS PALETTE. Those two are the LABEL pair, and the EDA
   plot switches between colour-as-use-case and colour-as-label — so a use case
   drawn in green read as "interesting" in one mode and as a use case in the other.
   Six hues that are none of the reserved two, ordered for maximum adjacent
   contrast rather than by hue wheel: teal, indigo, plum, ochre, slate, cyan. */
/* NO GREEN AND NO RED IN THIS PALETTE. Those two are the LABEL pair, and the EDA
   plot switches between colour-as-use-case and colour-as-label — so a use case
   drawn in green read as "interesting" in one mode and as a use case in the other.

   ORDERED FOR THE PAIR THAT OVERLAPS. Now that positions mean similarity, the two
   most alike use cases sit on top of each other, and they are the two that most
   need telling apart. With `SIX` that pair is index 0 and index 4, so ochre sits at
   4 against blue at 0 — the widest separation in the set, given to the only place
   it matters. The rest are spaced round the remaining arc. */
const CMPCOL=['hsl(214,56%,40%)','hsl(286,38%,46%)','hsl(332,48%,42%)',
  'hsl(184,52%,26%)','hsl(36,68%,38%)','hsl(250,16%,44%)'];
function renderCompare(host,opt){
  if(host!==undefined) CMP.host=host;
  if(opt) Object.assign(CMP,opt);
  const lean=!!CMP.lean;
  const all=cmpAll(), sel=CMP.sel.map(k=>all.find(a=>a.key===k)).filter(Boolean);
  const W=700,H=380, pad=lean?54:26;
  const lay=cmpLayout(sel);
  /* DESCPROJ, not PROJ — the description space, not the cement papers' vocabulary.
     See the note above DESCPROJ in stats.js for why: in PROJ five of these six had
     two to four of their words in the space and every cosine floored at zero. */
  const DP=DESCPROJ();
  sel.forEach((c,i)=>{ c.at=lay[i]; const r=DP.raw(c.txt); c.vec=r.v; c.hit=r.hit; c.tot=r.tot });
  const pts=[];
  sel.forEach((c,ci)=>{
    if(c.live){
      /* the real cloud, translated to the collection's laid-out position and
         scaled down to fit beside the others — its INTERNAL shape is untouched,
         which is the part that is real */
      const mx=PAPERS.reduce((a,p)=>a+XY[p.id][0],0)/PAPERS.length;
      const my=PAPERS.reduce((a,p)=>a+XY[p.id][1],0)/PAPERS.length;
      PAPERS.forEach(p=>pts.push({ci,
        x:Math.min(.97,Math.max(.03,c.at[0]+(XY[p.id][0]-mx)*0.42)),
        y:Math.min(.95,Math.max(.05,c.at[1]+(XY[p.id][1]-my)*0.42)),
        pos:p.g==='pos',real:true,t:p.t}));
      return;
    }
    /* MORE POINTS ON THE LANDING PAGE. In the modal a cloud of 24-70 sits beside a
       picker and a similarity table and only has to read as "a group"; full-screen it
       is the whole figure, and 40 marks in a 700x380 frame looked like a sparse
       scatter rather than a corpus. Nothing about the SHAPE changes — same centre,
       same relevant rate, same deterministic jitter — only how densely it is sampled,
       and the jitter hash is per (collection, index) so the extra points extend the
       cloud instead of redrawing it. */
    const n = lean
      ? Math.min(190,Math.max(90,Math.round(c.papers/300)+90))
      : Math.min(70, Math.max(24,Math.round(c.papers/900)+24));
    const rate=c.pos/(c.pos+c.neg);
    for(let i=0;i<n;i++){
      /* deterministic jitter — a fixed hash of (collection, index), never random,
         so the cloud is the same picture on every open */
      const h=(ci*7919+i*104729)%100000;
      /* A FILLED DISC, NOT AN ANNULUS. The radius used to start at 0.028, which left a
         hole in the middle of every cloud — invisible while a numbered marker sat in it
         and unmistakable the moment label mode took the markers away, because six
         doughnuts is not what a corpus looks like. `u^0.8` fills the disc with the
         density falling off toward the rim: `u^0.5` would be perfectly uniform, below
         that the rim gets denser, above it the centre does. */
      const u=(((h>>>7)%100)+0.5)/100;
      const a=(h%628)/100, rr=0.088*Math.pow(u,0.8);
      pts.push({ci,x:Math.min(.97,Math.max(.03,c.at[0]+Math.cos(a)*rr)),
        y:Math.min(.95,Math.max(.05,c.at[1]+Math.sin(a)*rr*1.15)),
        pos:((h>>>3)%1000)/1000 < rate, real:false, t:c.nm});
    }
  });
  const byColour=CMP.enc==='colour';
  const dots=pts.map(p=>{
    /* ALWAYS A CIRCLE ON THE LANDING PAGE. Shape encoded the use case so that colour
       could encode the label — but the clusters are spatially separate and each one
       carries its own name on its centroid, so the shape was a third encoding of
       something already said twice, and five marker shapes at 60% opacity are harder
       to read as a cloud than one. The app's modal keeps them: there the six can be
       toggled and re-selected, so a stable per-use-case mark still earns its place. */
    const kind = (lean || byColour)? 'circle' : SHAPES[p.ci];
    /* THE LABEL PAIR, not a green/grey. When shape carries the use case, colour is
       free to carry the label, and the label colours are the same two the whole
       page uses — the labelling bins, the scatter's mistakes, the Sankey. Grey for
       "not interesting" made it read as "no data", which is a different fact. */
    const col  = byColour? CMPCOL[p.ci] : (p.pos?'hsl(145,38%,23%)':'hsl(26,72%,45%)');
    /* still transparent, so overlapping clusters read as overlapping rather than as
       whichever one happened to be drawn last — but 40%/28% over a paper ground made
       an individual mark hard to FIND, which is the other half of the job. Up to
       62%/48%, which still stacks visibly where two clusters cross. */
    const op   = lean ? (p.pos?.62:.48)
               : byColour? (p.pos?.92:.3) : (p.pos?.9:.62);
    /* bigger at full screen for the same reason the labels are */
    const r    = lean ? (p.pos? 5.6 : 4.4) : (p.pos? 4.2 : 3);
    return shapeAt(kind, (pad+p.x*(W-2*pad)).toFixed(1), (pad+p.y*(H-2*pad)).toFixed(1), r, col, op);
  }).join('');
  /* NUMBERED MARKERS, NOT NAMES. Six names on one 700x380 plot collide however
     hard you nudge them — the previous version stepped each label up to eight
     times and still stacked three of them, because MDS legitimately puts
     unrelated use cases at similar heights. A number cannot collide, it is the
     SAME number as the similarity table's row below, and the name it stands for
     is in the key under the plot. Three ways to read one thing beats one
     unreadable label. */
  /* ── THE CENTROID MARK, AND THE LABEL THAT HAS TO GET OUT OF ITS WAY ────────
     Now that the positions mean something — two similar use cases genuinely overlap
     — the labels cannot sit on their own centroids any more. So the MARK stays exactly
     where the maths put it and the LABEL moves: eight candidate bearings at two radii,
     scored against every label already placed and every other centroid, first clear
     slot wins, and a leader line runs back to the mark whenever it moved far enough to
     need one. A label is free to be anywhere; a centroid is not. */
  const lines=nm=>{ const w=nm.split(/\s+/), out=[]; let cur='';
    w.forEach(x=>{ if((cur+' '+x).trim().length>19){ out.push(cur); cur=x }
      else cur=(cur+' '+x).trim() });
    if(cur) out.push(cur);
    /* three lines, and an ellipsis if that was not all of it — a name that just
       stops mid-phrase reads as the name, which is worse than admitting the trim */
    if(out.length>3){ const k=out.slice(0,3); k[2]=k[2].replace(/\s+\S*$/,'')+'…'; return k }
    return out };
  const CW=8.2, LH=18;                       /* per-character width, line height */
  const pxOf=c=>pad+c.at[0]*(W-2*pad), pyOf=c=>pad+c.at[1]*(H-2*pad);
  const marks=sel.map((c,ci)=>{
    const px=pxOf(c), py=pyOf(c);
    return `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="${lean?7:11}"
      fill="var(--paper)" fill-opacity=".95" stroke="${CMPCOL[ci]}" stroke-width="2.6"/>`;
  }).join('');
  let centres='';
  if(!lean){
    centres=marks+sel.map((c,ci)=>`<text x="${pxOf(c).toFixed(1)}" y="${(pyOf(c)+4).toFixed(1)}"
      text-anchor="middle" class="cmpNum" fill="${CMPCOL[ci]}">${ci+1}</text>`).join('');
  } else if(!byColour){
    /* LABEL MODE NAMES NOTHING. The names are per use case and so are the centroid
       rings — both of them re-encode the very thing colour has just stopped encoding,
       so leaving them on gives the plot two contradictory keys at once (a green dot
       inside a purple ring). Colour means the label here, the key below says which is
       which, and the clusters are still in the same places they were a click ago. */
    centres='';
  } else {
    const placed=[];                          /* boxes already taken */
    const others=sel.map(c=>[pxOf(c),pyOf(c)]);
    const BEAR=[[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1]];
    const labs=sel.map((c,ci)=>{
      const ln=lines(c.nm), lw=Math.max(...ln.map(s=>s.length))*CW, lh=ln.length*LH;
      const px=pxOf(c), py=pyOf(c);
      let best=null, bestScore=-Infinity;
      [46,78,112].forEach(rad=>BEAR.forEach(([bx,by])=>{
        const cx=px+bx*rad*0.92, cy=py+by*rad*0.62;
        const box={x:cx-lw/2, y:cy-lh/2, w:lw, h:lh};
        if(box.x<6||box.x+box.w>W-6||box.y<14||box.y+box.h>H-6) return;
        /* overlap with a placed label is disqualifying; distance from the nearest
           centroid and the nearest label is the tie-break, so a label drifts into
           empty canvas rather than onto another cluster */
        const hit=placed.some(q=>box.x<q.x+q.w+7&&q.x<box.x+box.w+7&&
                                 box.y<q.y+q.h+5&&q.y<box.y+box.h+5);
        if(hit) return;
        let near=Infinity;
        others.forEach(([ox,oy])=>{ near=Math.min(near,Math.hypot(cx-ox,cy-oy)) });
        const score=near-rad*0.55;
        if(score>bestScore){ bestScore=score; best={cx,cy,box} }
      }));
      if(!best){ const cy=py-lh/2-16;
        best={cx:px, cy, box:{x:px-lw/2,y:cy-lh/2,w:lw,h:lh}} }
      placed.push(best.box);
      const y0=best.cy-lh/2+LH*0.72;
      /* the leader stops short of both the mark and the text so it reads as a tie,
         not as a stroke through either */
      const dx=best.cx-px, dy=best.cy-py, d=Math.hypot(dx,dy)||1;
      const leader = d>30
        ? `<line x1="${(px+dx/d*9).toFixed(1)}" y1="${(py+dy/d*9).toFixed(1)}"
             x2="${(best.cx-dx/d*(lh/2+3)).toFixed(1)}" y2="${(best.cy-dy/d*(lh/2+3)).toFixed(1)}"
             stroke="${CMPCOL[ci]}" stroke-opacity=".45" stroke-width="1.3"/>` : '';
      return leader+ln.map((s,k)=>`<text x="${best.cx.toFixed(1)}" y="${(y0+k*LH).toFixed(1)}"
        text-anchor="middle" class="cmpName" fill="${CMPCOL[ci]}">${s}</text>`).join('');
    }).join('');
    centres=labs+marks;                        /* marks on top of the leaders */
  }
  /* similarity: cosine between the same description vectors, in the plane */
  /* cosine of the description vectors themselves — never of the screen positions.
     Measuring similarity off the picture would make the table a restatement of the
     layout instead of the thing the layout is drawn FROM. */
  const cos=(a,b)=>{ let d=0,na=0,nb=0;
    for(let j=0;j<a.length;j++){ d+=a[j]*b[j]; na+=a[j]*a[j]; nb+=b[j]*b[j] }
    return d/(Math.sqrt(na*nb)||1) };
  const simCell=(a,b)=>{
    if(a===b) return {v:1,cls:'self'};
    const v=Math.max(0,cos(sel[a].vec,sel[b].vec));
    return {v,cls:v>0.30?'hi':v>0.12?'mid':''};
  };
  /* LEAN: no picker, no similarity table, no explanatory block. All six are on and
     none can be turned off — on a landing page a control that can empty the figure
     is a control that will. The toggle stays, because it is the point of the
     section: the SAME six points, coloured by which question they belong to, then
     by the answer that was given. */
  const encBar=`<div class="cmpBar">
      <span>Colour by</span>
      <span class="segs">${[['colour','Use case'],['shape','Label']].map(([k,n])=>
        `<button class="${CMP.enc===k?'on':''}" onclick="CMP.enc='${k}';renderCompare()">${n}</button>`).join('')}</span>
      <span style="color:var(--ink3)">${byColour
        ? 'one colour per use case'+(lean?' · named on the plot':'')
        : lean ? ''            /* the key below says it, in swatches */
               : 'green is interesting, orange is not · shape is the use case'}</span>
      <span class="sp" style="flex:1"></span>
      ${lean?'':`<span style="color:var(--ink3)">${CMP.sel.length} of ${SHAPES.length} selected</span>`}
    </div>`;
  /* IN LEAN, THE KEY GOES INSIDE THE PLOT. Under a full-screen scatter a key is a
     second reading two hundred pixels from the thing it explains, and the eye does not
     make the trip — the same argument that put the 2x2 inside the operating-point
     frame. Bottom-left, over the emptiest corner of the six-cluster layout, and sized
     to be read at the plot's distance rather than a modal's.
     Only in label mode: colour-by-use-case names every cluster on the plot, so a key of
     six identical circles beside six names would just be the names again. */
  const leanKey = (lean && !byColour) ? `<div class="cmpInKey">
      <span><i class="lg" style="background:hsl(145,38%,23%)"></i>Interesting</span>
      <span><i class="lg" style="background:hsl(26,72%,45%)"></i>Not interesting</span>
    </div>` : '';
  const stage=`<div class="cmpStage${lean?' lean':''}"><svg viewBox="0 0 ${W} ${H}" role="img"
          preserveAspectRatio="xMidYMid meet"
          aria-label="use cases in the same projection">${dots}${centres}</svg>${leanKey}</div>`;
  const key=`<div class="cmpKey">${sel.map((c,ci)=>`<span class="kk">
          <svg viewBox="0 0 12 12">${shapeAt(byColour?'circle':SHAPES[ci],6,6,4.4,
            byColour?CMPCOL[ci]:'hsl(150,25%,30%)',.9)}</svg>${
            c.nm.length>26?c.nm.slice(0,25)+'…':c.nm}</span>`).join('')}</div>`;
  /* nothing under the plot in lean: the key, when there is one, is inside the stage */
  if(lean){ document.getElementById(CMP.host).innerHTML=encBar+stage; return }
  const list=(hd,items)=>`<div class="hd">${hd}</div>`+items.map(a=>{
    const on=CMP.sel.includes(a.key), full=!on&&CMP.sel.length>=SHAPES.length;
    return `<button class="cmpItem ${on?'on':''} ${full?'full':''}" onclick="cmpToggle('${a.key}')">
      <span class="mk">${icon(on?'circleCheck':'plus')}</span>
      <span><b>${a.nm}</b><span>${a.papers.toLocaleString()} papers · ${
        (a.pos/(a.pos+a.neg)*100).toFixed(1)}% relevant${a.live?' · real papers':''}</span></span></button>`
  }).join('');
  document.getElementById(CMP.host).innerHTML=`
    ${encBar}
    <div class="cmpGrid">
      <div class="cmpPick">${list('Yours',all.filter(a=>a.mine))}${list('Benchset v1',all.filter(a=>!a.mine))}</div>
      <div>${stage}
        <div class="cmpKey">${sel.map((c,ci)=>`<span class="kk">
          <b class="kn" style="color:${CMPCOL[ci]};border-color:${CMPCOL[ci]}">${ci+1}</b>
          <svg viewBox="0 0 12 12">${shapeAt(byColour?'circle':SHAPES[ci],6,6,4.4,
            byColour?CMPCOL[ci]:'hsl(150,25%,30%)',.9)}</svg>${
            c.nm.length>26?c.nm.slice(0,25)+'…':c.nm}</span>`).join('')}</div>
      </div>
    </div>
    <div class="cmpSim">
      <div class="pt" style="font:600 10px/1 var(--bd);letter-spacing:.13em;text-transform:uppercase;
        color:var(--ink3);margin-bottom:10px">How alike their descriptions are</div>
      <table class="simT"><thead><tr><th></th>${sel.map((c,i)=>
        `<th>${i+1}</th>`).join('')}</tr></thead><tbody>
        ${sel.map((c,a)=>`<tr><th class="rw">${a+1} · ${c.nm}<em>${c.hit} of ${c.tot} words
          in this space</em></th>${sel.map((_,b)=>{
          const {v,cls}=simCell(a,b);
          return `<td class="${cls}">${a===b?'—':v.toFixed(2)}</td>` }).join('')}</tr>`).join('')}
      </tbody></table>
    </div>
    <div class="ucaNote">
      <b>What is real on this plot.</b> Every collection's <i>position</i> is real — its own
      description is pushed through a tf-idf over all ${DP.N} use-case descriptions, live and
      library, so two sitting close together are close because their words are.
      <b>Low-carbon cement binders</b> also has real papers at their real coordinates with the
      labels you gave them.<br>
      <b>What is not.</b> No other collection has papers in this fixture, so its cloud is drawn
      around its real centre with fixed jitter at its real relevant rate. It shows the shape of the
      comparison, not that collection's corpus. In the app every point is a paper.<br>
      <b>The layout is drawn FROM the table.</b> Positions come from a two-component fit over just
      the collections you selected — classical MDS of those similarity numbers — so the picture
      changes when the selection does, because the question did. The numbers themselves are cosines
      between the descriptions, never distances measured off the picture.<br>
      <b>Read groupings, never distances.</b> Two components carry a fraction of what the full space
      holds; this is a similarity ordering, not a metric.<br>
      <b>What this space can and cannot see.</b> It is a bag of ${DP.dim} words that appear in at
      least two descriptions, so it measures shared VOCABULARY. Two descriptions that ask the same
      question in different words score low here and would not under a real encoder — in the app the
      vectors come from <i>bge-small-en-v1.5</i>, where every text embeds fully and "coverage" is not
      a concept. The <i>words in this space</i> figure under each row is how much of that description
      the measurement could actually use.</div>`;
}

/* ═══════════════════════════════════════════════════════════════════════
   THE POOL MAP + THE LANGUAGE OF THE BOUNDARY
   Two honest things, and the honesty is the point of both.

   The MAP is a 2-D flattening of the pool, and as of v6 it is a REAL projection
   (see PROJ below) rather than a hash of the title. The caption still has to say
   the positions are approximate, because a projection of a 384-dimension space
   onto a page is lossy however it is computed and a reader will otherwise measure
   distances off it. The CURRENT CARD IS NEVER MARKED: that would put a score in
   front of an unlabelled paper and break the shadow rule.

   The LANGUAGE is computed the same way the filter suggestions are, in both
   directions: words that appear in what you kept and not in what you rejected,
   and the reverse. It is the analyst's own boundary read back to them in their
   own vocabulary, which is the thing a bar chart of counts cannot do.
   ═══════════════════════════════════════════════════════════════════════ */
const MAPW=278, MAPH=150;

/* `tune` switches the map from "what you have decided" to "what a model would
   decide at this cut-off": every paper gets a predicted verdict, and the ones you
   have judged yourself keep a ring so your own work stays distinguishable from
   the model's guess. The contours are dropped in that mode — the operating point
   IS the boundary there, so drawing three of them alongside it would be showing
   the same choice twice and disagreeing with itself. */
function poolMap(big,tune){
  const W=big?720:MAPW, H=big?430:MAPH, p=nPos();
  const r=v=>v.toFixed(1);
  const dots=PAPERS.map(pp=>{
    const [x,y]=XY[pp.id], lab=S.lab[pp.id];
    const cx=x*W, cy=y*H;
    let cls = lab==='pos'?'dPos':lab==='neg'?'dNeg':'dNone', rad, ring='';
    if(tune){
      const inCut=tune.cut.has(pp.id);
      if(tune.err){
        /* FOUR OUTCOMES, NOT TWO. `tune.cut` alone says what the model predicts;
           it cannot show what that costs. Against the gold answer the same
           prediction splits into a correct keep, a correct pass, a paper you
           will read for nothing, and a paper you will never see. The last two
           are the whole subject of a precision/recall setting, so they get the two
           warning colours and the two correct outcomes stay quiet.
           FP AND FN GET DIFFERENT HUES — orange and red. They were orange filled
           and orange hollow, which reads as one error drawn two ways; they are two
           errors with completely different prices, and the section exists to trade
           one against the other. */
        cls = inCut ? (pp.g==='pos'?'dTP':'dFP') : (pp.g==='pos'?'dFN':'dTN');
        rad = cls==='dTN' ? 3 : cls==='dTP' ? 5 : 5.5;
      } else {
        cls = inCut? 'dYes':'dNo';
        rad = inCut? 5:3;
        if(lab) ring=` stroke="${lab==='pos'?'hsl(145,45%,15%)':'hsl(150,15%,25%)'}" stroke-width="1.6"`;
      }
    } else {
      rad = big? (lab==='pos'?5.5:lab==='neg'?4:2.6) : (lab==='pos'?3.6:lab==='neg'?2.6:1.5);
    }
    return `<circle class="${cls}" cx="${r(cx)}" cy="${r(cy)}" r="${rad}"${ring}
      ${big?`data-i="${pp.id}" onmouseenter="mapHover(${pp.id},${r(cx)},${r(cy)})" onmouseleave="mapHover(null)"`:''}/>`;
  }).join('');

  /* THREE CONTOURS, not one. The shipped operating metric is recall@k% of the
     reviewed pool rather than a probability threshold, so the useful picture is
     not "the boundary" but where the boundary SITS at different operating points:
     tight = high precision and misses more, loose = high recall and reads more.
     Same fitted model, three cut-offs — which is what a precision/recall trade-off
     actually looks like on a corpus. */
  let contours='';
  /* Guard on the REAL labelled positives, not on nPos(): nPos() includes the demo
     strip's ghost counters, so jumping to a later state made this branch run with
     an empty `kept` array and every ellipse coordinate came out NaN. A count that
     can be inflated by something other than the data it describes must never be
     the guard for a computation over that data. */
  const kept=Object.keys(S.lab).filter(i=>S.lab[i]==='pos').map(i=>XY[i]);
  if(!tune && kept.length>=3){
    const mx=kept.reduce((a,k)=>a+k[0],0)/kept.length*W, my=kept.reduce((a,k)=>a+k[1],0)/kept.length*H;
    const base=Math.max(big?70:26, (big?170:64)-kept.length*(big?3.4:1.4));
    const LV=[[0.62,'tight','.20'],[1.0,'balanced','.13'],[1.5,'loose','.07']];
    contours=LV.map(([k,lbl,op])=>`
      <ellipse cx="${r(mx)}" cy="${r(my)}" rx="${r(base*1.15*k)}" ry="${r(base*0.86*k)}"
        fill="hsl(145,38%,23%)" fill-opacity="${op}"/>
      <ellipse cx="${r(mx)}" cy="${r(my)}" rx="${r(base*1.15*k)}" ry="${r(base*0.86*k)}"
        fill="none" stroke="hsl(145,38%,23%)" stroke-opacity=".34" stroke-dasharray="4 3"/>
      ${big?`<text x="${r(mx+base*1.15*k-4)}" y="${r(my-base*0.86*k+13)}" class="mlbl">${lbl}</text>`:''}`).join('');
  }
  return `<svg viewBox="0 0 ${W} ${H}" class="${big?'mapBig':''}" role="img" aria-label="map of the pool">
    <style>.dPos{fill:hsl(145,38%,23%)}.dNeg{fill:hsl(150,10%,52%);opacity:.75}
      .dNone{fill:hsl(150,20%,42%);opacity:.2}
      .dYes{fill:hsl(145,38%,23%);opacity:.9}.dNo{fill:hsl(150,10%,52%);opacity:.28}
      .dTP{fill:hsl(145,38%,23%);opacity:.92}.dTN{fill:hsl(150,10%,52%);opacity:.22}
      .dFP{fill:hsl(26,72%,45%);opacity:.95}
      .dFN{fill:hsl(2,66%,46%);opacity:.95}
      ${big?'.dNone{opacity:.28}.dPos,.dNeg,.dNone{cursor:pointer}.dPos:hover,.dNeg:hover,.dNone:hover{stroke:hsl(38,55%,40%);stroke-width:2}':''}
      .mlbl{font:600 10px var(--mn);fill:hsl(145,38%,23%);fill-opacity:.75;text-anchor:end}</style>
    ${contours}${dots}</svg>`;
}
function mapHover(id,x,y){
  const t=document.getElementById('mapTip'); if(!t) return;
  if(id===null){ t.classList.remove('on'); return }
  const pp=PAPERS[id], lab=S.lab[id];
  /* citations/venue are fixture values; in the app they come off the paper row. */
  const cites=40+((id*37)%420);
  t.innerHTML=`<div class="mtT">${pp.t}</div>
    <div class="mtM">${pp.v} · ${pp.y} · ${cites} citations</div>
    <div class="mtL">doi.org/10.1016/… <span class="mtS">${
      lab==='pos'?'you kept this':lab==='neg'?'you turned this down':'not yet seen'}</span></div>`;
  t.style.left=Math.min(x+14,520)+'px'; t.style.top=Math.max(y-10,4)+'px';
  t.classList.add('on');
}
/* ═══════════════════════════════════════════════════════════════════════
   WHERE EVERY PAPER WENT — a Sankey, and two decisions inside it

   (1) THE SOURCES ARE ATTRIBUTED, NOT RAW. 4,921 records were retrieved and
   4,120 survive deduplication, but the per-source counts we hold are the
   ATTRIBUTED ones — each unique paper credited to a source — so they sum to
   4,120, not 4,921. Drawing them as a 4,921-wide first stage would require
   splitting the 801 duplicates between sources, which nobody recorded. So the
   diagram conserves flow from 4,120 and the 801 is stated ON the dedup node
   instead of drawn as a ribbon. A Sankey that does not balance is a Sankey that
   is guessing.

   (2) CORE RETURNED NOTHING AND THAT IS NOT ZERO. It carries no ribbon and is
   drawn as a dashed stub reading "no answer", because "we asked and it gave us
   nothing" and "we never found out" are different facts and only one of them is
   a number. Same rule as everywhere else in this file.
   ═══════════════════════════════════════════════════════════════════════ */
function openFlow(){ document.getElementById('flowModal').classList.add('on'); renderFlow('flowBody',{key:true}) }
/* same remembered-host trick as renderUca — see the note there. `key` is
   nullable: the landing page drops the legend strip because its labels are big
   enough to carry their own colour. */
const FLOWH={host:'flowBody', key:true, lean:false, scope:null};
/* `rej` is the label pair's orange, not a grey. Grey said "no data" where the
   fact is "a decision was taken and it was no" — the same distinction --no exists
   for in base.css. */
const FLOWCOL={src:'hsl(150,14%,58%)',keep:'hsl(145,38%,23%)',rej:'hsl(26,72%,45%)',
  filt:'#de9815',none:'hsl(150,18%,45%)',gap:'hsl(28,45%,52%)'};
function renderFlow(host,opt){
  if(host!==undefined) FLOWH.host=host;
  if(opt) Object.assign(FLOWH,opt);
  const lean=!!FLOWH.lean;
  /* SCOPE. The app's modal describes ONE use case's run — 4,120 unique of 4,921
     retrieved, which is what the search actually returned. The landing page
     describes the whole workspace, so it sums the six use cases and scales the
     rest of the diagram by the same ratio rather than inventing per-stage figures
     nobody recorded. The dedup rate, the no-abstract rate and the label counts are
     all carried across as PROPORTIONS of the one run that was measured; that is a
     stated extrapolation, not a second measurement, and it is why the six-use-case
     view rounds. */
  const sc=FLOWH.scope? sixUcs() : null;
  const UNIQ = sc? sc.reduce((a,u)=>a+u.papers,0) : 4120;
  const k=UNIQ/4120;
  const RAW=Math.round(4921*k), NOABS=Math.round(212*k), WITHABS=UNIQ-NOABS;
  const pos = sc? sc.reduce((a,u)=>a+u.pos,0) : nPos();
  const neg = sc? sc.reduce((a,u)=>a+u.neg,0) : nNeg();
  const filt = sc? Math.round(140*k) : (S.filtered||0);
  const unlab=Math.max(0,WITHABS-pos-neg-filt);
  const srcs=SRC.filter(x=>x[1]!==null).map(([nm,v])=>[nm,Math.round(v*k)]);
  const nulls=SRC.filter(x=>x[1]===null);
  /* LEAN IS A VIEWBOX MATCHED TO ITS FRAME, not a bigger font. `meet` scales the
     whole diagram — labels included — to fit, so a viewBox whose aspect is far from
     the frame's gets letterboxed and every label shrinks with it. Sharing a screen
     with the stack list gave this a 5:1 slot and needed 1560x400; on its own screen
     the slot is about 2.4:1, so 1500x620 fills it and the labels render at the size
     the stylesheet actually asks for. */
  const W=lean?1500:920, H=lean?620:430, PAD=lean?26:26, NW=lean?22:12, LBL=lean?230:118;
  const x0=LBL, x1=Math.round(W*0.40), x2=Math.round(W*0.60), x3=W-LBL-NW;
  const avail=H-PAD*2, GAP=7;
  /* THE BAR IS PROPORTIONAL; THE SLOT IT SITS IN IS AT LEAST ONE LABEL TALL.
     arXiv's 6 papers and OpenAlex's 2,841 cannot share a label pitch — at one
     scale for both, four source names printed on top of each other and the
     diagram was unreadable exactly where it was most precise. Growing the
     WHITESPACE around a small bar costs nothing true: the bar's height still
     means its count, and only the gap changes. Faking the bar's height would
     have been the other way out, and it would have been a lie. */
  const SLOT=lean?52:27;
  /* ── THE SCALE HAS TO ACCOUNT FOR THE FLOORS ──────────────────────────────
     This was `(avail - gaps) / UNIQ`, which is the scale that makes the BARS fit.
     But five small sources each get a 34-unit slot regardless of their bar, so the
     source column's real height is one big bar plus five floors — and it came out
     529 units tall inside a 360-unit canvas. arXiv and CORE were being drawn below
     the bottom edge of the viewBox: not clipped visibly, just absent, which is the
     worst way for a diagram to be wrong.

     Solved by iteration rather than algebra, because which items are AT the floor
     depends on the scale and the scale depends on which items are at the floor.
     Four passes converge; the loop is capped anyway. One scale still governs every
     column — that is what makes a ribbon's width mean the same thing throughout —
     so the binding column is whichever needs the smallest, and all four get it. */
  const fitScale=colsOf=>{
    let best=Infinity;
    colsOf.forEach(items=>{
      const live=items.filter(i=>i.v>0);
      if(!live.length) return;
      const gaps=GAP*(items.length-1);
      let s=(avail-gaps)/Math.max(1,live.reduce((a,i)=>a+i.v,0));
      for(let k=0;k<8;k++){
        const floored=items.filter(i=>i.v*s<SLOT).length;
        const rest=items.filter(i=>i.v*s>=SLOT);
        const room=avail-gaps-SLOT*floored;
        const sumRest=rest.reduce((a,i)=>a+i.v,0);
        if(!sumRest||room<=0) break;
        const ns=room/sumRest;
        if(Math.abs(ns-s)<1e-7) break;
        s=ns;
      }
      best=Math.min(best,s);
    });
    return best;
  };
  /* CORE rides along as a zero-value item so the layout RESERVES its slot. It used
     to be positioned at `lastSource.y + h + 30`, a magic offset that walked off the
     canvas the moment the column got taller than the space. A stub that is part of
     the column cannot be orphaned by it. */
  const srcItems=srcs.map(([nm,v])=>({nm,v,col:FLOWCOL.src}))
    .concat(nulls.map(([nm])=>({nm,v:0,col:FLOWCOL.src,stub:1})));
  const uniqItems=[{nm:'Unique',v:UNIQ,col:FLOWCOL.src}];
  const absItems=[{nm:'Has an abstract',v:WITHABS,col:FLOWCOL.src},
                  {nm:'Waiting on abstract',v:NOABS,col:FLOWCOL.gap}];
  const decItems=[{nm:'Interesting',v:pos,col:FLOWCOL.keep,key:1},
                  {nm:'Not interesting',v:neg,col:FLOWCOL.rej,key:1},
                  {nm:'Hidden by a filter',v:filt,col:FLOWCOL.filt,key:1},
                  {nm:'Not sorted yet',v:unlab,col:FLOWCOL.none,key:1}];
  const scale=fitScale([srcItems,uniqItems,absItems,decItems]);
  const lay=(items,x)=>{
    const hs=items.map(i=>i.v>0?Math.max(1.5,i.v*scale):0);
    const slots=hs.map(h=>Math.max(h,SLOT));
    const tot=slots.reduce((a,b)=>a+b,0)+GAP*(items.length-1);
    let y=PAD+Math.max(0,(avail-tot)/2);
    return items.map((i,k)=>{
      const o={...i,x,y:y+(slots[k]-hs[k])/2,h:hs[k]};
      y+=slots[k]+GAP; return o });
  };
  const c0=lay(srcItems,x0);
  const c1=lay(uniqItems,x1);
  const c2=lay(absItems,x2);
  /* `key:true` makes the node's own label take the node's colour, so the terminal
     column IS the legend. A five-item key strip above a diagram asks the reader to
     hold five colour-to-word pairs in their head and then look down; a coloured
     label asks nothing. The strip stays available for the app's modal, where the
     panel is narrow and the labels are small. */
  const c3=lay(decItems,x3);
  const ribbon=(a,ao,b,bo,h,col,op)=>{
    const x1_=a.x+NW, x2_=b.x, mid=(x1_+x2_)/2;
    return `<path d="M${x1_} ${ao} C${mid} ${ao}, ${mid} ${bo}, ${x2_} ${bo}
      L${x2_} ${bo+h} C${mid} ${bo+h}, ${mid} ${ao+h}, ${x1_} ${ao+h} Z"
      fill="${col}" fill-opacity="${op}"/>`;
  };
  let links='', o1=c1[0].y;
  c0.forEach(n=>{ if(n.v<=0) return; links+=ribbon(n,n.y,c1[0],o1,n.h,FLOWCOL.src,.22); o1+=n.h });
  let o1b=c1[0].y;
  c2.forEach(n=>{ links+=ribbon(c1[0],o1b,n,n.y,n.h,n.col,.22); o1b+=n.h });
  let o2=c2[0].y;
  c3.forEach(n=>{ if(n.v<=0) return;
    links+=ribbon(c2[0],o2,n,n.y,n.h,n.col,.26); o2+=n.h });
  const node=(n,side)=>`<g>
    <rect x="${n.x}" y="${n.y.toFixed(1)}" width="${NW}" height="${n.h.toFixed(1)}"
      fill="${n.col}" fill-opacity=".85"/>
    <text class="fw-nm${n.key?' fw-keyed':''}" x="${side==='l'?n.x-10:n.x+NW+10}"
      y="${(n.y+n.h/2-(lean?4:1)).toFixed(1)}" ${n.key?`fill="${n.col}"`:''}
      text-anchor="${side==='l'?'end':'start'}">${n.nm}</text>
    <text class="fw-v" x="${side==='l'?n.x-10:n.x+NW+10}" y="${(n.y+n.h/2+(lean?24:12)).toFixed(1)}"
      text-anchor="${side==='l'?'end':'start'}">${n.v.toLocaleString()}</text></g>`;
  const nullStubs=c0.filter(n=>n.stub).map(n=>{
    const y=n.y+SLOT/2;
    return `<g><line x1="${x0}" y1="${y.toFixed(1)}" x2="${x0+NW}" y2="${y.toFixed(1)}"
      stroke="hsl(150,10%,58%)" stroke-dasharray="3 3"/>
      <text class="fw-nm" x="${x0-8}" y="${(y+4).toFixed(1)}" text-anchor="end">${n.nm}</text>
      <text class="fw-null" x="${x0+NW+8}" y="${(y+4).toFixed(1)}">no answer — not zero</text></g>` }).join('');
  document.getElementById(FLOWH.host).innerHTML=`
    ${FLOWH.key===false?'':`<div class="flowKey">
      <span><i class="lg" style="background:${FLOWCOL.keep}"></i>Interesting</span>
      <span><i class="lg" style="background:${FLOWCOL.rej}"></i>Not interesting</span>
      <span><i class="lg" style="background:${FLOWCOL.filt}"></i>Hidden by a filter — never deleted</span>
      <span><i class="lg" style="background:${FLOWCOL.gap}"></i>Waiting on an abstract</span>
      <span><i class="lg" style="background:${FLOWCOL.none}"></i>Not sorted yet</span>`}
    </div>
    <div class="flowStage${lean?' lean':''}"><svg viewBox="0 0 ${W} ${H+26}" role="img"
      preserveAspectRatio="xMidYMid meet"
      aria-label="flow from sources to decisions">
      <text class="fw-col" x="${x0-8}" y="14" text-anchor="end">SOURCE</text>
      <text class="fw-col" x="${x1}" y="14" text-anchor="middle">DEDUPLICATED</text>
      <text class="fw-col" x="${x2+NW+8}" y="14">READABLE?</text>
      <text class="fw-col" x="${x3+NW+8}" y="14">DECISION</text>
      ${links}
      ${c0.map(n=>n.v>0?node(n,'l'):'').join('')}
      <g><rect x="${c1[0].x}" y="${c1[0].y.toFixed(1)}" width="${NW}" height="${c1[0].h.toFixed(1)}"
        fill="${FLOWCOL.src}" fill-opacity=".85"/>
        <text class="fw-v" x="${c1[0].x+NW/2}" y="${(c1[0].y-8).toFixed(1)}" text-anchor="middle">${UNIQ.toLocaleString()} unique</text>
        <text class="fw-null" x="${c1[0].x+NW/2}" y="${(c1[0].y+c1[0].h+15).toFixed(1)}" text-anchor="middle">${(RAW-UNIQ).toLocaleString()} duplicate records merged</text></g>
      ${c2.map(n=>node(n,'r')).join('')}
      ${c3.map(n=>n.v>0?node(n,'r'):'').join('')}
      ${nullStubs}
    </svg></div>
    ${lean?`<div class="flowNote lean">${SRC.length} sources · ${RAW.toLocaleString()} records ·
      ${UNIQ.toLocaleString()} unique · ${(pos+neg).toLocaleString()} decided by hand
      <span>CORE returned no answer, which is not a zero. Stage rates are carried from the one
      measured run.</span></div>`:`<div class="flowNote">
      <b>The source counts are attributed, not raw.</b> ${RAW.toLocaleString()} records came back and
      ${UNIQ.toLocaleString()} survive deduplication, but what is stored per source is which source a
      unique paper is credited to — so they sum to ${UNIQ.toLocaleString()}. Splitting the
      ${(RAW-UNIQ).toLocaleString()} duplicates between sources would be inventing a number nobody
      recorded, so it is written on the node instead of drawn as a ribbon.<br>
      <b>CORE returned no answer, which is not a zero.</b> It gets a dashed stub and no width. "We
      asked and there was nothing" and "we never found out" are different facts.<br>
      <b>Nothing leaves this diagram.</b> A filter hides papers and never deletes them, so
      "hidden by a filter" is a band here rather than a gap in the total — every one of the
      ${UNIQ.toLocaleString()} is somewhere on the right-hand side.</div>`}`;
}

/* ═══════════════════════════════════════════════════════════════════════
   LANDING
   Three sections, and the only two things worth noting in code.

   (1) THE RADAR BLIPS ARE THE FIXTURE'S REAL PAPERS. Same PROJ coordinates as
   the pool map, re-expressed in polar: the projection plane is centred on the
   radar face, so the clusters a reader sees sweeping past are the same clusters
   they will see inside the app. A marketing page drawing decorative noise where
   the product draws data is a small lie that costs nothing to avoid.

   (2) THE SWEEP AND THE BLIPS SHARE ONE CLOCK. Each blip's animation-delay is
   its own bearing as a fraction of the 7s revolution, negative so the cycle is
   already under way on load. Nothing is scripted frame by frame; there is no
   timer running behind this page.
   ═══════════════════════════════════════════════════════════════════════ */
function radarSVG(){
  const R=250, C=300, SWEEP=7;
  const rings=[0.28,0.52,0.76,1].map(k=>
    `<circle class="rgrid" cx="${C}" cy="${C}" r="${(R*k).toFixed(1)}"/>`).join('');
  const spokes=[...Array(12)].map((_,i)=>{ const a=i*Math.PI/6;
    return `<line class="rspoke" x1="${C}" y1="${C}" x2="${(C+Math.cos(a)*R).toFixed(1)}"
      y2="${(C+Math.sin(a)*R).toFixed(1)}"/>` }).join('');
  /* the projection, centred and clamped inside the face */
  const blips=PAPERS.map(p=>{
    const [x,y]=XY[p.id];
    let dx=(x-0.5)*2.0, dy=(y-0.5)*2.2;
    const m=Math.hypot(dx,dy)||1, k=Math.min(1,0.93/m);
    dx*=k; dy*=k;
    let deg=Math.atan2(dy,dx)*180/Math.PI; if(deg<0) deg+=360;
    const hot=p.g==='pos';
    return `<circle class="rblip${hot?' hot':''}" cx="${(C+dx*R).toFixed(1)}"
      cy="${(C+dy*R).toFixed(1)}" r="${hot?4.2:2.6}"
      style="animation-delay:-${(deg/360*SWEEP).toFixed(2)}s"/>`;
  }).join('');
  return `<svg viewBox="0 0 600 600" role="img" aria-label="radar">
    <defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="hsl(145,38%,23%)" stop-opacity="0"/>
      <stop offset="1" stop-color="hsl(145,38%,23%)" stop-opacity=".20"/></linearGradient></defs>
    ${rings}${spokes}
    <g class="rsweepg"><path d="M${C} ${C} L${C+R} ${C} A${R} ${R} 0 0 0 ${(C+R*Math.cos(-Math.PI/4)).toFixed(1)} ${(C+R*Math.sin(-Math.PI/4)).toFixed(1)} Z"
      fill="url(#rg)"/>
      <line x1="${C}" y1="${C}" x2="${C+R}" y2="${C}" stroke="hsl(145,38%,23%)" stroke-opacity=".45" stroke-width="1.2"/></g>
    ${blips}
    <circle cx="${C}" cy="${C}" r="3" fill="hsl(145,38%,23%)" opacity=".55"/></svg>`;
}
/* the low-fi stack: five cards on one 6s clock, evenly phased, alternating side */
function lofiHTML(){
  /* THE CARD HAS TO READ AS A PAPER WITHOUT A CAPTION SAYING SO. Four grey bars
     could be anything — a to-do, a tweet, a form. A journal line, a two-line
     title, an author rule and an abstract block is the shape of the thing, and
     the shape is what makes the animation self-explanatory at a glance.
     The venues are the fixture's own; nothing here is a made-up journal. */
  const VEN=[['J. Clean. Prod.','2024'],['Cem. Concr. Res.','2023'],
             ['Constr. Build. Mater.','2024'],['Cem. Concr. Compos.','2022'],
             ['J. CO₂ Util.','2023']];
  /* three of the five go left. Most papers in a real pool are not relevant, and
     an illustration that keeps as many as it rejects is drawing the wrong job. */
  const left=[1,2,4];
  /* THE PILES HAVE COUNTS, and they are the fixture's real ones — the 76-paper
     deck's own split, not a pair of numbers chosen to look good. Three of five
     cards go left because the ratio in a real pool is nothing like even, and the
     counts under the bins say what that ratio actually was. */
  const nY=PAPERS.filter(p=>!p.nab&&p.g==='pos').length;
  const nN=PAPERS.filter(p=>!p.nab&&p.g==='neg').length;
  return `<div class="lofiBin no">${icon('x')}<span>Not interesting</span><b>${nN}</b></div>
    <div class="lofiBin yes">${icon('check')}<span>Interesting</span><b>${nY}</b></div>`+
    VEN.map(([v,y],i)=>`<div class="lcard ${left.includes(i)?'toNo':'toYes'}"
      style="animation-delay:-${(i*1.2).toFixed(1)}s;
      animation-name:${left.includes(i)?'lswipeL':'lswipeR'};z-index:${5-i}">
      <span class="lcHd">${v}<em>${y}</em></span>
      <i class="lcT"></i><i class="lcT sh"></i>
      <span class="lcAu"></span>
      <i></i><i></i><i class="sh"></i>
    </div>`).join('');
}
/* the funnel: a thousand technologies, the money, the decades, one plant.
   The field really is 1,000 squares (40 x 25). "Thousands of technologies" is a
   claim, and a caption that says thousands over a picture of twenty-four is the
   same defect as a header contradicting its own rows — cheap to avoid by just
   drawing them. */
/* TWO ILLUSTRATIONS, NOT ONE PLANT IN TWO COLOURS. `assets/icon-efficient-industry`
   and `icon-inefficient-industry` are a matched pair drawn for this project, so the
   reader is comparing two states of one thing rather than the same outcome in a
   warning colour. They replace the hand-drawn sawtooth-and-stack that stood in while
   there was nothing to draw from.

   Embedded with <image> inside the SVG so they scale with the diagram's own viewBox
   like every other element in it. The drawn fallback is kept and used when the file
   is missing — over `file://` from an unexpected directory, say — because a section
   whose punchline is an icon should not lose its punchline to a 404. */
const GOODPLANT=`
  <path class="fl-fac" d="M0 76 L0 30 L20 44 L20 30 L40 44 L40 30 L60 44 L60 76 Z"/>
  <path class="fl-fac" d="M9 14 L9 34 M9 14 q7 -9 14 0"/>
  <rect class="fl-fac" x="25" y="56" width="14" height="20"/>
  <path class="fl-fac fl-rise" d="M46 68 L46 56 M52 68 L52 50 M58 68 L58 44"/>`;
const BADPLANT=`
  <path class="fl-fac" d="M0 76 L0 30 L20 44 L20 30 L40 44 L40 30 L60 44 L60 76 Z"/>
  <path class="fl-fac" d="M9 14 L9 34 M9 14 q7 -9 14 0"/>
  <rect class="fl-fac" x="25" y="56" width="14" height="20"/>
  <path class="fl-puff" d="M6 8 q4 -6 9 -2 M14 2 q5 -6 10 -1 M20 9 q5 -6 10 -1"/>
  <path class="fl-crack" d="M20 44 L26 52 L21 58 L27 66"/>
  <path class="fl-drip" d="M32 76 L32 82 M36 76 L36 80"/>`;
function plantArt(bad){
  const src = typeof ICO==='function' && ICO(bad?'Inefficient industry':'Efficient industry');
  return src
    ? `<image href="${src}" x="0" y="0" width="150" height="150" preserveAspectRatio="xMidYMid meet"/>`
    : `<g transform="scale(1.7)" class="fl-plant ${bad?'bad':'good'}">${bad?BADPLANT:GOODPLANT}</g>`;
}
function flowSVG(bad){
  /* ONE DIAGRAM, TWO ENDINGS. The toggle is not decoration: the same funnel is
     the argument for the product and the description of the status quo, and a
     reader who has only seen the happy version has not been told what the cost
     of missing is. So the geometry is identical and only the OUTCOME moves.

     ── ZONES, BECAUSE SVG TEXT CANNOT BE MEASURED BEFORE IT IS DRAWN ──────────
     This broke on a 13-inch MacBook and the reason is worth writing down. A label
     placed at an x with `text-anchor:start` occupies however many units its glyphs
     need, and that width scales with the frame — so at 1.0x "OVER DECADES" cleared
     "INDUSTRIAL R&D" and at 1.4x it did not. There is no measurement available at
     build time and a collision that appears at one scale is invisible at another.

     So the diagram is divided into four ZONES with a guaranteed gutter, and every
     label is anchored INSIDE its own zone: A starts at its left edge, D ends at its
     right edge, B and C start at theirs. The widest label any zone can hold is
     about 230 units and no two zone anchors are closer than 280, so the collision
     cannot happen at any scale rather than not happening at the one that was
     checked. The viewBox went from 1000 to 1200 wide to buy that room. */
  const W=1200, H=430, CY=214, C=bad?'bad':'good';
  const Z={a:[30,300], b:[380,620], c:[700,880], d:[950,1170]};
  const T=bad
    ? ['Technology missed','Dollars wasted','Longer timelines','Less efficient industry']
    : ['Thousands of technologies','Millions of dollars','Over decades','Efficient industry'];
  /* the field is exactly 1,000 squares — 40 x 25 — because the label says
     "thousands of technologies" and a field of 224 would be the caption lying.

     ── IT IS CENTRED ON CY, AND THE TOP IS SOLVED RATHER THAN TYPED ──────────
     The coins, the decade axis and the plant all sit on CY. The field was pinned at
     y=88, which put its own centre 45 units ABOVE that line, so the first element in
     the row was the one thing not aligned to it — visible as a step down from the
     squares to the coins. The top is now derived from the row count and the pitch, so
     changing either keeps the field centred instead of quietly re-breaking it. */
  const FR=25, FC=40, FP=6.6, FS=3.2;          /* rows, cols, pitch, square size */
  const fieldTop = CY - ((FR-1)*FP+FS)/2;
  let field='';
  for(let r=0;r<FR;r++) for(let c=0;c<FC;c++){
    /* in the bad view a scatter of them is never picked up at all. A deterministic
       scatter, not Math.random, so the picture is the same on every load. */
    const miss = bad && ((r*FC+c)*7919)%23===0;
    field+=`<rect class="fl-t${miss?' fl-miss':''}" x="${Z.a[0]+c*FP}" y="${(fieldTop+r*FP).toFixed(1)}"
      width="${miss?4.4:FS}" height="${miss?4.4:FS}"/>`;
  }
  /* five streams from the field to the money, three of them breaking short of it
     when it goes wrong. They END at the first coin's edge rather than at a point in
     mid-air, which is what made the flow look like it stopped. */
  const coinX=i=>Z.b[0]+30+i*54, coinR=21;
  const meet=coinX(0)-coinR-6;
  /* the streams leave the field and converge on the coins, so their fan is centred on
     CY as well — five at a 52-unit pitch is a 208-unit span, half of it either side */
  const streams=[0,1,2,3,4].map(i=>{
    const y0=CY-104+i*52, brk=bad&&[0,2,4].includes(i);
    const end = brk ? `${Z.a[1]+52} ${y0+10}` : `${meet} ${CY}`;
    const c1 = brk ? `${Z.a[1]+22} ${y0}, ${Z.a[1]+38} ${y0+6}` : `${Z.a[1]+70} ${y0}, ${Z.a[1]+84} ${CY}`;
    return `<path class="fl-line ${C}${brk?' fl-brk':''}" d="M${Z.a[1]+4} ${y0} C ${c1}, ${end}"
      style="animation-delay:-${(i*0.3).toFixed(1)}s"/>` }).join('');
  const coins=[0,1,2,3].map(i=>{ const cx=coinX(i);
    return `<g><circle class="fl-coinR ${C}" cx="${cx}" cy="${CY}" r="${coinR}"/>
      <text class="fl-coin ${C}" x="${cx}" y="${CY+6}" text-anchor="middle">$</text>
      ${bad?`<path class="fl-strike" d="M${cx-17} ${CY+17} L${cx+17} ${CY-17}"/>`:''}</g>` }).join('');
  /* the decade axis fills zone C whatever the number of ticks, so the bad view's
     extra decade compresses the pitch instead of running into the next zone */
  const yrs = bad? ["'30","'40","'50","'60","'70"] : ["'30","'40","'50","'60"];
  const axL=Z.c[0]+8, axR=Z.c[1]-8, pitch=(axR-axL)/(yrs.length-1);
  const dec=yrs.map((y,i)=>{ const x=axL+i*pitch;
    return `<g><line x1="${x.toFixed(1)}" y1="${CY-17}" x2="${x.toFixed(1)}" y2="${CY+17}"
      stroke="hsl(150,12%,45%)" stroke-opacity=".6" stroke-width="1.8"/>
      <text class="fl-sub" x="${x.toFixed(1)}" y="${CY+44}" text-anchor="middle">${y}</text></g>` }).join('');
  const plantX=Z.d[0]+22, plantCx=plantX+63;
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img"
      aria-label="${bad?'technologies missed, money wasted, timelines stretched'
        :'a thousand technologies, narrowing through capital and time to one efficient plant'}">
    <text class="fl-lbl ${C}" x="${Z.a[0]}" y="62">${T[0]}</text>
    ${field}${streams}
    <text class="fl-lbl ${C}" x="${Z.b[0]}" y="62">${T[1]}</text>
    ${coins}
    <path class="fl-line ${C}" d="M${coinX(3)+coinR+4} ${CY} C ${Z.b[1]+10} ${CY}, ${Z.b[1]+22} ${CY}, ${axL-6} ${CY}"/>
    <text class="fl-lbl ${C}" x="${Z.c[0]}" y="62">${T[2]}</text>
    <line x1="${axL}" y1="${CY}" x2="${axR}" y2="${CY}" stroke="hsl(150,12%,45%)"
      stroke-opacity=".34" stroke-width="1.8"/>
    ${dec}
    <path class="fl-line ${C}" d="M${axR+8} ${CY} C ${Z.c[1]+26} ${CY}, ${Z.c[1]+40} ${CY}, ${plantX-10} ${CY}"/>
    <text class="fl-lbl" x="${Z.d[1]}" y="62" text-anchor="end">Industrial R&amp;D</text>
    <g transform="translate(${plantX-12},${CY-75})">${plantArt(bad)}</g>
    <text class="fl-sub ${C}" x="${plantCx}" y="${CY+96}" text-anchor="middle">${T[3]}</text>
  </svg>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   THE OPERATING POINT: ONE CONTROL, ONE 2x2
   Both surfaces show the same trade-off — the app in its "Tune your model"
   modal, the landing page in "Don't miss weak signals" — so both come from
   here. The landing page is the one place a reader meets precision and recall
   for the first time, which is why the 2x2 exists at all; the app inherits it
   because the analyst choosing a setting deserves the same picture.
   ══════════════════════════════════════════════════════════════════════════ */

/* Settings that resolve to the same cut-off are FOLDED INTO ONE CHOICE. Fβ
   optima coincide whenever recall plateaus, which happens often on a model that
   is working — and five controls doing three things is a failure this file has
   already made once, with the alert's tight/balanced collision. */
function fGroups(F){
  const groups=[];
  FSET.forEach(([k])=>{ const last=groups[groups.length-1];
    if(last && F[last.keys[0]]===F[k]) last.keys.push(k); else groups.push({keys:[k]}) });
  return groups;
}
const fName=k=>FSET.find(f=>f[0]===k)[1];

/* `cb` is a JS expression string, given the chosen key as `%k`. A string rather
   than a function because these buttons live in innerHTML and each surface
   re-renders itself differently.

   TWO MODES, BECAUSE THE TWO SURFACES ARE DOING DIFFERENT JOBS.
   · FOLDED (the app's default). Settings that share a cut-off become one button,
     so a control never offers five choices that do three things.
   · UNFOLDED (`opt.fold===false`, the landing page). All five stops are shown,
     because the section's subject IS the scale — precision at one end, recall at
     the other — and a reader meeting the words for the first time has to see
     where F0.5, F1 and F2 sit between them. Collapsing to three would teach the
     wrong thing. It does not pretend they differ: a stop sharing its cut-off with
     the one before says so, on the card, in its own line. */
function fRadio(pts,F,cur,cb,opt){
  if(opt&&opt.fold===false){
    /* THE NAME AND NOTHING ELSE. Each card used to carry a gloss ("only what it is
       surest of") and a cost ("87 papers · 41% right"), which is the right content in
       the app's modal — an analyst is choosing — and the wrong content here, where the
       reader is being shown a SCALE. Five names in a row, ordered precision to recall,
       IS the scale; the costs are drawn full-screen above it as the sample changes
       colour, so printing them again in 13px made the strip the thing being read.
       The one line that stays is conditional: when two stops resolve to the same
       cut-off the card says so, because two identical settings presented as different
       choices is the failure this whole control was rebuilt to avoid. */
    return `<div class="radioRow bare" style="--cols:${FSET.length}">
      ${FSET.map(([k],i)=>{
        const same=i>0 && F[FSET[i-1][0]]===F[k];
        return `<button class="fOpt ${k===cur?'on':''} ${same?'dup':''}" onclick="${cb.replace('%k',k)}">
          <span class="fk">${fName(k)}</span>
          ${same?`<span class="fd">same cut-off as ${fName(FSET[i-1][0])}</span>`:''}</button>` }).join('')}
    </div>`;
    /* No "tighter / looser" strip in unfolded mode: the row is ordered by it, so the
       two captions were restating the axis the row already draws. */
  }
  const groups=fGroups(F);
  return `<div class="radioRow" style="--cols:${groups.length}">
    ${groups.map(g=>{ const p=pts[F[g.keys[0]]];
      return `<button class="fOpt ${g.keys.includes(cur)?'on':''}" onclick="${cb.replace('%k',g.keys[0])}">
        <span class="fk">${g.keys.map(fName).join(' · ')}</span>
        <span class="fs">${g.keys.length>1 ? 'these all land on the same cut-off'
          : FSET.find(f=>f[0]===g.keys[0])[2]}</span>
        <span class="fn">${p.n} papers · ${(p.pr*100).toFixed(0)}% right</span></button>` }).join('')}
  </div>
  <div class="radioEnds"><span>Tighter — read less, miss more</span>
    <span>Looser — miss less, read more</span></div>`;
}

/* The 2x2, counted against the gold answers. This is legitimate ONLY on the
   fixture, where every paper's answer is known; the app's own panel measures on
   the analyst's labels alone and says so, because nobody knows the answers for
   the rest of a real pool. The landing page is allowed the full table precisely
   because it is explaining what the words mean, not reporting on your corpus. */
function confMatrix(cutSet,items,opt){
  const IT=items||PAPERS, lean=!!(opt&&opt.lean);
  let tp=0,fp=0,fn=0,tn=0;
  IT.forEach(p=>{ const inCut=cutSet.has(p.id), rel=p.g==='pos';
    if(inCut&&rel)tp++; else if(inCut&&!rel)fp++; else if(!inCut&&rel)fn++; else tn++ });
  const pr=tp+fp?tp/(tp+fp):0, rc=tp+fn?tp/(tp+fn):0;
  /* ── THE DOT IS THE KEY, NOT THE TYPE COLOUR ────────────────────────────────
     The first version tinted each cell's number and name in that outcome's colour,
     which made the table its own legend by re-colouring text — and coloured text is
     the weakest way to carry a key: it fails for anyone with a red-green deficiency,
     it drags the type off the page's ink colour, and it cannot be matched to a dot
     on a scatter plot without the reader doing the mapping themselves. A swatch in
     the mark's own shape and colour, beside a name in ordinary ink, is the standard
     annotation and it maps one-to-one onto the plot.

     Everything else got shorter and bigger: standard terms (true/false positive and
     negative), no gloss sentence per cell, and the numbers are the largest thing in
     the block. Inside a chart frame this table is a key, and a key is read at a
     glance or not at all. */
  const cell=(cls,n,k,d)=>`<div class="cmC ${cls}">
    <b>${n.toLocaleString()}</b>
    <span class="cmk"><i class="cmDot ${cls}"></i>${k}</span>
    ${lean?'':`<span class="cmd">${d}</span>`}</div>`;
  /* LEAN DROPS THE AXES AND THE SUMMARY. Inside a chart frame this block is the
     plot's key, and the four cell names already say which quadrant each is —
     "false positive" IS "predicted yes, actually not". The axis strips restated
     that in eight more words on two more rows, and the precision/recall line
     underneath restated it again as percentages. Standard terms carry it. */
  if(lean) return `<div class="cmGrid lean">
      ${cell('ctp',tp,'True positive','')}
      ${cell('cfp',fp,'False positive','')}
      ${cell('cfn',fn,'False negative','')}
      ${cell('ctn',tn,'True negative','')}
    </div>`;
  return `<div class="cmGrid">
      <div class="cmCorner"></div>
      <div class="cmAx">Actually interesting</div>
      <div class="cmAx">Actually not</div>
      <div class="cmRow">Model<br>says yes</div>
      ${cell('ctp',tp,'True positive','you read it, and it mattered')}
      ${cell('cfp',fp,'False positive','you read it for nothing')}
      <div class="cmRow">Model<br>says no</div>
      ${cell('cfn',fn,'False negative','you never saw it')}
      ${cell('ctn',tn,'True negative','you were right to skip it')}
    </div>
    <div class="cmSum">
      <span><b>${(pr*100).toFixed(0)}%</b> precision</span>
      <span><b>${(rc*100).toFixed(0)}%</b> recall</span>
    </div>`;
}
