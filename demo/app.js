/* ── THE RANKED ORDER ───────────────────────────────────────────────────
   The fixture lists all the relevant papers first, so dealing in array order
   put 10 positives in the first 11 cards — which is not a ranking, it is a
   sorted answer key, and it destroys the only thing this deck exists to show.
   A real ranked deck front-loads relevance and THINS: positive density starts
   high and decays as you work down. Placing the jth positive at
   (N-1)·(j/(P-1))^1.3 reproduces that shape deterministically — ~45% in the
   first dozen, ~12% by the end — which is the "generous but not absurd" yield
   the header declares. Negatives fill the gaps in their own order. */
const RANKED=(()=>{
  const pos=DEALABLE.filter(p=>p.arm==='ranked'&&p.g==='pos');
  const neg=DEALABLE.filter(p=>p.arm==='ranked'&&p.g==='neg');
  const N=pos.length+neg.length, slots=new Array(N).fill(null);
  pos.forEach((p,j)=>{
    let s=Math.round((N-1)*Math.pow(j/(pos.length-1||1),1.3));
    while(s<N&&slots[s]) s++;            // nearest free slot, never overwrite
    while(s>=N&&slots[s]) s--;
    slots[Math.min(Math.max(s,0),N-1)]=p;
  });
  let n=0;
  for(let i=0;i<N;i++) if(!slots[i]) slots[i]=neg[n++];
  return slots.filter(Boolean);
})();
function dealBatch(){
  const done=new Set(Object.keys(S.lab).map(Number));
  const inDeck=id=>S.deck.some(c=>c.k==='paper'&&c.id===id);
  const ranked=RANKED.filter(p=>!done.has(p.id)&&!inDeck(p.id)).slice(0,BATCH-1).map(p=>({k:'paper',id:p.id}));
  const ctrl=DEALABLE.filter(p=>p.arm==='control'&&!done.has(p.id)&&!inDeck(p.id))
    .slice(0,1).map(p=>({k:'paper',id:p.id}));
  ctrl.forEach((c,n)=>ranked.splice(9+n*7,0,c));
  S.batch++;
  return ranked;
}
const TPL={q:'',tag:null};
function setArch(v){ S.arch=v;
  document.getElementById('tabActive').classList.toggle('on',!v);
  document.getElementById('tabArch').classList.toggle('on',v); renderHome() }
function newUc(){ S.lab={};S.deck=[];S.i=0;S.got.clear();S.ghostPos=0;S.ghostNeg=0;S.batch=0;
  go('describe') }
function renderHome(){
  const live={...UCS[0],papers:seen()||S.deck.length?4120:4120,lab:seen(),pos:nPos(),neg:nNeg(),
    auc:(nPos()>=10&&nNeg()>=50)?0.83:null};
  const rows=[live,...UCS.slice(1)].filter(u=>!!u.arch===S.arch);
  document.getElementById('wSub').textContent=
    'Each use case is its own pool, its own labels and its own model. Nothing is shared between them.';
  document.getElementById('wCount').textContent=`${rows.length} ${S.arch?'archived':'active'}`;
  const wrap=document.getElementById('ucWrap');
  if(!rows.length){
    wrap.innerHTML=`<div class="empty">
      <h2>Nothing here yet</h2>
      <p>A use case is one question you want the literature to answer, and everything else —
        the papers, your decisions, the model — hangs off it.</p>
      <div class="howstrip">
        <div><span class="n">1</span><span class="t">Describe it</span><span class="d">Five questions. Two minutes.</span></div>
        <div><span class="n">2</span><span class="t">We find the papers</span><span class="d">Seven sources, deduplicated and ranked.</span></div>
        <div><span class="n">3</span><span class="t">Sort about seventy</span><span class="d">Two keys. Twenty to thirty minutes.</span></div>
        <div><span class="n">4</span><span class="t">The model takes over</span><span class="d">It sorts the rest and you check its picks.</span></div>
      </div>
      <div style="display:flex;gap:10px;justify-content:center">
        <button class="btn" onclick="newUc()">Create your first use case</button>
        <button class="btn ghost" onclick="openTpl()">Start from an example</button></div></div>`;
    return;
  }
  wrap.innerHTML=`<table class="ucT"><thead><tr>
      <th>Use case</th><th class="r">Papers</th><th class="r">Sorted</th><th class="r">Relevant / not</th>
      <th class="r">Model quality</th><th class="r">Created</th><th class="r">Last worked on</th><th></th>
    </tr></thead><tbody>${rows.map(u=>`
    <tr class="row ${u.arch?'arch':''}" onclick="${u.live?"openLive()":`alert('Fixture: only “Low-carbon cement binders” is walkable in this prototype.')`}">
      <td><span class="nm">${u.nm}</span><span class="sub">${u.sub}</span></td>
      <td class="r"><span class="num">${(u.papers||0).toLocaleString()}</span></td>
      <td class="r"><span class="num">${u.lab||0}</span></td>
      <td class="r"><span class="split">${u.lab? `<b>${u.pos}</b> <span>/ ${u.neg}</span>` : '<span>—</span>'}</span></td>
      <td class="r">${u.auc
        ? `<span class="mq"><span class="qbar"><i style="width:${Math.round((u.auc-.5)*200)}%"></i></span><span class="num">${u.auc.toFixed(2)}</span></span>`
        : `<span class="qnone">${(u.pos||0)>=3?'not yet':'needs 3 relevant'}</span>`}</td>
      <td class="r"><span class="sub">${u.created}</span></td>
      <td class="r"><span class="sub">${u.last}</span></td>
      <td class="r"><span class="go">${u.live? (seen()? 'Carry on →':'Start →') : 'Open →'}</span></td>
    </tr>`).join('')}</tbody></table>`;
}
/* ── breadcrumb ─────────────────────────────────────────────────────────
   Four levels, and every ancestor is reachable. A step you have not earned is
   disabled rather than hidden, because "you cannot go there yet" is useful and
   "that does not exist" is misleading. */
const CRUMB=[['home','Use cases',()=>true],
 ['describe','Low-carbon cement binders',()=>true],
 ['sort','Labelling',()=>S.deck.length>0||seen()>0],
 ['use','Model',()=>S.got.has('auto')]];
function renderCrumbs(){
  const at=CRUMB.findIndex(c=>c[0]===S.screen||(S.screen==='find'&&c[0]==='describe')
    ||(S.screen==='emailprev'&&c[0]==='use'));
  document.getElementById('crumbs').innerHTML=CRUMB.map((c,i)=>{
    const ok=c[2](), here=i===at;
    return (i?'<span class="sep">›</span>':'')+
      `<button class="${here?'here':''}" ${ok?`onclick="go('${c[0]}')"`:'disabled'}
        title="${ok?'':'Not reached yet'}">${c[1]}</button>`}).join('');
}

/* ── welcome ──────────────────────────────────────────────────────────────
   It used to be gated on a localStorage flag, which is right for the product and
   wrong for a prototype under review: Warren opened it a second time and it never
   appeared again, so the thing he asked to look at was the one thing he could not
   see. The flag is now per-SESSION only — every reload shows it, dismissing it
   within a visit still works — and the app's real first-run behaviour is one line
   away, marked below.
   (localStorage is still wrapped where it is used elsewhere because it THROWS
   rather than returning null inside a data: URL, which killed a boot sequence.) */
/* Before the split this was an in-memory `let` and the comment said "the app
   reads localStorage". Now that the landing page is a separate document the flag
   has to CROSS a navigation, so localStorage is no longer the note — it is the
   mechanism, and the in-memory copy is only the fallback for a browser that
   refuses storage (Safari private mode throws on write, not on read). */
/* ── THE FIRST-RUN FLAG, AND THE WAY BACK ──────────────────────────────────
   Before the split this was an in-memory `let`, so the welcome fired on every
   page load. Moving it to localStorage — which it had to be, because the landing
   page is now a separate document and the flag has to cross a navigation — made it
   PERMANENT, and there was no way to see the modal again. That is a defect in two
   places at once: the prototype could no longer show its own first-run state, and a
   product whose one explanation of itself can only ever be seen once, by accident,
   on a machine that has never opened it, is a product that does not explain itself.

   So the flag stays, and there are three ways past it: `openWelcome()` from the
   workspace header (the product answer — "How it works" is a thing you go back to),
   the prototype strip, and `resetDemo()`, which clears the flag so that "reset"
   means reset rather than "reload with yesterday's state still in storage". */
let seenFlag=false;
const flagGet=()=>{ try{ return localStorage.getItem('tiri_seen')==='1' }catch(e){ return seenFlag } };
const flagSet=()=>{ seenFlag=true; try{ localStorage.setItem('tiri_seen','1') }catch(e){} };
function maybeWelcome(){
  if(flagGet()) return;
  openWelcome();
}
/* on demand, and it does NOT consult the flag — asking for it IS the answer */
function openWelcome(){
  wi=0; renderWlc();
  document.getElementById('wlcModal').classList.add('on');
}
function dismissWelcome(){ flagSet();
  document.getElementById('wlcModal').classList.remove('on') }
/* the strip's reset: clear the first-run flag too, or the one thing a reader most
   wants to re-see is the one thing reset cannot bring back */
function resetDemo(){ try{ localStorage.removeItem('tiri_seen') }catch(e){}
  seenFlag=false; location.reload() }

/* The four slides' CONTENT is in shared/data.js — the landing page's Data flow
   section renders the same four as a side-scroll, so a change to one of them has to
   change both. `renderWlc`/`wslide` below stay here: they are the modal's slider,
   which only the app has. */
let wi=0;
function renderWlc(){
  document.getElementById('wtrack').innerHTML=WSL.map((w,i)=>`
    <div class="wsl"><span class="wn">${i+1} / ${WSL.length}</span>
      <h2>${w.t}</h2><div class="wdia">${w.d}</div>
      <p>${w.c}</p><div class="fill"><b>Placeholder copy.</b> ${w.f}</div></div>`).join('');
  document.getElementById('wtrack').style.transform=`translateX(${-wi*100}%)`;
  document.getElementById('wdots').innerHTML=WSL.map((_,i)=>
    `<button class="${i===wi?'on':''}" onclick="wi=${i};renderWlc()" aria-label="slide ${i+1}"></button>`).join('');
  document.getElementById('wback').style.visibility= wi? 'visible':'hidden';
  document.getElementById('wfwd').textContent= wi===WSL.length-1? 'Get started':'Next →';
}
function wslide(d){
  if(wi+d>=WSL.length){ dismissWelcome(); return }
  wi=Math.max(0,wi+d); renderWlc();
}

/* ── templates ── */
function openTpl(){ document.getElementById('tplModal').classList.add('on'); renderTpl() }
function renderTpl(){
  const tags=[...new Set(TEMPLATES.map(t=>t[2]))].sort();
  document.getElementById('tplTags').innerHTML=
    `<button class="tplTag ${TPL.tag===null?'on':''}" onclick="TPL.tag=null;renderTpl()">All ${TEMPLATES.length}</button>`+
    tags.map(t=>`<button class="tplTag ${TPL.tag===t?'on':''}" onclick="TPL.tag='${t}';renderTpl()">${t}</button>`).join('');
  document.getElementById('tplCount').textContent=`${TEMPLATES.length} real screening projects`;
  const q=(TPL.q||'').toLowerCase();
  const rows=TEMPLATES.filter(t=>(!TPL.tag||t[2]===TPL.tag)&&
    (!q||t[0].toLowerCase().includes(q)||t[5].toLowerCase().includes(q)));
  document.getElementById('tplList').innerHTML = rows.length
    ? rows.map(([nm,v,tag,n,pos,ob])=>`<button class="tpl" onclick="useTpl('${nm.replace(/'/g,"\\'")}')">
        <span><span class="src">benchset ${v} · ${tag}</span><span class="nm">${nm}</span><span class="ob">${ob}</span></span>
        <span class="vol">${n.toLocaleString()}<span>papers screened</span></span>
        <span class="use">${pos.toLocaleString()} relevant<br><span style="color:var(--ink3);font-weight:400">use this →</span></span>
      </button>`).join('')
    : `<div class="tplNone">Nothing matches “${TPL.q}”.</div>`;
  document.getElementById('tplWarn').innerHTML=
    `These are real screening projects with published answers — good for seeing what a
     complete description looks like. <span class="im" data-tip="Their descriptions come from each review's own abstract, so they sit closer to the answer than a real protocol would. Copy the shape, not the wording.">i</span>`;
}
function useTpl(nm){
  const t=TEMPLATES.find(x=>x[0]===nm); if(!t) return;
  S.ans={name:t[0],kind:'A solution to a problem',systems:[],outcome:[],numbers:'',near:''};
  S.tplFrom=t; S.focusQ=0;
  document.getElementById('tplModal').classList.remove('on');
  go('describe');
}

function openLive(){
  if(S.got.has('auto')) return go('use');
  if(seen()||S.deck.length) return go('sort');
  go('describe');
}

/* ═══════════════════════════════════════════════════════════════════════
   DESCRIBE · the interview
   ═══════════════════════════════════════════════════════════════════════ */
/* ── THE QUESTIONS ────────────────────────────────────────────────────────
   Titles are NOUNS now, not sentences. A sentence reads like editorial and a noun
   reads like a field, and this is a form. The sentence that used to be the title
   survives as the hint underneath wherever it was doing work.

   `cost` is gone from every question. Each one carried a measured consequence —
   true, sourced, and three lines long on a screen that already had a hint, an
   example and a note. Six of them made the column a wall. The measurements live
   in DATA_BRIEF and NUMBERS where they can be cited; the form keeps only what
   changes what the analyst types.

   `providers` sits second on purpose. It is the only question whose answer
   changes what the REST OF THE APPLICATION is allowed to do, so it cannot be
   discovered at the bottom of the form after six LLM buttons have been offered. */
const QS=[
 {id:'name',t:'Short Name',hint:'Two to five words. It labels the use case everywhere, and it gives the questions below something to work from.',
  kind:'name',ph:'Low-carbon cement binders',ic:'tag',
  ai:'gated',   /* nothing to improve until there is something typed */
  sup:{h:'Short and descriptive beats clever',p:'You will see this in a list of a dozen others in three months. Name the subject, not the project.',
   eg:'<b>Works:</b> “Solar cells for LEO satellites” · “Carbon capture — low temperature”<br><b>Doesn\'t:</b> “Round 2”, “Scouting exercise”, “New idea”'}},
 {id:'providers',t:'AI Providers',hint:'Where AI features may run — and whether they run at all. This one setting applies everywhere in the app.',
  kind:'llm',ic:'shield',always:true,
  sup:{h:'Where your data goes',privacy:true}},
 {id:'kind',t:'Use Case Type',hint:'This aims the search. Most use cases are one of three.',
  kind:'opts',ic:'shapes',ai:false,
  opts:[['A material or technology','A thing that exists or could exist — a binder, a cell, a sorbent. Searches for the thing itself.'],
        ['A way of doing something','A method, process or measurement technique. Searches for how work is done.'],
        ['A solution to a problem','An outcome you want, whatever delivers it. The broadest of the three.']],
  other:true,
  sup:{h:'Which one to pick',
   p:'The same words mean different things on each axis. “Carbon capture” as a <b>technology</b> means sorbents and processes; as a <b>problem</b> it means anything that lowers emissions, including things that are not capture at all.',
   eg:'<b>Pick “material or technology”</b> when you could point at the thing — <i>perovskite solar cells</i>, <i>alkali-activated slag</i>, <i>amine sorbents</i>.<br><br><b>Pick “a way of doing something”</b> when the thing is a procedure — <i>named entity recognition</i>, <i>accelerated carbonation testing</i>, <i>LLM scoring of papers</i>.<br><br><b>Pick “a solution to a problem”</b> when you do not care what delivers it — <i>cut embodied carbon by half</i>, <i>predict a breakthrough earlier</i>, <i>improve soil health</i>. This is the broadest and returns the most off-target work, so it needs the near-misses filled in.'}},
 {id:'systems',t:'Specific Interests',hint:'The actual things, not the category.',
  kind:'chips',ideal:[4,8],ic:'target',
  opts:['geopolymers','alkali-activated slag','calcined clay','limestone blends','LC3','reactive MgO','recycled fines','steel slag'],
  sup:{h:'Be specific, not broad',p:'This is what does the matching. A category name matches everything in the category, which is the same as matching nothing.',
   eg:'<b>Works:</b> “alkali-activated slag, calcined clay–limestone blends, reactive MgO”.<br><b>Doesn\'t:</b> “sustainable materials”, “green cement”.'}},
 {id:'outcome',t:'Desired Properties',hint:'The properties or results that make a paper useful to you.',
  kind:'chips',ideal:[3,6],ic:'gauge',
  opts:['compressive strength','durability','carbonation','chloride resistance','setting time','shrinkage','emissions','cost'],
  sup:{h:'This is what separates near-misses',p:'Most papers that survive the first two questions are about the right materials for the wrong reason — a market forecast, a logistics study, a life-cycle assessment. What has to be <em>shown</em> is what rules those out.',
   eg:'“Compressive strength development, carbonation and chloride durability, setting behaviour.”'}},
 {id:'numbers',t:'Performance Metrics',hint:'Optional, and the strongest single thing you can add if you have one.',
  kind:'chips',ideal:[1,3],ic:'bars',
  opts:['above 40% clinker replacement','28-day strength','TRL 4–8','below €80/tonne','>50% embodied carbon cut'],
  sup:{h:'A number is worth more than a paragraph',p:'A threshold is unambiguous in a way prose is not, and it is the one thing a reader can check a paper against directly.',
   eg:'<b>Works — a quantity, a direction and a unit:</b><br>“above 40% clinker replacement” · “28-day strength ≥ 40 MPa” · “below €80/tonne” · “TRL 4–8” · “&gt;50% embodied carbon cut” · “operates under 100 °C”<br><br><b>Doesn\'t — no direction, or no unit:</b><br>“high strength” (how high?) · “cost effective” (against what?) · “40” (of what?) · “state of the art” · “significant reduction”<br><br><b>Include the age or condition when it moves the answer:</b> “28-day” and “90-day” strength diverge sharply in these systems, so the number without the age is two different claims.'}},
 {id:'near',t:'Near-misses',hint:'The near-misses. The question people skip, and the one that pays most.',
  kind:'chips',ideal:[2,5],ic:'warn',
  opts:['dental cements','plant economics','procurement policy','pavement logistics','heritage lime mortars','bibliometrics'],
  sup:{h:'What gets confused with this',
   p:'Things that use your exact vocabulary and are not what you mean. They are the papers you would otherwise read and reject one at a time, forever.',
   eg:'<b>Same word, different field:</b> “cement” → dental cements, bone cements, cement <i>industry</i> equities.<br><br><b>About your thing, but not about the science:</b> plant economics · market forecasts · procurement and policy · patent landscaping · bibliometrics of the field.<br><br><b>Adjacent material, different question:</b> heritage lime mortars · pavement logistics · geotechnical settlement.<br><br><b>Right words, wrong direction:</b> a life-cycle assessment that assumes the binder rather than measuring it.'}},
 {id:'seeds',t:'Interesting Papers',hint:'Optional. Paste identifiers or drop a file — they arrive already marked relevant.',
  kind:'seeds',ic:'filePlus',ai:false,
  sup:{h:'What a seed does',
   p:'A paper you already know is relevant is worth more than any sentence you could write about it, because the ranker reads the whole abstract rather than your summary of it. Its references come in too, and the review itself leads the pool rather than only seeding it.',
   eg:'DOIs, PubMed IDs or arXiv IDs, one per line or comma separated. Or a .ris / .bib / .csv straight out of Zotero, EndNote or Mendeley.<br><br>Nothing is uploaded — the file is read here.'}}
];

function setAns(id,v){ S.ans[id]=v; renderDescribe() }
function toggleChip(id,v){ const cur=S.ans[id]||[]; S.ans[id]=cur.includes(v)?cur.filter(x=>x!==v):[...cur,v]; renderDescribe() }
function focusQ(i){ S.focusQ=i; renderDescribe() }
/* Keeps the chips the analyst clicked and replaces only what they typed, so the two
   input routes into one field never overwrite each other. */
function typeOwn(id,v){
  const preset=(QS.find(q=>q.id===id)||{}).opts||[];
  const chosen=(S.ans[id]||[]).filter(x=>preset.includes(x));
  S.ans[id]=[...chosen,...v.split(',').map(s=>s.trim()).filter(Boolean)];
  renderDescribe();
  const el=document.querySelector(`.q:nth-child(${QS.findIndex(q=>q.id===id)+1}) input`);
  if(el){ el.focus(); el.setSelectionRange(el.value.length,el.value.length) }
}
const STRENGTH=[
 ['A short, clear name', ()=>!!(S.ans.name||'').trim()],
 ['A real description', ()=>compiled().length>90],
 ['Specific systems named', ()=>(S.ans.systems||[]).length>=3],
 ['What has to be shown', ()=>(S.ans.outcome||[]).length>=2],
 ['A number or threshold', ()=>((S.ans.numbers||[]).length||0)>0],
 ['Near-misses named', ()=>((S.ans.near||[]).length||0)>0],
 ['Papers you already know', ()=>!!(S.ans.seeds||'').trim()]
];
function terms(){
  const t=compiled().toLowerCase(), w=(t.match(/[a-z][a-z-]{2,}/g)||[]), f={};
  for(let i=0;i<w.length-1;i++){ if(STOP.has(w[i])||STOP.has(w[i+1]))continue; f[w[i]+' '+w[i+1]]=(f[w[i]+' '+w[i+1]]||0)+1 }
  w.filter(x=>!STOP.has(x)&&!FILLER.has(x)&&x.length>5).forEach(x=>f[x]=(f[x]||0)+.6);
  return Object.entries(f).sort((a,b)=>b[1]-a[1]).map(e=>e[0]).slice(0,9);
}
function dropTerm(t,e){ S.dropped.has(t)?S.dropped.delete(t):S.dropped.add(t); renderDescribe(); e.stopPropagation() }

/* Autosuggest vocabulary. Retrieved, never generated — in the app these come from
   the OpenAlex topic spine, S-CE's parenthetical gloss pairs and the corpus's own
   terms, which is why they are grouped by where they came from. Deliberately NOT
   accumulated across use cases: 68.9% of the vocabulary is singleton to one use
   case and global alias-folding produced 0 novel bridges out of 206 while merging
   cement's `fly ash` with the perovskite cation `formamidinium`. */
const VOCAB={
 systems:{'From the topic index':['alkali-activated slag','geopolymer','calcined clay','limestone calcined clay cement',
   'ground granulated blast-furnace slag','reactive magnesium oxide','fly ash','metakaolin','belite cement',
   'carbonatable calcium silicate','one-part activator','sodium carbonate activation'],
  'Seen in this corpus':['recycled concrete fines','steel slag','bauxite residue','dredged sediment','incinerator bottom ash']},
 outcome:{'Properties':['compressive strength','flexural strength','elastic modulus','carbonation','chloride resistance',
   'sulfate resistance','alkali-silica reaction','freeze-thaw durability','autogenous shrinkage','drying shrinkage',
   'setting time','workability retention','bond strength','pore structure'],
  'Whole-system':['embodied carbon','activator carbon footprint','cost per tonne','feedstock availability']}
};
function acFor(qid,frag){
  const src=VOCAB[qid]; if(!src||frag.length<2) return [];
  const f=frag.toLowerCase(), out=[];
  Object.entries(src).forEach(([grp,list])=>{
    const hits=list.filter(v=>v.toLowerCase().includes(f)&&!(S.ans[qid]||[]).includes(v));
    if(hits.length) out.push([grp,hits.slice(0,5)]);
  });
  return out;
}
function acPick(qid,v){
  S.ans[qid]=[...(S.ans[qid]||[]),v]; S.acFrag=''; renderDescribe();
}
function acType(qid,v){ S.acFrag=v; S.acQ=qid; renderDescribe(true) }

function renderDescribe(keepFocus){
  const answered=i=>{ const q=QS[i],v=S.ans[q.id];
    if(q.always) return true;   /* a setting with a default is never "unanswered" */
    return (q.kind==='chips')? (v||[]).length>0 : !!(v&&String(v).trim()) };
  const firstOpen=QS.findIndex((q,i)=>!answered(i));
  const reach=firstOpen<0? QS.length-1 : firstOpen;

  document.getElementById('iLede').innerHTML=`${QS.filter(q=>!q.always).length} questions, one at a
    time. They build the description every paper gets matched against — the bar at the bottom shows
    it taking shape.`;
  document.getElementById('iProg').innerHTML=QS.map((q,i)=>
    `<i class="${answered(i)?'f':(i===S.focusQ?'at':'')}"></i>`).join('');

  document.getElementById('qs').innerHTML=QS.map((q,i)=>{
    const v=S.ans[q.id], done=answered(i), open=S.focusQ===i, later=i>reach&&!done&&!open;
    /* State as an icon, sized to be seen: tick when done, chevron down when open,
       chevron right when collapsed. Each question also carries its OWN icon, which
       is what the heading gets once it is open. */
    const mark = done? icon('check') : open? icon('chevDown') : icon('chevRight');
    if(!open){
      return `<div class="q ${done?'done':''} ${later?'later':''}" onclick="${later?'':`focusQ(${i})`}">
        <div class="qh"><span class="qn">${mark}</span><span class="qt">${icon(q.ic)} ${q.t}</span>
          ${['numbers','seeds','near'].includes(q.id)?'<span class="qm">optional</span>':''}</div>
        ${done? `<div class="qans">${q.kind==='chips'? (v||[]).join(' · ')
                  : q.kind==='llm'? llmSummary() : v}</div>`
              : `<div class="qhint">${q.hint}</div>`}</div>`;
    }

    let body='';
    if(q.kind==='name') body=`<input placeholder="${q.ph}" value="${v||''}"
      oninput="setAns('name',this.value)" maxlength="60">
      <div class="cnote">${(v||'').trim().split(/\s+/).filter(Boolean).length||0} words${
        (v||'').trim().split(/\s+/).filter(Boolean).length>6?' — long for a list. Two to five reads better.':''}</div>`;

    if(q.kind==='opts'){
      body=`<div class="optCol">${q.opts.map(([o,d])=>`
        <button class="optBig ${v===o?'sel':''}" onclick="setAns('${q.id}','${o}')">
          <span class="mk">${v===o?'✓':'✕'}</span>
          <span><span class="ot">${o}</span><span class="od">${d}</span></span></button>`).join('')}
        ${q.other?`<button class="optBig ${v&&!q.opts.some(o=>o[0]===v)?'sel':''}" onclick="S.otherOpen=true;renderDescribe()">
          <span class="mk">${v&&!q.opts.some(o=>o[0]===v)?'✓':'✕'}</span>
          <span><span class="ot">Something else</span><span class="od">None of these fit</span></span></button>`:''}
      </div>
      ${S.otherOpen?`<div class="otherBox">
        <input placeholder="Describe the kind of thing you're after" value="${v&&!q.opts.some(o=>o[0]===v)?v:''}"
          oninput="setAns('${q.id}',this.value)">
        <div class="cnote warnNote"><span class="im warn" data-tip="The three options above are the axes we have measured. Anything else works, but the search is not tuned for it and we cannot tell you how well it will do.">!</span>
          The three above are the axes we have evidence for. This will still work — we just cannot promise how well.</div></div>`:''}`;
    }

    if(q.kind==='chips'){
      const chosen=v||[], sug=q.opts.filter(o=>!chosen.includes(o));
      const ac=S.acQ===q.id? acFor(q.id,S.acFrag||'') : [];
      const [lo,hi]=q.ideal||[3,8];
      const state = chosen.length===0? 'none' : chosen.length<lo? 'few' : chosen.length>hi? 'many' : 'good';
      /* ONE format for suggested and added (Warren): same chip, the only difference
         is a + or a ✓ and which group it sits in. Two visual languages for the same
         act was the confusion. */
      body=`<div class="chipGroup"><div class="cgl">Added${chosen.length?` · ${chosen.length}`:''}</div>
          <div class="chips">${chosen.length? chosen.map(o=>
            `<button class="chipU on" onclick="toggleChip('${q.id}','${o.replace(/'/g,"\\'")}')">
              <span class="ck">✓</span>${o}</button>`).join('')
            : '<span class="emptyq">Nothing yet — pick from below or start typing.</span>'}</div></div>
        <div class="cnote ${state}">${{
          none:`Aim for ${lo}–${hi}.`,
          few:`${chosen.length} so far. ${lo}–${hi} is where this starts to pay.`,
          good:`${chosen.length} — good range.`,
          many:`${chosen.length} is more than we would use. Extra terms dilute rather than add.`}[state]}
          <span class="im" data-tip="${q.id==='systems'
            ?'Five terms beat one by a wide margin. Past about eight, extra terms overlap with the ones you already have and stop adding signal — and padding with generic words measured worse than leaving the field empty.'
            :'Enough to separate a relevant paper from a near-miss, few enough that each one is doing work. Overlapping terms are not harmful, they are just not useful twice.'}">i</span></div>
        <div class="chipGroup"><div class="cgl">Suggested</div>
          <div class="chips">${sug.map(o=>
            `<button class="chipU" onclick="toggleChip('${q.id}','${o.replace(/'/g,"\\'")}')">
              <span class="ck">+</span>${o}</button>`).join('')}</div></div>
        <div class="ac"><input id="ac_${q.id}" placeholder="or type your own — we suggest as you go"
          value="${S.acQ===q.id?(S.acFrag||''):''}" oninput="acType('${q.id}',this.value)"
          onkeydown="if(event.key==='Enter'&&this.value.trim()){acPick('${q.id}',this.value.trim())}">
          ${ac.length?`<div class="acList">${ac.map(([g,hits])=>`<div class="grp">${g}</div>`+
            hits.map(h=>`<button onclick="acPick('${q.id}','${h.replace(/'/g,"\\'")}')">${
              h.replace(new RegExp('('+S.acFrag+')','i'),'<b>$1</b>')}</button>`).join('')).join('')}</div>`:''}
        </div>`;
    }

    if(q.kind==='llm'){
      body=`<div class="llmMaster ${S.llmOff?'off':''}">
        <span class="tgl ${!S.llmOff?'on':''}" onclick="setLLM(!S.llmOff)" style="cursor:pointer">
          <span class="sw"><i></i></span></span>
        <span class="lmT"><b>${S.llmOff?'AI features are off':'AI features are on'}</b>
          <span>${S.llmOff
            ? 'Every “ask a model” button is hidden, summaries are unavailable, and alerts can only send titles. Retrieval, ranking, labelling and export are unaffected — none of them ever used a model.'
            : 'Used only where you press something, plus a scheduled alert if you turn summarising on. Retrieval, ranking and labelling never call a model.'}</span></span>
      </div>
      ${!S.llmOff?`<div class="fRow" style="margin-top:16px;margin-bottom:0">
        <label>${icon('sparkles')} Default model</label>
        <div class="fh">Every model picker in the app starts here — the brief, summaries and alerts.</div>
        <select onchange="S.model=this.value;renderDescribe()">${MODELS_AVAIL
          .map(m=>`<option ${S.model===m?'selected':''}>${m}</option>`).join('')}</select></div>`:''}`;
    }

    if(q.kind==='seeds'){
      /* Two routes side by side rather than three explainers stacked above one
         dropzone. The old version described the ways of getting papers in and then
         showed one of them; these are the two things you can actually do. */
      body=`<div class="seedTwo">
        <div class="seedCol"><div class="scH">${icon('file')} Paste identifiers</div>
          <textarea placeholder="10.1016/j.cemconres.2024.107…&#10;10.1016/j.cemconcomp.2024.105…&#10;arXiv:2403.01234" rows="6"
            oninput="setAns('seeds',this.value.trim()?this.value.trim().split(/[\s,;]+/).filter(Boolean).length+' identifiers pasted':'')"></textarea>
          <div class="scN">DOI, PubMed or arXiv. One per line, or comma separated.</div></div>
        <div class="seedCol"><div class="scH">${icon('upload')} Or drop a file</div>
          <div class="dropzone ${S.dragOver?'over':''}" id="dz"
            ondragover="event.preventDefault();S.dragOver=true;renderDescribe()"
            ondragleave="S.dragOver=false;renderDescribe()"
            ondrop="event.preventDefault();S.dragOver=false;setAns('seeds','24 papers from library.ris')">
            <div class="dzIn">${S.ans.seeds
              ? `<b>${icon('circleCheck')} ${S.ans.seeds}</b><span>Marked relevant on arrival.
                 <button class="linkbtn" onclick="setAns('seeds','')">Remove</button></span>`
              : `<b>Drop it here</b><span>or <button class="linkbtn" onclick="setAns('seeds','24 papers from library.ris')">choose a file</button></span>`}</div></div>
          <div class="scN">.ris · .bib · .csv from Zotero, EndNote or Mendeley. Read here, never uploaded.</div></div>
      </div>`;
    }

    const sup = q.sup.privacy? privacyBlock() : `
      <div class="qsup"><h4>${q.sup.h}</h4><p>${q.sup.p}</p>
        ${q.sup.eg?`<div class="egline">${q.sup.eg}</div>`:''}</div>`;

    /* AI help: subtle, collapsed, and absent entirely where it has no job.
       `ai:false`  — the question does not benefit (a fixed choice, a file drop).
       `ai:'gated'` — nothing to improve until something is typed, so the button
                      says why rather than sitting there live and returning
                      advice about an empty field. */
    const aiGate = q.ai==='gated' && !String(v||'').trim();
    const ai = (q.ai===false || !aiOK())? (q.ai===false? '' : `<div class="aiWrap">
        <span class="aiOffNote">${icon('shield')} AI features are off for this workspace.
          <button class="linkbtn" onclick="focusQ(1)">Change that</button></span></div>`)
      : `<div class="aiWrap">
      ${S.aiOpen===q.id? `<div class="aiPanel">
          <div class="aiTop"><span>${icon('sparkles')} Ask a model to help with this</span>
            <select id="mp" onchange="S.model=this.value">${MODELS_AVAIL
              .map(m=>`<option ${S.model===m?'selected':''}>${m}</option>`).join('')}</select>
            <button class="btn ghost sm" onclick="askAI()">Ask</button>
            <button class="linkbtn" onclick="S.aiOpen=null;renderDescribe()">Hide</button></div>
          ${S.aiBrief&&S.aiBrief.q===q.id?aiOut(q):''}
        </div>`
      : `<button class="aiLink" ${aiGate?'disabled':''} onclick="${aiGate?'':`S.aiOpen='${q.id}';renderDescribe()`}">
           ${aiGate? 'Type a name first — there is nothing to improve yet' : 'Ask a model for help'}</button>`}</div>`;

    return `<div class="q open ${done?'done':''}">
      <div class="qcols">
        <div class="qmain">
          <div class="qh"><span class="qn">${mark}</span><span class="qt">${icon(q.ic,'lg')} ${q.t}</span>
            ${['numbers','seeds','near'].includes(q.id)?'<span class="qm">optional</span>':''}</div>
          <div class="qhint">${q.hint}</div>
          <div class="qbody">${body}</div>
          ${ai}
          <div class="qnav">
            ${i<QS.length-1?`<button class="btn sm" onclick="focusQ(${i+1})">${done?'Next':'Skip'} <span style="opacity:.7">›</span></button>`:''}
            ${i>0?`<button class="btn ghost sm" onclick="focusQ(${i-1})"><span style="opacity:.7">‹</span> Back</button>`:''}
          </div>
        </div>
        <aside class="qtips">${sup}</aside>
      </div>
    </div>`;
  }).join('');

  const nDone=STRENGTH.filter(x=>x[1]()).length;
  document.getElementById('dockBars').innerHTML=STRENGTH.map((x,i)=>`<i class="${i<nDone?'f':''}"></i>`).join('');
  document.getElementById('dockTxt').innerHTML=`<b>${nDone} of ${STRENGTH.length}</b> things that make it work`;
  document.getElementById('dockList').innerHTML=STRENGTH.map(x=>`<div class="si ${x[1]()?'f':''}">
    <span class="m">${x[1]()?'✓':'○'}</span><span>${x[0]}</span></div>`).join('');
  const c=compiled();
  document.getElementById('compiled').className='txt'+(c?'':' ph');
  document.getElementById('compiled').textContent=c||'Your answers become a description here as you go.';
  const tm=terms();
  document.getElementById('dTerms').innerHTML = tm.length
    ? tm.map(x=>`<span class="chip${S.dropped.has(x)?' gone':''}" onclick="dropTerm('${x.replace(/'/g,"\\'")}',event)">${x}<x>×</x></span>`).join('')
    : '<span class="emptyq">Nothing yet.</span>';

  const ok=(S.ans.systems||[]).length>=2 && c.length>60 && !!(S.ans.name||'').trim();
  const btn=document.getElementById('goFind');
  btn.disabled=!ok;
  btn.innerHTML = ok? 'Preview and search <span style="opacity:.7">›</span>'
    : !(S.ans.name||'').trim()? 'Name it to continue' : 'Name two systems to continue';

  if(keepFocus){ const el=document.getElementById('ac_'+S.acQ);
    if(el){ el.focus(); el.setSelectionRange(el.value.length,el.value.length) } }
}

function privacyBlock(){
  return `<div class="qsup"><div class="privacy">
    <div class="h">Where your data goes</div>
    <ul><li><b>Stays here:</b> your papers, your decisions, and the maths that ranks them. Nothing is
      uploaded and nothing is shared with your other use cases.</li>
    <li><b>Leaves when you ask, and when an alert runs:</b> “ask a model” sends your description plus
      one paper at a time. A scheduled alert that summarises does the same on its schedule.
      Turn AI off above and neither happens.</li>
    <li><b>Never used to train anyone's model</b> but yours.</li></ul>
    <div class="regions"><div class="rh">${icon('shield')} Regions AI may run in <span class="im right" data-tip="Applies only to the AI features. Retrieval and ranking are local either way. If nothing is available in the regions you allow, the request fails rather than quietly going elsewhere — pipeline.py's eu_only policy errors instead of rerouting.">i</span></div>
      ${[['eu','EU / EEA','evidenced'],['us','United States',''],['cn','China',''],['row','Rest of world','']]
        .map(([k,l,tag])=>`<button class="rg ${S.regions[k]?'on':''}" onclick="togRegion('${k}')">
          <span class="mk">${S.regions[k]?'✓':'✕'}</span><span>${l}${tag?`<em>${tag}</em>`:''}</span></button>`).join('')}
      <div class="cnote">${Object.values(S.regions).filter(Boolean).length===0
        ? '<span class="im warn" data-tip="With no region allowed, every AI feature is off. Everything else still works — retrieval, ranking, labelling and export are all local.">!</span> No AI features will run.'
        : S.regions.eu&&!S.regions.us&&!S.regions.cn&&!S.regions.row
          ? 'EU/EEA only — the strictest setting that still runs.'
          : 'Wider routing means more models available and less control over where text goes.'}</div>
    </div></div></div>`;
}
function togRegion(k){ S.regions[k]=!S.regions[k]; renderDescribe() }
/* ── ASKING A MODEL ───────────────────────────────────────────────────────
   The output format follows the INPUT format. Six of these questions are answered
   with chips, and returning a paragraph about which chips to add left the analyst
   re-typing terms out of prose — the model had done the thinking and handed back
   homework. For those the answer is a row of chips that add on click.

   Two questions still get prose, because prose is what they take: the name is one
   string, and the type is a judgement about which of three axes fits.

   The label never moves: it is always the model's own name and always the word
   OPINION, because a non-expert cannot otherwise tell this apart from the
   deterministic measurements sitting beside it. */
const AISUG={
 systems:['one-part activators','carbonated slag aggregate','sodium carbonate activation',
   'ternary blends','ladle slag','calcined shale'],
 outcome:['setting and workability retention','autogenous shrinkage','sulfate resistance',
   'freeze–thaw durability','heat of hydration'],
 numbers:['90-day strength','activator modulus 1.2–1.6','curing below 15 °C','w/b below 0.45'],
 near:['bone and dental cements','cement industry equities','patent landscaping',
   'life-cycle assessment only','geotechnical settlement','concrete 3D printing']
};
const AIPROSE={
 name:'“Low-carbon cement binders” — four words, names the subject, and reads clearly next to nine others. Avoid a year or a round number; those belong in the version history.',
 kind:'Your systems list reads like a technology question rather than a problem one — you are naming binders, not outcomes. If you actually want anything that lowers embodied carbon, including things that are not binders, pick the third option instead.'
};
function askAI(){
  const m=document.getElementById('mp');
  const name=m? m.options[m.selectedIndex].text : S.model;
  S.model=name;
  const q=QS[S.focusQ];
  S.aiBrief={m:name,q:q.id,
    chips:(AISUG[q.id]||[]).filter(x=>!(S.ans[q.id]||[]).includes(x)),
    t:AIPROSE[q.id]||''};
  renderDescribe();
}
function aiOut(q){
  const b=S.aiBrief;
  const head=`<span class="who">${b.m.toUpperCase()} · AN OPINION</span>`;
  if(b.chips&&b.chips.length){
    return `<div class="aiout">${head}
      <div style="margin-bottom:10px">Six it thinks are missing. Add the ones you agree with —
        it has not seen your papers, only your description.</div>
      <div class="chips">${b.chips.map(c=>
        `<button class="chipU" onclick="toggleChip('${q.id}','${c.replace(/'/g,"\\'")}')">
          <span class="ck">+</span>${c}</button>`).join('')}</div>
      <div class="use"><button class="btn sm" onclick="useAI()">Add all ${b.chips.length}</button>
        <button class="linkbtn" style="margin-left:11px" onclick="S.aiBrief=null;renderDescribe()">Dismiss</button></div></div>`;
  }
  if(!b.t) return '';
  return `<div class="aiout">${head}${b.t}
    <div class="use">${q.kind==='name'
      ? `<button class="btn sm" onclick="useAI()">Use this name</button>`
      : `<button class="linkbtn" onclick="S.aiBrief=null;renderDescribe()">Dismiss</button>`}</div></div>`;
}
/* `useAI` was referenced by the panel's own button and never existed — clicking
   "Use this" threw. It applies the suggestion in whatever form the question takes. */
function useAI(){
  const b=S.aiBrief; if(!b) return;
  if(b.chips&&b.chips.length){
    S.ans[b.q]=[...(S.ans[b.q]||[]), ...b.chips];
  } else if(b.q==='name'){
    const m=b.t.match(/“([^”]+)”/); if(m) S.ans.name=m[1];
  }
  S.aiBrief=null; renderDescribe();
}
function setLLM(off){
  S.llmOff=off;
  if(off){ S.aiOpen=null; S.aiBrief=null; AL.summarise=false }
  renderDescribe();
}
function llmSummary(){
  if(S.llmOff) return 'Off — no AI anywhere in this workspace';
  const on=Object.entries({eu:'EU/EEA',us:'US',cn:'China',row:'Rest of world'})
    .filter(([k])=>S.regions[k]).map(([,l])=>l);
  return `${S.model} · ${on.length? on.join(', ') : 'no region allowed, so nothing will run'}`;
}

function demoFill(){ S.ans={...DEMO_ANS};
  S.focusQ=QS.length-1; renderDescribe() }

/* ── the preview: the one moment on this screen that is a reward ──────────
   Dark green, roomy, and it shows what the six questions actually produced —
   because up to here the analyst has been answering questions and has not once
   seen the thing they were building. */
function openPreview(){
  const nDone=STRENGTH.filter(x=>x[1]()).length, a=S.ans;
  const list=k=>(a[k]||[]).join(' · ')||'—';
  document.getElementById('prevBody').innerHTML=`
    <div class="prevHero">
      <span class="eyebrow">Ready to search</span>
      <h2>${a.name||'Your use case'}</h2>
      <p class="pq">${compiled()}</p>
      <div class="prevGrid">
        ${[['kind','Use Case Type'],['systems','Specific Interests'],['outcome','Desired Properties'],
           ['numbers','Performance Metrics'],['near','Near-misses'],['seeds','Interesting Papers']]
          .map(([k,lbl])=>`<div><span class="pk">${lbl}</span><span class="pv2">${
            Array.isArray(a[k])? (a[k].join(' · ')||'—') : (a[k]||'—')}</span></div>`).join('')}
      </div>
      <div class="prevFoot">
        <span class="pfl">${nDone} of ${STRENGTH.length} things filled in</span>
        <span class="pfb">${STRENGTH.map(x=>`<i class="${x[1]()?'f':''}"></i>`).join('')}</span>
        <span class="pfn">${nDone===STRENGTH.length
          ? 'Everything that pays is in place.'
          : `The ${STRENGTH.filter(x=>!x[1]()).map(x=>x[0].toLowerCase()).join(' and ')} would still help — you can add ${nDone>=STRENGTH.length-1?'it':'them'} later.`}</span>
      </div>
    </div>
  `;
  document.getElementById('prevModal').classList.add('on');
}

/* ═══════════════════════════════════════════════════════════════════════
   FIND
   ═══════════════════════════════════════════════════════════════════════ */
function startFind(){
  go('find');
  const rows=document.getElementById('hvRows'), st=document.getElementById('hvStages'), tot=document.getElementById('hvTotal');
  rows.innerHTML=''; st.innerHTML=''; tot.innerHTML='';
  document.getElementById('findH').textContent='Looking in seven places…';
  const max=Math.max(...SRC.map(s=>s[1]||0));
  SRC.forEach(([n,c],k)=>{
    const el=document.createElement('div'); el.className='hv'; el.style.animationDelay=(k*.12)+'s';
    el.innerHTML=`<span class="nm">${n}</span><span class="bw"><span class="bf${c===null?' none':''}"></span></span>
      ${c===null?'<span class="ct nul">no answer</span>':'<span class="ct">0</span>'}`;
    rows.appendChild(el);
    setTimeout(()=>{ el.querySelector('.bf').style.width=(c===null?100:Math.max(3,c/max*100))+'%';
      if(c!==null){ let v=0; const iv=setInterval(()=>{ v=Math.min(c,v+Math.ceil(c/13));
        el.querySelector('.ct').textContent=v.toLocaleString(); if(v>=c)clearInterval(iv) },40) }
    },300+k*120);
  });
  const t0=300+SRC.length*120+500;
  STAGES.forEach(([l,v],k)=>{
    setTimeout(()=>{
      if(k===0) document.getElementById('findH').textContent='Sorting out what came back…';
      const el=document.createElement('div'); el.className='stg run';
      el.innerHTML=`<span class="d"></span><span class="l">${l}</span><span class="v">…</span>`;
      st.appendChild(el);
      setTimeout(()=>{ el.className='stg fin'; el.querySelector('.v').textContent=v },620);
    },t0+k*720);
  });
  setTimeout(()=>{
    document.getElementById('findH').textContent='Ready.';
    tot.innerHTML=`<div class="hvtot"><div><div class="big">4,120 papers</div>
      <div class="sub">Ordered so the ones closest to your description come first.</div></div>
      <button class="btn" onclick="startSorting()">Start sorting →</button></div>
      <div class="hvnote">CORE didn't answer, and two of the others can't handle an “and”, so they got a
        looser version of the question — worth knowing before trusting a count from them.
        <button class="linkbtn" onclick="alert('Per source: the exact string sent, how it was rendered (boolean / flattened / unsupported), what came back, and whether a daily budget cut it short.\\n\\nBlocked on one migration today — search.py computes the string and throws it away.')">what each place was asked</button></div>`;
  }, t0+STAGES.length*720+400);
}
function startSorting(){ if(!S.deck.length) S.deck=dealBatch(); go('sort') }

/* ═══════════════════════════════════════════════════════════════════════
   THE UNLOCKS
   ═══════════════════════════════════════════════════════════════════════ */
const GATES=[
 {id:'first',n:'First relevant paper',s:'The search is aimed the right way',
  test:()=>nPos()>=1, h:'Found one.',
  p:'One relevant paper out of a pool where about one in fifty is. You are no longer checking whether this will work — you are checking how fast.',
  nextT:'Next: three relevant',nextB:()=>Math.min(nPos()/3,1),
  nextW:'At three, the model has enough to start learning from.',cta:'Keep going'},
 {id:'learn',n:'The model starts learning',s:'Three relevant — below this it guesses worse than plain search',
  test:()=>nPos()>=3, h:'It has started learning.',
  p:'Three relevant papers is the point where there is something to generalise from. Below three it does measurably worse than the plain search you started with, so it stayed switched off rather than showing you a number worth nothing.',
  nextT:'Next: it reorders your deck',nextB:()=>Math.min((nPos()/10+nNeg()/50)/2,1),
  nextW:'At ten relevant and fifty rejections it starts putting the likely ones first.',cta:'Carry on'},
 {id:'reorder',n:'It reorders your deck',s:'10 relevant and 50 not — likely ones come first',
  test:()=>nPos()>=10&&nNeg()>=50, h:'It just reordered 4,046 papers.',
  p:'Everything you have not seen has been put back in order, by a model built from your decisions in the last twenty minutes. The papers most likely to matter to you are now at the front of the deck.',
  p2:'You should feel this in the next few cards: the hit rate goes up. The original order is one click away if you want to compare.',
  big:'4,046',bigL:'papers re-sorted, in under a second',
  nextT:'Next: it can work without you',nextB:()=>Math.min(nPos()/20,1),
  nextW:'At twenty relevant it stops needing the deck at all — it can go through the whole pool and hand you a shortlist.',cta:'Show me'},
 {id:'auto',n:'It can sort the rest',s:'20 relevant and 50 not — it can work on its own',
  test:()=>nPos()>=20&&nNeg()>=50, h:'It can take the rest from here.',
  p:'You have given it enough to work on its own. It can go through the four thousand papers you have not seen and tell you which ones it thinks you would keep. You should still spot-check it — but you no longer have to read them all.',
  nextT:null,cta:'Let it sort the rest'}
];
function checkGates(){ const out=[];
  GATES.forEach(g=>{ if(!S.got.has(g.id)&&g.test()){ S.got.add(g.id); out.push(g) } }); return out }

/* ═══════════════════════════════════════════════════════════════════════
   SORT
   ═══════════════════════════════════════════════════════════════════════ */
function curCard(){ return S.deck[S.i] }
function paperAt(){ const c=curCard(); return c&&c.k==='paper'? PAPERS[c.id] : null }

function draw(){
  const c=curCard(), slot=document.getElementById('slot'), acts=document.getElementById('acts');
  if(!c){ drawEnd(); return }
  if(c.k==='milestone'){ drawMile(c.gate); return }
  if(c.k==='suggestion'){ drawSug(c); return }
  if(c.k==='check'){ drawChk(c); return }
  acts.style.display='flex';
  const p=PAPERS[c.id], sent=p.ab.split('|'), hasCue=p.cue.length>0;
  slot.innerHTML=`<div class="card">
    <div class="meta"><span>${p.v}</span><span>·</span><span>${p.y}</span>
      ${p.tr?`<span class="badge tr tip" data-tip="The original is in ${p.tr}. Title and abstract were translated so you can judge it — the original is one click away.">TRANSLATED · ${p.tr.toUpperCase()}</span>`:''}</div>
    <h2>${p.t}</h2><div class="auth">${p.a}</div>
    <div class="abs${S.lens&&hasCue?' lens':''}">${sent.map((s,i)=>
      `<span class="${p.cue.includes(i)?'cue':'dim'}">${s}</span>`).join(' ')}</div>
    ${S.lens&&!hasCue?`<div class="nofocus"><b>Nothing stood out in this one.</b> Focus lights the
      sentences that usually carry the decision, and about one abstract in ten does not have any —
      often the interesting ones. Read it all.</div>`:''}
    <div class="cardfoot">
      <button class="tgl ${S.lens?'on':''}" onclick="S.lens=!S.lens;draw()">
        <span class="sw"><i></i></span>
        <span class="tip" data-tip="Two or three sentences carry the decision in about nine abstracts in ten. This shades the rest — it never hides them.">Focus</span></button>
      <span style="flex:1"></span>
      <button class="linkbtn" onclick="helpMe()">Help me decide</button>
    </div>
    <div id="helperSlot"></div><div id="revealSlot"></div></div>`;
  document.getElementById('dh1').textContent=`${S.i+1} of ${S.deck.length} in this batch`;
  document.getElementById('dh2').innerHTML = seen()>7
    ? `<span class="tip" data-tip="A tenth of the cards take more than half the time. They are slow because the boundary is still being decided, not because you are.">the slow ones are normal</span>` : '';
}
function helpMe(){
  document.getElementById('helperSlot').innerHTML=`<div class="helper"><div class="top">
    <span class="t">Ask a model whether this fits what you described.</span>
    <select id="hm"><option>Claude Sonnet 4.6</option><option>Nemotron 3 Super 120B</option>
      <option>Gemma 4 31B</option><option>Qwen3 32B</option></select>
    <button class="btn ghost sm" onclick="runHelp()">Ask</button></div><div id="hmOut"></div></div>`;
}
function runHelp(){
  const p=paperAt(), sel=document.getElementById('hm'), name=sel.options[sel.selectedIndex].text;
  document.getElementById('hmOut').innerHTML='<div class="out" style="color:var(--ink3)">Reading it…</div>';
  setTimeout(()=>{ const fits=p.g==='pos';
    document.getElementById('hmOut').innerHTML=`<div class="out"><span class="who">${name.toUpperCase()} · AN OPINION</span>
      <span class="verdict">${fits?'Probably a fit.':'Probably not a fit.'}</span>
      ${fits?'It reports measured performance for a low-clinker system, which is what your description asks to be shown.'
            :'It uses your vocabulary but the subject is different — this is about operations, policy or a neighbouring field rather than the binder itself.'}
      <span class="caveat">Your decision is what trains the model; this is not. If you label after
      reading this, we record that you had help — otherwise your labels would quietly be a mix of two
      different things and nothing downstream could tell them apart.</span></div>`;
  },560);
}
function drawMile(g){
  document.getElementById('acts').style.display='none';
  const nb=g.nextB?Math.round(g.nextB()*100):0;
  /* "Stop for now" is gone from these (Warren): a card that exists to mark something
     you achieved should not offer leaving as one of two equal choices. Quitting is
     always available from the header. */
  document.getElementById('slot').innerHTML=`<div class="card mile">
    <div class="kicker"><span class="tick">✓</span>Unlocked</div>
    <h2>${g.h}</h2>
    ${g.big?`<div class="mileBig"><span class="bn">${g.big}</span><span class="bl">${g.bigL}</span></div>`:''}
    <p>${g.p}</p>${g.p2?`<p>${g.p2}</p>`:''}
    ${g.nextT?`<div class="mileNext"><div class="nt">${g.nextT}</div>
      <div class="nb">${g.nextW}</div>
      <div class="nbar"><i style="width:${nb}%"></i></div>
      <div class="nfoot">${nb}% of the way there</div></div>`:''}
    <div style="display:flex;gap:10px;align-items:center;margin-top:${g.nextT?'0':'20px'}">
      <button class="btn onGreen" onclick="${g.id==='auto'?"go('use')":'next()'}">${g.cta}</button>
      ${g.id==='auto'?'<button class="btn ghostGreen" onclick="next()">Keep sorting first</button>':''}</div></div>`;
  document.getElementById('dh1').textContent=''; document.getElementById('dh2').textContent='';
}
function drawSug(c){
  document.getElementById('acts').style.display='none';
  document.getElementById('slot').innerHTML=`<div class="card sug">
    <div class="kicker">A suggestion</div>
    <h2>Hide papers about “${c.term}”?</h2>
    <p>You have turned down <b>${c.n}</b> papers with this word in them and kept none.
      We can drop them out of what you are shown.</p>
    <div class="pricebox">
      <div class="pl"><span>Papers it would hide</span><b>${c.removes} of 4,120</b></div>
      <div class="pl"><span>Relevant ones you would lose</span><b>none of your ${nPos()}</b></div>
      <div class="pl"><span>Can you undo it?</span><b>Yes, any time — nothing is deleted</b></div></div>
    <div style="display:flex;gap:9px;align-items:center">
      <button class="btn onGold" style="flex:1;justify-content:center" onclick="sugNo(${c.idx})">No, keep showing them <kbd style="border-color:hsla(45,60%,94%,.4);color:hsl(45,60%,94%);background:none">N</kbd></button>
      <button class="btn ghostGold" onclick="sugYes(${c.idx})">Hide them <kbd style="border-color:hsla(38,55%,20%,.4)">Y</kbd></button>
      <button class="btn ghostGold" onclick="sugLater(${c.idx})">Later</button></div>
    <div class="foot">Saying no is the more useful answer more often than people expect — it tells us
      where the edge of your topic is, which nothing else here can.</div></div>`;
  document.getElementById('dh1').textContent=''; document.getElementById('dh2').textContent='';
}
function drawEnd(){
  document.getElementById('acts').style.display='none';
  const p=nPos(), mult=p&&seen()?Math.round((p/seen())/POOL_PREV):0;
  const left=DEALABLE.filter(x=>S.lab[x.id]===undefined).length;
  const needP=Math.max(0,20-p), needN=Math.max(0,50-nNeg()), gated=!needP&&!needN;
  document.getElementById('slot').innerHTML=`<div class="card mile">
    <div class="kicker"><span class="tick">${gated?'✓':'—'}</span>Batch ${S.batch} done</div>
    <h2>${p?`${p} relevant so far, from ${seen()} read.`:`Nothing relevant in ${seen()} yet.`}</h2>
    <p>${p?`This pool is about 2% relevant, so you are finding them roughly <b>${mult}×</b> faster than
        reading in order — about ${Math.round(p/POOL_PREV).toLocaleString()} papers you did not have to open.`
      :`At this pool's rate that can happen. It has now gone on long enough to be worth saying: the
        ranking may not be working for this description, and that is the case where sorting pays off
        most, not least.`}</p>
    ${gated?'':`<div class="mileNext"><div class="nt">To the last unlock</div>
      <div class="nb">${needP?`<b>${needP}</b> more relevant`:''}${needP&&needN?' and ':''}${needN?`<b>${needN}</b> more rejections`:''} — about ${Math.ceil(Math.max(needP/0.3,needN/0.7))} more cards at your current rate.</div>
      <div class="nbar"><i style="width:${Math.round(Math.min((p/20+nNeg()/50)/2,1)*100)}%"></i></div>
      <div class="nfoot">${left} papers left in the pool that we can deal</div></div>`}
    <div style="display:flex;gap:10px;margin-top:${gated?'20px':'0'}">
      ${gated?`<button class="btn onGreen" onclick="go('use')">Let it sort the rest</button>`
        : left? `<button class="btn onGreen" onclick="nextBatch()">Deal another ${Math.min(BATCH,left)}</button>`
        : `<button class="btn onGreen" onclick="go('use')">See where this got to</button>`}
      <button class="btn ghostGreen" onclick="go('home')">Stop for now</button></div></div>`;
  document.getElementById('dh1').textContent=''; document.getElementById('dh2').textContent='';
}
function nextBatch(){ const b=dealBatch(); if(!b.length){drawEnd();return}
  S.deck=S.deck.concat(b); S.i=S.deck.length-b.length; draw(); paint() }

function reveal(p,v){
  const rank=S.i+1, dec=Math.max(2,Math.round(S.deck.length/10));
  const surp=(v==='pos'&&rank>S.deck.length-dec)||(v==='neg'&&rank<=dec-1);
  const s=document.getElementById('revealSlot'); if(!s) return;
  s.innerHTML=`<div class="reveal ${surp?'surp':''}">${p.arm==='control'
    ? `Dealt at random rather than by rank — a few are, so we can tell you the pool's real rate. You cannot tell which, on purpose.`
    : `We had this at <b>#${rank}</b> in this batch.`}${
    surp?` <b>${v==='pos'?'We had it near the bottom — worth knowing.':'We had it near the top and you disagreed.'}</b>`:''}</div>`;
}
function label(v){
  if(S.pending) return;                       // rapid-fire guard: see v2 commit
  const p=paperAt(); if(!p) return;
  S.pending=true; S.lab[p.id]=v; S.hist.push(p.id);
  paint(); reveal(p,v);
  /* One interruption at a time, in order of what it costs to ignore: a milestone
     is earned, a check is a question about work already done, a filter offer is
     the cheapest of the three. */
  const gates=checkGates(), chk=maybeCheck(), sug=maybeSuggest(v);
  setTimeout(()=>{
    if(gates.length) S.deck.splice(S.i+1,0,{k:'milestone',gate:gates[0]});
    else if(chk) S.deck.splice(S.i+1,0,chk);
    else if(sug) S.deck.splice(S.i+1,0,sug);
    next();
  },600);
}
function next(){ S.pending=false; S.i++; draw(); paint() }
function undo(){ S.pending=false; const id=S.hist.pop(); if(id===undefined)return; delete S.lab[id];
  S.i=Math.max(0,S.i-1); while(S.deck[S.i]&&S.deck[S.i].k!=='paper'&&S.i>0)S.i--; draw(); paint() }

function candidates(){
  const kept=new Set();
  Object.keys(S.lab).filter(i=>S.lab[i]==='pos').forEach(i=>tokOf(i).forEach(w=>kept.add(w)));
  const mine=new Set(compiled().toLowerCase().match(/[a-z][a-z-]{3,}/g)||[]);
  const c={};
  Object.keys(S.lab).filter(i=>S.lab[i]==='neg').forEach(i=>tokOf(i).forEach(w=>{
    if(kept.has(w)||STOP.has(w)||FILLER.has(w)||SCAFFOLD.has(w)||mine.has(w)||POOLDF.has(w))return;
    c[w]=(c[w]||0)+1 }));
  return Object.entries(c).filter(([,n])=>n>=3).sort((a,b)=>b[1]-a[1]);
}
/* Two limits, and they are the difference between an instrument and a nag.
   A first pass offered six suggestions in nine cards — `plants`, `plant`,
   `model`, `sizes`, `existing` — i.e. it interrupted constantly and spent the
   analyst's rejections on noise and near-duplicates. The scarce act here is
   rejection (the failing grader accepted 47 of 50; the oracle's selectivity was
   worth +0.27 recall@10%), so a candidate has to clear a real bar and the
   interruption has to be rare enough to still read as a signal. */
const SUG_GAP=12;         // at most one suggestion per twelve decisions
const SUG_SHARE=0.25;     // and only if it covers a quarter of what you rejected

/* ═══════════════════════════════════════════════════════════════════════
   CONFLICTS · four detectors, one ladder, and one thing they may never do

   The problem this answers: a use case is not fully known when it is written.
   The analyst gets clearer as they read, sometimes changes their mind, sometimes
   mislabels, and sometimes drifts. All four look identical from the outside — two
   things disagree — and they need completely different responses.

   ── THE STRUCTURE ────────────────────────────────────────────────────────
                     LABEL ↔ SPEC                  LABEL ↔ LABEL
     spec behind     ① thin spec (an OFFER)        ④ drift (a VIEW)
     labels wrong    ② spec contradicted           ③ confusion

   The asymmetry that drives every decision below: in ①②③ the system can tell
   which side is wrong. In ④ it structurally cannot — early labels wrong and late
   labels wrong produce identical evidence — so ④ can never be a card with two
   buttons on it. It is deliberately NOT built here; the drift instrument has to
   be measured on real labels first, and a screen for an effect nobody has
   measured is the ADR 0009 failure.

   ── WHAT IS BUILT ────────────────────────────────────────────────────────
     ② spec-kept        you kept papers matching your own near-miss term
     ② spec-rejected    you rejected papers that satisfy your whole spec
     ⑤ near-duplicate   the same work twice, labelled both ways
     ⑥ seed-rejected    you turned down a paper you supplied as known-relevant
     ③ neighbour        two papers the model cannot tell apart, labelled apart
   ②⑤⑥ are DETERMINISTIC — no model, no embedding, explainable in one sentence,
   and they work from the second label. ③ is model-inferred and therefore gated:
   below 20 relevant the classifier is measurably worse than the plain ranker, so
   an inferred conflict would be least reliable exactly when the analyst is most
   confused.

   ── THE ONE THING THEY MAY NEVER DO ──────────────────────────────────────
   Tell a domain expert they are wrong. S-CA measured two views of the same paper
   disagreeing at κ 0.52–0.69, and their union scored WORSE than either alone —
   textual proximity is not decision-equivalence. Two abstracts can be nearly
   identical and differ on the one thing that decides it. So every card shows TWO
   OF THE ANALYST'S OWN DECISIONS and asks, and "this is deliberate" is a
   first-class answer that is RECORDED — because a distinction the model cannot
   see is precisely the thing the description is missing. A false positive then
   produces a spec refinement, which is the only design where the detector being
   wrong is still worth something.

   ⑤ is the single exception: the same paper twice cannot be a deliberate
   distinction, so it is the only detector allowed to be assertive.
   ═══════════════════════════════════════════════════════════════════════ */
const CONF_NEAR_MIN=0.62;   /* cosine in the ranker's own space */
const CONF_DUP_MIN=0.55;    /* title-token Jaccard */
const CHK_GAP=14;           /* at most one in-stack check per fourteen decisions */

/* ── matching a spec phrase against a paper ───────────────────────────────
   The first version required EVERY content word of the phrase, and it made this
   whole detector inert: "limestone blends" needs both `limestone` and `blend`,
   and real abstracts write "limestone calcined clay" or "blended binder", so
   across the entire fixture only TWO papers could satisfy a systems term and an
   outcome term at once. A detector that cannot fire is worse than none, because
   its silence reads as a clean bill of health.

   So a phrase matches on its most DISTINCTIVE word — the rarest one in the pool —
   provided that word is genuinely discriminating (in under a quarter of papers).
   "dental cements" keys on `dental`, "plant economics" on `economics`, "calcined
   clay" on `calcined`. Where the rare word is still common, the conjunction is
   required, because one common word is not evidence of anything. Erring loose in
   the first case and tight in the second is deliberate: this detector's output is
   shown to a domain expert as a question about their own work, and a false
   accusation costs far more than a missed one. */
const POOLDF_N=(()=>{ const d={}; PAPERS.forEach(p=>tokOf(p.id).forEach(w=>d[w]=(d[w]||0)+1)); return d })();
/* THE SAME TOKENISER ON BOTH SIDES, and this cost an hour. `tokOf` keeps words of
   five characters or more; the phrase splitter kept four. So "slag" and "clay"
   entered the phrase's word list, were absent from the pool's index, scored a
   document frequency of ZERO — which made them the RAREST word and therefore the
   chosen key — and then matched nothing, ever. Half the detector was switched off
   and its silence read as "your labels are consistent".
   Two comparisons of the same text have to be built from the same tokeniser, and
   a word the index has never heard of is unusable rather than maximally rare. */
const phraseWords=t=>new Set((String(t).toLowerCase().match(/[a-z][a-z-]{4,}/g)||[])
  .map(stem).filter(w=>!STOP.has(w) && (POOLDF_N[w]||0)>0));
function phraseKey(phrase){
  const w=[...phraseWords(phrase)];
  if(!w.length) return null;          /* nothing in this phrase exists in the pool */
  const rare=w.slice().sort((a,b)=>POOLDF_N[a]-POOLDF_N[b])[0];
  return (POOLDF_N[rare]/PAPERS.length < 0.25) ? [rare] : w;
}
const titleTok=id=>new Set((PAPERS[id].t.toLowerCase().match(/[a-z][a-z-]{4,}/g)||[]).map(stem));
const paperHas=(id,phrase)=>{ const k=phraseKey(phrase); if(!k) return false;
  const toks=tokOf(id); return k.every(x=>toks.has(x)) };
/* A STRICTER TEST, used only for near-misses. "plant economics" keys on
   `economics`, and a materials paper calling its result "economically attractive"
   then gets flagged for a paper that is nothing to do with plant economics. The
   near-miss detector is the one whose output most looks like an accusation, so it
   demands the key word IN THE TITLE — where topicality actually lives — or the
   full phrase in the body. Loose enough to catch a dental-cement paper, tight
   enough not to catch an adjective. */
const paperHasStrong=(id,phrase)=>{
  const k=phraseKey(phrase); if(!k) return false;
  const t=titleTok(id); if(k.every(x=>t.has(x))) return true;
  const all=[...phraseWords(phrase)], toks=tokOf(id);
  return all.length>1 && all.every(x=>toks.has(x));
};
const titleToks=id=>new Set((PAPERS[id].t.toLowerCase().match(/[a-z][a-z-]{3,}/g)||[])
  .map(stem).filter(w=>!STOP.has(w)));
function jaccard(a,b){ let hit=0; a.forEach(w=>{ if(b.has(w)) hit++ });
  return hit/(a.size+b.size-hit||1) }

/* every labelled paper's distance from the current cut-off, so a contradiction
   ON the boundary outranks one buried deep inside either cloud — same idea as
   SUG_SHARE: the budget goes to what actually costs the model something */
function boundaryRank(){
  const pos=Object.keys(S.lab).filter(i=>S.lab[i]==='pos').length;
  if(pos<3) return {};
  const sc=scoresFor('you'), pts=curveFor(sc), F=fPoints(pts);
  const at=pts[F.f1], cut=sc[at.order[at.n-1]];
  const out={}; Object.keys(S.lab).forEach(i=>out[i]=Math.abs(sc[i]-cut));
  return out;
}
function conflicts(){
  const out=[], lab=S.lab;
  const ids=Object.keys(lab).map(Number);
  const P_=ids.filter(i=>lab[i]==='pos'), N_=ids.filter(i=>lab[i]==='neg');
  const rank=boundaryRank();

  /* ── ⑤ near-duplicate, labelled both ways ─────────────────────────── */
  for(let a=0;a<ids.length;a++) for(let b=a+1;b<ids.length;b++){
    const i=ids[a], j=ids[b];
    if(lab[i]===lab[j]) continue;
    if(jaccard(titleToks(i),titleToks(j))<CONF_DUP_MIN) continue;
    out.push({key:`dup:${Math.min(i,j)}:${Math.max(i,j)}`,kind:'dup',sev:'conflict',
      pair:[i,j],assertive:true,
      hd:'The same work, labelled both ways',
      why:`These are the same study — one is the preprint. They cannot both be right, and whichever
        answer is wrong is teaching the model the opposite of what you mean.`,
      cost:'A contradictory pair pulls the boundary in two directions at once.'});
  }

  /* ── ⑥ a seed you turned down ──────────────────────────────────────── */
  N_.filter(i=>PAPERS[i].seed).forEach(i=>{
    out.push({key:`seed:${i}`,kind:'seed',sev:'check',one:i,
      hd:'You turned down a paper you supplied',
      why:`This arrived marked relevant because you named it before the search ran. Either the
        list was broader than the use case, or the use case has narrowed since you wrote it.`,
      cost:'Seeds set the initial direction, so one you would now reject aimed the whole pool.'});
  });

  /* ── ② you kept papers matching your own near-miss ──────────────────── */
  (S.ans.near||[]).forEach(term=>{
    const hits=P_.filter(i=>paperHasStrong(i,term));
    if(!hits.length) return;
    out.push({key:`spk:${term}`,kind:'spec-kept',sev:hits.length>=3?'conflict':'check',
      many:hits,term,
      hd:`You said “${term}” was not what you meant`,
      why:`It is listed as a near-miss in your description, and you have kept ${hits.length}
        paper${hits.length>1?'s':''} matching it. Near-misses are the strongest thing in the
        description — this one is now working against you.`,
      cost:'A near-miss term pushes matching papers down the ranking, including these.'});
  });

  /* ── ② you rejected papers that satisfy the whole spec ──────────────── */
  const sysT=(S.ans.systems||[]), outT=(S.ans.outcome||[]);
  if(sysT.length&&outT.length){
    const hits=N_.filter(i=>sysT.some(t=>paperHas(i,t))&&outT.some(t=>paperHas(i,t)));
    if(hits.length>=2) out.push({key:'sprj',kind:'spec-rejected',
      sev:hits.length>=4?'conflict':'check',many:hits,
      hd:'Papers that match your description, turned down',
      why:`${hits.length} papers name one of your systems AND one of the properties you said had to
        be shown, and you rejected them all. Something you are deciding on is not written down.`,
      cost:'The ranker cannot apply a rule it was never given, so it keeps offering these.'});
  }

  /* ── ③ two papers the model cannot separate, separated by you ──────── */
  if(P_.length>=20&&N_.length>=50){
    const seen=new Set();
    P_.forEach(i=>N_.forEach(j=>{
      const c=cosv(VEC.rows[i],VEC.rows[j]);
      if(c<CONF_NEAR_MIN) return;
      const k=`nbr:${Math.min(i,j)}:${Math.max(i,j)}`;
      if(seen.has(k)) return; seen.add(k);
      out.push({key:k,kind:'near',sev:'check',pair:[i,j],sim:c,
        hd:'Two that read alike, decided differently',
        why:`The model cannot tell these apart — they are ${(c*100).toFixed(0)}% alike in the space
          it reads. If the difference is real, it is something you know and the description does not
          say.`,
        cost:'An unexplained split near the boundary is where the model loses the most accuracy.'});
    }));
  }

  /* Drop anything already answered, then order by what it costs the model.
     SEVERITY IS THE PRIMARY KEY, and it has to be: the first version multiplied a
     boundary-proximity score by 1000 for an outright error, and a paper that
     happened to sit within 1e-4 of the cut-off scored ~10,000 on proximity alone
     and outranked it. A weight big enough to dominate an unbounded term does not
     exist, so the comparison is lexicographic instead — error, then cost, then
     proximity — and proximity is bounded so it can only ever be a tiebreak. */
  return out.filter(c=>!S.cRes[c.key])
    .map(c=>{ const t=c.pair||c.many||[c.one];
      c.near=Math.max(...t.map(i=>1/(0.02+(rank[i]??1))));
      c.tier=c.assertive?2:(c.sev==='conflict'?1:0);
      return c })
    .sort((a,b)=> b.tier-a.tier || b.near-a.near);
}
const openConflicts=()=>conflicts().filter(c=>c.sev==='conflict');
/* Only the ones that reach `conflict` count against the model. A single check is
   a question; a repeated pattern is a cost, and only the second one belongs on a
   gate. */
function resolveConf(key,how,note){
  S.cRes[key]=how;
  if(how==='deliberate'){
    const c=(S.cAll||[]).find(x=>x.key===key);
    S.notes.push({key,hd:c?c.hd:key,note:note||'',at:seen()});
  }
  draw(); paint();
}
function flipLabel(id){
  S.lab[id]=S.lab[id]==='pos'?'neg':'pos';
  /* the flip is a label like any other: append-only, latest wins, and it is the
     analyst's own — which is exactly why "undo is go back and relabel" works */
  paint();
}
/* the in-stack CHECK: cheap, forward-looking, once, never repeated */
function maybeCheck(){
  if(seen()<12) return null;
  if(seen()-(S.lastChk||0)<CHK_GAP) return null;
  const c=conflicts().find(x=>!S.cSeen[x.key]);
  if(!c) return null;
  S.lastChk=seen(); S.cSeen[c.key]=true;
  return {k:'check',conf:c,idx:S.i+1};
}


/* ── rendering a conflict ─────────────────────────────────────────────────
   Every one of these shows the analyst TWO OF THEIR OWN DECISIONS and asks. It
   never renders a verdict, it never says "wrong", and the row for each paper
   carries the answer the analyst actually gave — because the whole interaction
   depends on them recognising their own work rather than being told about it. */
function confRows(c,hard){
  const list=c.pair||c.many||[c.one];
  return `<div class="cfPair">${list.slice(0,4).map(i=>`
    <div class="cfRow"><span><span class="ct">${PAPERS[i].t}</span>
      <span class="cm">${PAPERS[i].v} · ${PAPERS[i].y}${PAPERS[i].seed?' · you supplied this':''}</span></span>
      <span style="display:flex;gap:9px;align-items:center">
        <span class="cfV ${S.lab[i]==='pos'?'y':'n'}">${S.lab[i]==='pos'?'you kept it':'you turned it down'}</span>
        <button class="btn sm ${hard?'onRust':'ghostRust'}" onclick="flipLabel(${i});draw();paint()">
          Change to ${S.lab[i]==='pos'?'not relevant':'relevant'}</button></span></div>`).join('')}
    ${list.length>4?`<div class="cfRow"><span class="ct" style="font-weight:400;opacity:.7">…and
      ${list.length-4} more like this</span><span></span></div>`:''}</div>`;
}
/* the in-stack CHECK card */
function drawChk(card){
  const c=card.conf, hard=!!c.assertive;
  document.getElementById('acts').style.display='none';
  document.getElementById('slot').innerHTML=`<div class="card chk ${hard?'hard':''}">
    <div class="kicker">${icon(hard?'warn':'compare')} ${hard?'These cannot both be right'
      :'Worth a second look'}</div>
    <h2>${c.hd}</h2>
    <p>${c.why}</p>
    ${confRows(c,hard)}
    ${S.cWhy===c.key? `<div class="cfWhy">
        <textarea id="cfNote" rows="2" placeholder="What is the difference? e.g. “these measure it on real structures, the others model it”"></textarea>
        <div class="cfAct" style="margin-top:10px">
          <button class="btn ${hard?'onRust':'onRust'}" onclick="saveDeliberate('${c.key}',${card.idx})">Record it and move on</button>
          <button class="btn ghostRust" onclick="S.cWhy=null;draw()">Cancel</button></div>
        <div class="cfNote">This goes on the description as a distinction the model could not see —
          which is the thing it was missing, not a note about you.</div></div>`
      : `<div class="cfAct">
        ${hard? '' : `<button class="btn onRust" onclick="S.cWhy='${c.key}';draw()">Both are right — say why</button>`}
        <button class="btn ${hard?'onRust':'ghostRust'}" onclick="chkLater(${card.idx})">${
          hard?'I will fix this now':'Later'}</button>
        ${hard? '' : `<button class="btn ghostRust" onclick="chkNever('${c.key}',${card.idx})">Stop asking</button>`}
      </div>
      <div class="cfNote">${hard
        ? 'The same study twice is the one case here that is not a judgement call. Change one of them above.'
        : c.cost+' If the split is deliberate, saying why once stops it being asked again — and it goes into your description.'}</div>`}
  </div>`;
  document.getElementById('dh1').textContent=''; document.getElementById('dh2').textContent='';
}
function saveDeliberate(key,idx){
  const el=document.getElementById('cfNote');
  S.cAll=conflicts();
  resolveConf(key,'deliberate',el?el.value.trim():'');
  S.cWhy=null; dropCard(idx);
}
function chkNever(key,idx){ S.cAll=conflicts(); resolveConf(key,'dismissed'); dropCard(idx) }
/* "Later" does NOT go to Set aside. Set aside is a deferral bin for offers; an
   unresolved conflict is a cost, and putting the two in one list flattens two
   different urgencies into one the user can ignore. It stays in the queue. */
function chkLater(idx){ dropCard(idx) }
function dropCard(idx){ if(S.deck[idx]) S.deck.splice(idx,1); draw(); paint() }
function openConf(){ document.getElementById('confModal').classList.add('on'); renderConf() }
function renderConf(){
  const all=conflicts(); S.cAll=all;
  const hard=all.filter(c=>c.sev==='conflict'), soft=all.filter(c=>c.sev==='check');
  const item=(c)=>`<div class="cfItem ${c.sev==='conflict'?'hardItem':''}">
    <div class="ih">${icon(c.assertive?'warn':c.sev==='conflict'?'warn':'compare')} ${c.hd}</div>
    <div class="iw">${c.why}</div>
    ${confRows(c,false)}
    <div class="cfAct">
      ${c.assertive?'':`<button class="btn ghost sm" onclick="S.cAll=conflicts();resolveConf('${c.key}','deliberate','');renderConf()">
        Deliberate — record it</button>`}
      ${c.assertive?'':`<button class="btn ghost sm" onclick="S.cAll=conflicts();resolveConf('${c.key}','dismissed');renderConf()">
        Stop asking</button>`}
      <span style="font-size:11.5px;color:var(--ink3);margin-left:4px">${c.cost}</span></div></div>`;
  document.getElementById('confBody').innerHTML=`
    ${hard.length?`<div class="cfSec">Costing your model — ${hard.length}</div>${hard.map(item).join('')}`:''}
    ${soft.length?`<div class="cfSec">Worth a look — ${soft.length}</div>${soft.map(item).join('')}`:''}
    ${!all.length?`<div class="cfItem"><div class="ih">${icon('circleCheck')} Nothing is contradicting
      itself</div><div class="iw">Your labels agree with each other and with your description, on
      everything that can be checked without guessing. This gets re-checked after every decision.</div></div>`:''}
    ${S.notes.length?`<div class="cfSec">Distinctions you recorded — ${S.notes.length}</div>
      ${S.notes.map(n=>`<div class="cfDone">${icon('check')} <b>${n.hd}</b>
        ${n.note?`<span class="nt">“${n.note}”</span>`:'<span class="nt">no reason given</span>'}
        <span class="nt">recorded at ${n.at} labels — these are the candidates for the next version
        of your description, because each one is something you can see and the model cannot.</span></div>`).join('')}`:''}`;
  document.getElementById('confFoot').innerHTML = hard.length
    ? `${hard.length} unresolved ${hard.length===1?'conflict is':'conflicts are'} counted against your
       model quality until you answer ${hard.length===1?'it':'them'}.`
    : 'Nothing here is counted against your model.';
}

function maybeSuggest(v){
  if(v!=='neg'||seen()<25) return null;
  if(seen()-(S.lastSug||0)<SUG_GAP) return null;
  const nNegSeen=Object.values(S.lab).filter(x=>x==='neg').length;
  const c=candidates().filter(([w,n])=>!S.seenTerms.has(w)&&n/nNegSeen>=SUG_SHARE);
  if(!c.length) return null;
  S.lastSug=seen(); S.seenTerms.add(c[0][0]);
  return {k:'suggestion',term:c[0][0],n:c[0][1],removes:Math.round(40+c[0][1]*37),idx:S.i+1};
}
function sugNo(i){ S.declined++; S.deck.splice(i,1); draw(); paint() }
function sugYes(i){ const c=S.deck[i]; S.excluded++; S.filtered+=c.removes; S.deck.splice(i,1); draw(); paint() }
function sugLater(i){ S.aside.push(S.deck[i]); S.deck.splice(i,1); draw(); paint() }
function reopen(t){ const c=S.aside.find(x=>x.term===t); if(!c)return;
  S.aside=S.aside.filter(x=>x.term!==t); c.idx=S.i; S.deck.splice(S.i,0,c); draw(); paint() }

/* the expanded view */
function toggleMap(){
  const on=!document.getElementById('mapModal').classList.contains('on');
  if(on) renderMapModal();
  document.getElementById('mapModal').classList.toggle('on',on);
}
function renderMapModal(){
  const p=nPos(), n=nNeg();
  document.getElementById('mapBody').innerHTML=`
    <div class="mapStage">${poolMap(true)}<div class="mapTip" id="mapTip"></div></div>
    <div class="mapLegend">
      <span><i class="lg pos"></i>You kept — ${p}</span>
      <span><i class="lg neg"></i>You turned down — ${n}</span>
      <span><i class="lg none"></i>Not yet seen</span>
      <span><i class="lg ring"></i>Where it expects relevant work, at three operating points
        <span class="im right" data-tip="Same model, three cut-offs, and each one is a defined setting rather than a mood: TIGHT is the least reading that still finds half of what matters, BALANCED is the F1 optimum, LOOSE is the least reading that reaches 95% of it. There is no setting that does neither — that trade is the whole decision. Pick one in Tune your model.">i</span></span>
    </div>
    <div class="mapCols">
      <div><div class="pt">Inclusions <span class="im" data-tip="Words running through what you kept and absent from what you rejected.">i</span></div>
        <div class="wcloud yes">${boundaryWords('yes').map(w=>`<span>${w}</span>`).join('')||'<span class="none">not yet</span>'}</div></div>
      <div><div class="pt">Exclusions <span class="im" data-tip="Words common to what you turned down and absent from what you kept.">i</span></div>
        <div class="wcloud no">${boundaryWords('no').map(w=>`<span>${w}</span>`).join('')||'<span class="none">not yet</span>'}</div></div>
    </div>
    <div class="mapNote">Positions are a flattening of 384 dimensions onto a page, so read the
      grouping and not the distances. Hover any point to see which paper it is.</div>`;
}
function boundaryWords(dir){
  const pos=Object.keys(S.lab).filter(i=>S.lab[i]==='pos'), neg=Object.keys(S.lab).filter(i=>S.lab[i]==='neg');
  const mine=new Set(compiled().toLowerCase().match(/[a-z][a-z-]{3,}/g)||[]);
  const A=dir==='yes'?pos:neg, B=dir==='yes'?neg:pos;
  if(!A.length) return [];
  const bWords=new Set(); B.forEach(i=>tokOf(i).forEach(w=>bWords.add(w)));
  const c={};
  A.forEach(i=>tokOf(i).forEach(w=>{
    if(bWords.has(w)||STOP.has(w)||FILLER.has(w)||SCAFFOLD.has(w)||POOLDF.has(w))return;
    if(dir==='yes'&&mine.has(w))return;     // words you already typed are not news
    c[w]=(c[w]||0)+1 }));
  /* A word has to be CHARACTERISTIC of the set, not merely present in two of its
     members. At the flat n>=2 bar this surfaced `below`, `roughly` and `relative` —
     true of the corpus, useless as a description of a boundary, and the fastest way
     to make the panel look like it is guessing. */
  /* Asymmetric on purpose. The keeps share a topic by construction, so a high bar
     works. The rejects are incoherent by construction — turned down for many
     different reasons — so the same bar leaves the list permanently empty. At 15
     rejections spanning dental cements, logistics, procurement and heritage mortars,
     no word appears in six of them, and that is a fact about the boundary rather
     than a failure of the panel. */
  const bar=dir==='yes'? Math.max(2,Math.ceil(A.length*0.35)) : Math.max(2,Math.ceil(A.length*0.2));
  return Object.entries(c).filter(([,n])=>n>=bar).sort((a,b)=>b[1]-a[1]).slice(0,6).map(e=>e[0]);
}
function langBlock(){
  const yes=boundaryWords('yes'), no=boundaryWords('no');
  const nP=Object.values(S.lab).filter(v=>v==='pos').length;
  const nN=Object.values(S.lab).filter(v=>v==='neg').length;
  /* Two different empty states, because they mean different things: "not enough
     yet" and "enough, and they have nothing in common" are not the same fact. */
  const empty=(n,kind)=> n<3
    ? `<span class="none">appears once you have ${kind==='yes'?'kept':'turned down'} a few</span>`
    : `<span class="none">${kind==='yes'?'nothing shared across your keeps yet'
        :`no word runs through your ${n} rejections — they are varied, which is normal`}</span>`;
  return `<div class="lang">
    <div class="lh"><span>Inclusions
      <span class="im" data-tip="Words that run through the papers you kept and not the ones you rejected. Learned from your decisions — you never typed these, and they are not the words from your description.">i</span></span>
      <em>${yes.length||''}</em></div>
    <div class="wcloud yes">${yes.length? yes.map(w=>`<span>${w}</span>`).join('') : empty(nP,'yes')}</div>
    <div class="lh"><span>Exclusions
      <span class="im" data-tip="The mirror: words common to what you turned down and absent from what you kept. Often empty for longer, because papers get rejected for many different reasons.">i</span></span>
      <em>${no.length||''}</em></div>
    <div class="wcloud no">${no.length? no.map(w=>`<span>${w}</span>`).join('') : empty(nN,'no')}</div>
    <button class="langMore" onclick="toggleMap()">Explore these →</button></div>`;
}

/* ── rails ── */
function gateMarks(el,marks,val,max){
  el.querySelectorAll('.gk').forEach(x=>x.remove());
  marks.forEach(m=>{ const d=document.createElement('span');
    d.className='gk'+(val>=m?' hit':''); d.style.left=(m/max*100)+'%'; d.dataset.l=m; el.appendChild(d) });
}
function paint(){
  const p=nPos(), n=nNeg();
  /* THE GATE FAILS BACKWARDS. The terminus is a gate in positives, so the honest
     reading of "20 relevant, 4 of them contradicted" is not 20 — and the caption
     that already names the next rung on the classifier ladder is the right place
     to say so. This is the persistence the interruption cards deliberately do not
     provide: un-ignorable, counted, and costing nothing in flow. */
  const cAll=seen()?conflicts():[]; S.cAll=cAll;
  const cHard=cAll.filter(c=>c.sev==='conflict'), cSoft=cAll.filter(c=>c.sev==='check');
  const contradicted=new Set(); cHard.forEach(c=>(c.pair||c.many||[c.one]).forEach(i=>contradicted.add(i)));
  const cPos=[...contradicted].filter(i=>S.lab[i]==='pos').length;

  document.getElementById('pv').innerHTML=p;
  document.getElementById('nv').textContent=n;
  document.getElementById('pf').style.width=Math.min(100,p/22*100)+'%';
  document.getElementById('nf').style.width=Math.min(100,n/60*100)+'%';
  gateMarks(document.getElementById('ptr'),[3,10,20],p,22);
  gateMarks(document.getElementById('ntr'),[50],n,60);
  /* the standing prevalence sentence is gone (Warren) — it repeated on every card
     and the same fact now lives once, in Use case stats. */
  document.getElementById('railnote').innerHTML = !seen() ? ''
    : (cPos? `<span style="color:var(--rust)">${icon('warn')} <b>${p} relevant, ${cPos}
        contradicted.</b></span> Until those are answered the model is being taught two things at
        once, so treat its quality number as ${cPos>2?'unreliable':'provisional'}.`
      : p? `Reading in order, ${p} relevant would have taken about <b>${Math.round(p/POOL_PREV).toLocaleString()}</b> papers.`
       : 'Nothing relevant yet. At this pool\'s rate that is not unusual this early.');

  document.getElementById('ucsHd').innerHTML=icon('bars')+' Use case stats';
  document.getElementById('ucStats').innerHTML=`
    <div class="usr"><span class="k">Papers found</span><span class="v">4,120</span></div>
    <div class="usr"><span class="k">Relevant, estimated
      <span class="im" data-tip="Estimated from the papers we deal at random rather than by rank — the only way to know the pool's true rate. It appears once a couple of those have been seen.">i</span></span>
      <span class="v">${ctrlSeen()>=2?'2.1%':'not yet'}</span></div>
    <div class="usr"><span class="k">Waiting on abstracts
      <span class="im" data-tip="A card with nothing to read is not a decision, it is a gap. These are fetched in the background and dealt when they arrive.">i</span></span>
      <span class="v">212</span></div>
    <div class="usr"><span class="k">Hidden by a filter</span><span class="v">${S.filtered?S.filtered.toLocaleString():'none'}</span></div>`;

  document.getElementById('unlq').textContent=`${S.got.size} of 4`;
  document.getElementById('unlocks').innerHTML=GATES.map(g=>{
    const got=S.got.has(g.id), nx=!got&&GATES.filter(x=>!S.got.has(x.id))[0]?.id===g.id;
    return `<div class="unl ${got?'got':''} ${nx?'next':''}"><span class="ic">${got?'✓':(nx?'○':'')}</span>
      <div><span class="n">${g.n}</span><span class="s">${g.s}</span></div></div>`}).join('');

  /* WHAT THE MODEL HAS — a map of the pool plus the language of the boundary. */
  const conf = p<3?0 : p<10?1 : (p<20||n<50)?2:3;
  document.getElementById('build').innerHTML=`
    <div class="bhead"><span class="t">${['Nothing to learn from yet','Learning','Working','Trained'][conf]}
      <span class="im" data-tip="Every paper in the pool, placed by a two-component PCA of the same text the ranker reads — so the grouping is discovered, not decorative. Positions are approximate: any projection onto a page loses most of the dimensions it started with, so read which papers sit together and never the distance between them. The shading is where the model currently expects relevant work, at three cut-offs.">i</span></span>
      <span class="v">${['stage 1 of 4','stage 2 of 4','stage 3 of 4','stage 4 of 4'][conf]}</span></div>
    <button class="mapwrap" onclick="toggleMap()" title="expand">${poolMap(false)}</button>
    ${langBlock()}
    <div class="buildfoot">${cHard.length
      ? `<span style="color:var(--rust)">${icon('warn')} ${cHard.length} unresolved
         conflict${cHard.length>1?'s':''} in your labels.</span> Whatever this says about quality is
         measured on decisions that disagree with each other, which is the one thing a number here
         cannot correct for. <button class="linkbtn" onclick="openConf()">Answer them</button>`
      : conf===0?'A model trained on this now would do worse than the plain search — nothing is hidden, there is just nothing worth showing yet.'
      :conf===3?'Trained on your decisions and nobody else\'s. It orders your deck and can sort the pool on its own.'
      :'Running, and not yet better than the plain search on every use case this size.'}</div>`;

  document.getElementById('cqHead').innerHTML=`${icon('compare')} Label conflicts
    ${cHard.length?`<span class="cqBadge">${cHard.length}</span>`
      :cSoft.length?`<span class="cqBadge q">${cSoft.length}</span>`:''}`;
  document.getElementById('cqGrp').style.display = (cAll.length||S.notes.length)?'block':'none';
  document.getElementById('cqBody').innerHTML = cAll.length
    ? `<div class="cq">${cAll.slice(0,3).map(c=>`
        <button class="cqR ${c.sev==='check'?'chk':''}" onclick="openConf()">
          <span class="ic">${icon(c.sev==='conflict'?'warn':'compare')}</span>
          <span><span class="n">${c.hd}</span><span class="s">${
            c.sev==='conflict'?'counted against your model quality':'a question, not a cost'}</span></span>
          <span class="go">Look →</span></button>`).join('')}
        ${cAll.length>3?`<button class="cqR chk" onclick="openConf()"><span class="ic"></span>
          <span><span class="n">…and ${cAll.length-3} more</span></span><span class="go">All →</span></button>`:''}</div>`
    : `<div class="cqNone">${icon('circleCheck')} Nothing contradicts itself. Re-checked after every
        decision, and ${S.notes.length
          ? `${S.notes.length} distinction${S.notes.length>1?'s':''} you recorded ${S.notes.length>1?'are':'is'} waiting for the description.`
          : 'nothing is being counted against your model.'}</div>`;

  document.getElementById('setaside').innerHTML = S.aside.length
    ? S.aside.map(c=>`<div class="aside"><span>“${c.term}” — ${c.n} you turned down</span>
        <button class="re" onclick="reopen('${c.term}')">Look</button></div>`).join('')
    : '<div class="emptyq">Anything you put off comes here. Nothing waiting.</div>';
  document.getElementById('ctx').innerHTML = S.screen==='sort'? `<b>${p}</b> relevant · <b>${n}</b> not` : '';
  document.getElementById('legend').classList.toggle('hid',seen()>=8);
  renderCrumbs(); renderHome(); renderUse();
}

/* ═══════════════════════════════════════════════════════════════════════
   USE
   ═══════════════════════════════════════════════════════════════════════ */
/* The model's picks. `why` carries only what is HONESTLY available: the classifier
   works on whole-abstract vectors, so there is no term-level explanation to give.
   What can be shown truthfully is (a) the confidence, (b) which of the analyst's
   OWN keeps it most resembles, and (c) which of their described words appear in it.
   Inventing feature importances for an embedding model would be the ADR 0009
   failure — a fabricated explanation looks exactly like a measurement. */
const PRED=[
 {t:'Sodium silicate modulus and its effect on one-part geopolymer strength',v:'Cement and Concrete Research',y:2025,c:0.94,
  ab:"Activator modulus is the single most influential formulation variable in one-part alkali-activated systems, and published optima disagree by a factor of two.|Modulus was varied from 0.9 to 2.1 across four precursor blends with strength, calorimetry and pore solution analysis to 90 days.|Peak 28-day strength occurred at a modulus of 1.4 in every blend tested, at 47 to 58 MPa.|Above 1.6 the systems gained early strength and lost it again by 56 days, which the authors link to gel restructuring.|The consistency of the optimum across precursors is the practically useful result.",
  cue:[2,4],near:'Curing regimes for one-part alkali-activated binders',words:['alkali-activated','one-part','strength'],sum:'Finds the activator modulus optimum sits at 1.4 across every precursor tested — a transferable number, which is what you asked for.'},
 {t:'Chloride ingress in high-volume slag concrete after ten years of marine exposure',v:'Cement and Concrete Composites',y:2025,c:0.91,
  ab:"Ten-year field data on high-replacement slag concrete in marine exposure is scarce, and service-life models for it are calibrated almost entirely on laboratory migration testing.|Cores were taken from tidal and splash-zone elements at three replacement levels and profiled for chloride.|Apparent diffusion coefficients were three to five times lower than the laboratory prediction at the same age.|The ageing exponent required to fit the field data was 0.61 against the 0.30 commonly assumed.|Reinforcement depassivation had not begun at any location.",
  cue:[2,3],near:'Chloride binding in slag-rich binders: a twelve-month marine exposure',words:['chloride','slag','durability'],sum:'Ten-year field chlorides come in three to five times lower than the lab test predicts, so your durability criterion may be conservative.'},
 {t:'Limestone calcined clay cement: a field trial in a tropical climate',v:'Construction and Building Materials',y:2025,c:0.89,
  ab:"LC3 has been characterised extensively in the laboratory but field evidence from the climates where it is being deployed fastest remains limited.|A 900 m³ pour was placed at 34 °C ambient with full instrumentation and companion laboratory specimens.|In-situ strength at 28 days was 8% below the laboratory companions, within normal expectations.|Setting was 40 minutes faster than the laboratory prediction, which required a change of admixture mid-trial.|Thermal profiles were lower than the OPC control throughout.",
  cue:[2,3],near:'Setting behaviour and workability retention of calcined clay systems in hot climates',words:['calcined clay','limestone','setting'],sum:'A real pour at 34 °C: strength held, but setting ran 40 minutes fast and forced an admixture change mid-trial.'},
 {t:'Early-age cracking of alkali-activated concrete under restrained shrinkage',v:'Materials and Structures',y:2024,c:0.87,
  ab:"Restrained shrinkage cracking is the failure mode most likely to prevent structural use of alkali-activated concrete, and ring test data for it is sparse.|Twelve mixes were tested in restrained ring and slab configurations alongside free shrinkage measurement.|Cracking occurred between 3 and 19 days depending on activator modulus.|Internal curing delayed cracking beyond 28 days in three of four mixes where it was applied.|Free shrinkage alone was a poor predictor of cracking age.",
  cue:[2,3],near:'Autogenous and drying shrinkage of alkali-activated slag',words:['alkali-activated','shrinkage','strength'],sum:'Restrained cracking between 3 and 19 days depending on activator modulus — the failure mode most likely to block structural use.'},
 {t:'Supplementary cementitious materials from bauxite residue: reactivity screening',v:'Journal of Cleaner Production',y:2025,c:0.84,
  ab:"Bauxite residue stockpiles represent one of the largest mineral waste streams available, and reactivity screening across sources has not been done systematically.|Residues from eleven refineries were screened by R3 calorimetry after a common thermal treatment.|Heat release ranged from 74 to 232 J/g, a spread wider than between different material classes.|Iron and sodium content together explained 71% of the variance.|A screening index is proposed so sources can be triaged before pilot work.",
  cue:[2,3],near:'Supplementary cementitious materials from bauxite residue: activation and performance',words:['supplementary','reactivity'],sum:'Screens eleven bauxite residue sources; the spread between them is wider than between material classes, so source choice dominates.'}];
let expPreset=1, dwIdx=null;
const predState={};   /* id -> 'yes' | 'no' */
function renderUse(){
  const p=nPos(), n=nNeg(), ready=S.got.has('auto'), body=document.getElementById('useBody');
  if(!ready){
    body.innerHTML=`<h1>Nearly there</h1>
      <p class="lede">The model needs a few more examples before it can sort the pool on its own.
        Here is exactly what is missing.</p>
      <div class="lockcard"><div class="need">
        <div><div class="v">${Math.max(0,20-p)}</div><div class="l">more relevant papers</div></div>
        <div><div class="v">${Math.max(0,50-n)}</div><div class="l">more you turn down</div></div></div>
        <div style="font-size:13.5px;color:var(--ink2);line-height:1.6;max-width:56ch">
          The relevant ones are the part that matters — one is worth about ten rejections to the
          model. The fastest way to find them is to keep sorting, because the deck already puts the
          likely ones near the front.</div>
        <div style="margin-top:18px;display:flex;gap:10px">
          <button class="btn" onclick="go('sort')">Carry on sorting</button>
          <button class="btn ghost" onclick="go('describe')">Add papers I already know</button></div></div>
      <div class="firewall"><b>Why we don't just let it try now.</b> With fewer than twenty relevant
        examples it does worse than the plain search you started with — measurably, on eight test
        collections. We would rather say that than show you a confident-looking list built on nothing.</div>`;
    return;
  }
  const decided=Object.keys(predState).length;
  /* Order (Warren): the achievement, then keeping it running, then the picks.
     "Keep it running" is the thing with ongoing value, so it stops being a
     footer and becomes the second thing on the screen. */
  body.innerHTML=`
    ${openConflicts().length?`<div class="firewall" style="border-left-color:var(--rust);
      background:hsla(12,45%,45%,.08);margin-bottom:20px">
      <b>${openConflicts().length} unresolved conflict${openConflicts().length>1?'s':''} in the labels
      this was built from.</b> The model is fitted on decisions that disagree with each other, so
      everything below is real but optimistic. <button class="linkbtn" onclick="openConf()">See
      them</button> — most take one click.</div>`:''}
    <div class="readycard">
      <span class="eyebrow">Built from your ${p+n} decisions</span>
      <h2>You have a model that nobody else has, focused on ${S.ans.name||UCS[0].nm}.</h2>
      <p>It learned what you mean from ${p} papers you kept and ${n} you turned down — and from
        nothing else. No other use case fed it, and it will not work for anyone else's question.
        Then it read the whole pool in about a second.</p>
      <div class="readyStats">
        <div><div class="v">4,094</div><div class="l">papers it read that you never opened</div></div>
        <div><div class="v">61</div><div class="l">it expects you would keep</div></div>
        <div><div class="v">~30 hrs</div><div class="l">of reading it did instead of you</div></div>
      </div>
      <div style="display:flex;gap:10px;margin-top:4px;flex-wrap:wrap">
        <button class="btn onGreen" onclick="openTune()">${icon('sliders')} Tune your model</button>
        <button class="btn ghostGreen" onclick="openPred(0)">${icon('eye')} Read what it found</button></div></div>

    <div class="alertCard">
      <h3>Keep it running without you</h3>
      <p>New work is published every day and the model already knows what you want. It can keep
        watching and email you only what clears your bar.</p>
      <div class="alertRow">
        <button class="btn" onclick="openAlert()">${icon('mail')} Set up an alert</button>
        <button class="btn ghost" onclick="openEmailPrev()">${icon('eye')} Preview the email</button>
        <button class="btn ghost" onclick="openExport()">${icon('download')} Export instead</button></div></div>

    <div class="predHead"><h3>What it picked out
        <span class="im" data-tip="These are the model's guesses, not decisions. A guess never becomes something the model learns from — a model fed its own output starts agreeing with itself and every number after that is worthless. Confirming one in the drawer does train it.">!</span></h3>
      <span class="sub">${decided? `${decided} checked · ` : ''}open any one to see why</span></div>
    <div class="predlist" id="pl">${PRED.map((r,i)=>`
      <button class="pred ${dwIdx===i?'sel':''}" onclick="openPred(${i})">
        <span class="rk">${i+1}</span>
        <span class="ti">${r.t}<span class="vn">${r.v} · ${r.y}</span></span>
        <span class="conf"><span class="cbar"><i style="width:${r.c*100}%"></i></span><span class="cn">${(r.c*100).toFixed(0)}%</span></span>
        <span class="st ${predState[i]||''}">${predState[i]==='yes'?'You agreed':predState[i]==='no'?'You said no':'not checked'}</span>
      </button>`).join('')}
      <div class="pred" style="color:var(--ink3);cursor:default"><span></span><span>…and 56 more</span><span></span><span></span></div></div>
    <div class="predFoot"><span>Spot-checking a handful is how you find out whether to trust it.</span>
      <span class="sp" style="flex:1"></span>
      <button class="linkbtn" onclick="alert('Opens the same deck as the sorting screen, filtered to the model\'s picks — same two keys.')">Check them all in the deck</button></div>`;
}

/* ── the drawer: read the paper, see why, and re-decide ── */
function openPred(i){
  dwIdx=i; const r=PRED[i];
  document.getElementById('dwN').textContent=`Pick ${i+1} of 61 · ${(r.c*100).toFixed(0)}% confident`;
  const sent=r.ab.split('|');
  document.getElementById('dwBody').innerHTML=`
    <h2>${r.t}</h2><div class="auth">${r.v} · ${r.y}</div>
    <div class="dwWhy"><div class="t">Why it picked this</div>
      <div class="r"><span>How sure it is</span><b>${(r.c*100).toFixed(0)}% — its 4th highest of 4,094</b></div>
      <div class="r"><span>Closest to one you kept</span><b>${r.near}</b></div>
      <div class="r"><span>Your words that appear</span><b>${r.words.join(', ')}</b></div>
      <div class="lim">The model compares whole abstracts, not keywords, so it has no list of reasons
        to give you. The closest honest answer is which of your own picks this most resembles —
        the matching words are shown because they are checkable, not because it used them.</div></div>
    <div class="abs lens">${sent.map((s,k)=>`<span class="${r.cue.includes(k)?'cue':'dim'}">${s}</span>`).join(' ')}</div>
    <div class="cardfoot"><span class="tip" data-tip="The two or three sentences that usually carry the decision. The rest is shaded, never hidden.">Focus is on</span></div>`;
  document.getElementById('dwFoot').innerHTML=`
    <button class="act keep" onclick="decidePred(${i},'yes')"><kbd>J</kbd> Relevant</button>
    <button class="act" onclick="decidePred(${i},'no')"><kbd>K</kbd> Not for me</button>
    <span class="sp"></span>
    <span style="font-size:11.5px;color:var(--ink3);max-width:22ch;line-height:1.4">Deciding here
      does train it — that is the difference from its guess.</span>`;
  document.getElementById('drawer').classList.add('on');
  document.getElementById('scrim').classList.add('on');
  renderUse();
}
function closeDrawer(){ dwIdx=null;
  document.getElementById('drawer').classList.remove('on');
  document.getElementById('scrim').classList.remove('on'); renderUse() }
function dwStep(d){ if(dwIdx===null)return; const n=dwIdx+d; if(n<0||n>=PRED.length)return; openPred(n) }
function decidePred(i,v){ predState[i]=v; if(i+1<PRED.length){openPred(i+1)}else{closeDrawer()} }

/* ═══════════════════════════════════════════════════════════════════════
   TUNE · THREE MODELS AND ONE OPERATING POINT

   ── The one thing this panel must not do ──────────────────────────────
   Warren's brief says the save "will label N papers interesting/not
   interesting". It writes PREDICTIONS, not labels, and the screen says so in
   the confirmation rather than burying it: ADR 0004 is that `labeller='model'`
   rows never train. A model fed its own output starts agreeing with itself and
   every number after that is worthless — which is exactly the failure the
   confirmation has to describe, because "it labelled 61 papers for me" is what
   a user would otherwise reasonably believe happened.

   ── The three models, and what each one actually is here ──────────────
   Each is really computed, and each one's AUC on this screen is really measured
   from those scores — asserted numbers are what ADR 0009 is about.
     · SEARCH ORDER  cosine to your own description. No labels anywhere in it.
       Identical in the app: this is the order the pool arrived in.
     · LIBRARY MODEL a Rocchio centroid fitted on a FIXED half of the fixture,
       never on your labels. It stands in for a model fitted on the 28 public
       screening projects — which is a real thing the product could ship, since
       `acagent_demo` holds 181k labelled records.
     · YOUR MODEL    a Rocchio centroid on your own labels. Real. Below three
       positives there is nothing to fit, and the panel says which stand-in it
       fell back to rather than quietly drawing the library model's line.

   ── Why the F1 and F2 marks are computed, not placed ──────────────────
   Both optima are found by sweeping every cut-off and taking the argmax. If they
   were painted at decorative thirds of the track they would be a fake
   measurement, which looks exactly like a real one.

   ── The denominator, declared once ────────────────────────────────────
   Precision and recall here are measured on the fixture's 75 papers at 29%
   relevant. The real pool is 4,120 at 2.1%, where every number below is worse.
   What survives the difference is the SHAPE of the trade, which is what the
   panel is for. In the app these come from cross-validation on the analyst's own
   labels (`model_metrics.roc_auc`), never from a pool whose answers nobody knows.
   ═══════════════════════════════════════════════════════════════════════ */
const TUNE={model:'you',set:'f1',confirm:false,saved:null};
function openTune(){ TUNE.confirm=false; document.getElementById('tuneModal').classList.add('on'); renderTune() }
function closeTune(){ document.getElementById('tuneModal').classList.remove('on') }
function setTuneModel(m){ TUNE.model=m; TUNE.confirm=false; renderTune() }
function renderTune(){
  const sc=scoresFor(TUNE.model), fell=sc.fell, pts=curveFor(sc), N=pts.length;
  const F=fPoints(pts);
  if(TUNE.set===undefined||!(TUNE.set in F)) TUNE.set='f1';
  /* the folding lives in shared/charts.js now — the landing page shows the same
     five settings and had to fold them the same way */
  const groups=fGroups(F), merged=groups.filter(g=>g.keys.length>1);
  const at=pts[F[TUNE.set]], cutSet=new Set(at.order.slice(0,at.n));
  const nLab=Object.keys(S.lab).length, nP=Object.values(S.lab).filter(v=>v==='pos').length;
  const missed=Math.round((1-at.rc)*PAPERS.filter(p=>p.g==='pos').length);
  const mDesc=MODELS.find(m=>m[0]===TUNE.model)[2];
  document.getElementById('tuneBody').innerHTML=`
    <div class="tuneTop">
      <span class="lbl">Model</span>
      <span class="segs">${MODELS.map(([k,n])=>
        `<button class="${TUNE.model===k?'on':''}" onclick="setTuneModel('${k}')">${n}</button>`).join('')}</span>
      <span class="im right" data-tip="${mDesc.replace(/"/g,'&quot;')}${fell
        ? '  —  You have fewer than three labelled relevant, and a model needs three, so this is the General model standing in rather than your model drawn optimistically.'
        : ''}  Positions come from a two-component PCA of the same text the ranker reads: read the grouping, never the distances.">i</span>
      <span class="sp" style="flex:1"></span>
      ${fell?`<span style="font-size:11.5px;color:var(--gold-ink)">${icon('warn')} General model standing in —
        you have ${nP} labelled relevant of the 3 it needs</span>`:''}
    </div>
    <div class="tuneGrid">
      <div class="tuneMap">${poolMap(true,{cut:cutSet})}<div class="mapTip" id="mapTip"></div></div>
      <div class="tuneSide">
        <div class="pt">At this setting</div>
        <div class="tuneNums">
          <div class="tn"><span class="k">From your labels</span><span class="v sm">${nLab}</span></div>
          <div class="tn"><span class="k">Predicts interesting</span><span class="v">${at.n}</span></div>
          <div class="tn"><span class="k">Correct calls</span><span class="v">${(at.pr*100).toFixed(0)}%</span></div>
          <div class="tn"><span class="k">Interesting papers missed</span><span class="v">${missed}</span></div>
          <div class="tn"><span class="k">Interesting missed</span><span class="v">${((1-at.rc)*100).toFixed(0)}%</span></div>
        </div>
        <div class="tuneWho">${icon('info')} Correct calls and misses are measured on the papers you
          labelled yourself, by leaving each one out of the model that scores it. Nobody knows the
          answers for the rest of the pool — that is what you are choosing a setting for.</div>
      </div>
    </div>
    <div class="sldWrap">
      ${fRadio(pts,F,TUNE.set,"TUNE.set='%k';TUNE.confirm=false;renderTune()")}
      ${groups.length<FSET.length?`<div class="cnote" style="margin-top:13px">${icon('info')}
        <span><b>${FSET.length} settings, ${groups.length} distinct choices.</b> ${
        merged.map(g=>g.keys.map(fName).join(', ')).join('; ')} pick the same
        ${pts[F[merged[0].keys[0]]].n} papers, because recall does not improve between them — the ones
        it is still missing are ranked far down, so weighting misses more heavily changes nothing.
        Five buttons that do three things would teach you the names mean less than they do.</span></div>`:''}
    </div>
    ${TUNE.confirm? confirmBlock(at,cutSet,N) : TUNE.saved? doneBlock() : ''}
    <div class="tuneSave">
      <span class="sp"></span>
      <button class="btn ghost" onclick="closeTune()">Close</button>
      <button class="btn" onclick="TUNE.confirm=true;renderTune()" ${TUNE.confirm?'disabled':''}>
        ${icon('check')} Save this setting</button>
    </div>`;
}
function confirmBlock(at,cutSet,N){
  /* Warren's brief: "This will have impact X, but not change your model — confirm?"
     Two sentences, and the second is the one that matters: ADR 0004 is that a
     `labeller='model'` row never trains, so the honest summary of a save is that
     it changes what you SEE and not what the model KNOWS. */
  return `<div class="tuneConf">
    <span class="big">${at.n} papers will be flagged as interesting</span>
    This changes what you see. It does not change your model — flagged papers are guesses, and a
    guess never trains anything.
    <div style="margin-top:14px;display:flex;gap:10px">
      <button class="btn" onclick="TUNE.saved={n:${at.n},model:'${TUNE.model}',set:TUNE.set};TUNE.confirm=false;renderTune()">
        Confirm</button>
      <button class="btn ghost" onclick="TUNE.confirm=false;renderTune()">Back</button></div></div>`;
}
function doneBlock(){
  const nm=MODELS.find(m=>m[0]===TUNE.saved.model)[1];
  const lbl=(FSET.find(f=>f[0]===TUNE.saved.set)||['','this setting'])[1];
  return `<div class="tuneDone">${icon('circleCheck')} <b>Saved.</b> ${TUNE.saved.n} papers are
    flagged as interesting by <b>${nm}</b> at <b>${lbl}</b>. Pick another setting and confirm again
    to replace them — the last one wins, and anything you labelled by hand is untouched either way.</div>`;
}

/* ═══════════════════════════════════════════════════════════════════════
   THE EMAIL, AS THE READER GETS IT

   The reminder strip at the top reuses the WELCOME MODAL'S OWN DIAGRAMS — the
   same four SVGs, at digest scale, with the fourth marked as where the reader is
   standing. Redrawing them would let the two drift, and an alert that explains
   the product differently from the product is how a recipient who never used the
   app forms a wrong model of what they are being sent.

   Who the email is FOR matters here: the recipients need not have an account, so
   this is often the only surface a stakeholder ever sees. That is the argument for
   the strip, and the argument against F2 as its default operating point — someone
   else is doing the reading.
   ═══════════════════════════════════════════════════════════════════════ */
function openEmailPrev(){ go('emailprev') }
function renderEmailPrev(){
  const OP=opPoints(), at=OP[AL.op], FLAG=61;
  const nSend=Math.min(AL.count, Math.max(0,Math.round(FLAG*at.n/OP.loose.n)));
  const shown=PRED.slice(0,Math.min(nSend,PRED.length));
  const sum=AL.summarise&&aiOK();
  document.getElementById('epCtl').innerHTML=`<span class="epCtl">
    <span>${icon('clock')} ${AL.freq==='daily'?'every day':AL.freq==='weekly'?AL.day+'s':'monthly'} at ${AL.time}</span>
    <span>${icon('filter')} ${OPLBL[AL.op][0]}</span>
    ${sum?`<span>${icon('sparkles')} summarised by
      <select onchange="setAL('model',this.value);renderEmailPrev()">${MODELS_AVAIL
        .map(m=>`<option ${AL.model===m?'selected':''}>${m}</option>`).join('')}</select></span>`
      :`<span style="color:var(--ink3)">${icon('sparkles')} no summaries — ${
        S.llmOff?'AI is off for this workspace':'summarising is switched off'}</span>`}
    <button class="btn ghost sm" onclick="openAlert()">${icon('sliders')} Change settings</button></span>`;
  document.getElementById('epBody').innerHTML=`
    <div class="mail">
      <div class="mailHd">
        <div class="frm">${icon('radar')} TIRI · ${S.ans.name||UCS[0].nm}</div>
        <h2>${AL.subject} — ${shown.length||'nothing'} ${shown.length===1?'paper':'papers'}</h2>
        <div class="to">to ${AL.to}${AL.to.includes(',')?'':''} · Monday 24 August 2026, ${AL.time}</div>
      </div>
      <div class="mailIn">
        <div class="mailWhy"><div class="mwh">Why you are getting this</div>
          <div class="mwGrid">${WSL.map((w,i)=>`<div class="mwStep ${i===3?'at':''}">
            <div class="d">${w.d}</div><b>${i+1}. ${w.t}</b>
            <span>${['You described what matters','You sorted the first batch by hand',
              'It learned from those decisions only','This is that model, still reading'][i]}</span></div>`).join('')}
          </div></div>

        <div class="mailSec">${icon('circleCheck')} What cleared your bar this week</div>
        ${shown.length? shown.map(r=>`<div class="mp">
          <div class="mpt">${r.t}</div>
          <div class="mpm">${AL.meta.journal?`<span>${r.v}</span>`:''}${AL.meta.year?`<span>${r.y}</span>`:''}
            ${AL.meta.confidence?`<span class="mpc">${icon('trend')}${(r.c*100).toFixed(0)}% match</span>`:''}</div>
          ${sum?`<div class="mps"><span class="who">${AL.model} · a summary, not a finding</span>${r.sum}</div>`
            :(AL.meta.abstract?`<div class="mps" style="border-left-color:var(--line3);background:var(--paper2)">
               <span class="who" style="color:var(--ink3)">the paper's own abstract</span>${r.ab.split('|')[0]}</div>`:'')}
          ${AL.meta.link?`<div class="mpl">doi.org/10.1016/j.cemconres.2025.10${(700+r.y%40)}</div>`:''}
        </div>`).join('')
        : `<div class="mailNone"><b>Nothing cleared your bar.</b> We read 38 new papers since the last
           one and none of them reached ${OPLBL[AL.op][0].toLowerCase()}. This message exists so you
           can tell that apart from us not having looked.</div>`}
      </div>
      <div class="mailFt">
        Sorted by how well each one matches your description. ${sum
          ? `Summaries are written by ${AL.model} against your description — an opinion about the
             paper, not a finding from it.`
          : `No text was sent anywhere to produce this email.`}<br>
        Change settings · send to someone else · stop these emails
      </div>
    </div>
    <div class="epNote">${icon('info')} The recipients do not need an account, so for most of them this
      email is the whole product. That is why the four steps are repeated at the top of every one, and
      why the default bar is <b>${OPLBL[AL.op][0].toLowerCase()}</b> rather than the loosest setting —
      somebody else is doing the reading.</div>`;
}

/* ── alerts ── */
const AL={to:'warren@tiri.tech',freq:'weekly',day:'Monday',time:'08:00',
  subject:'New in low-carbon binders',count:10,op:'balanced',
  meta:{title:true,journal:true,year:true,link:true,confidence:true,abstract:false},
  summarise:true, model:'Claude Sonnet 4.6', sendEmpty:true};
function openAlert(){ document.getElementById('alertModal').classList.add('on'); renderAlert() }
function setAL(k,v){ AL[k]=v; renderAlert() }
function togMeta(k){ AL.meta[k]=!AL.meta[k]; renderAlert() }
const OPLBL={tight:['Only the best of it — half of what matters','tight'],
  balanced:['Balanced','the F1 setting'],
  loose:['Miss as little as possible — 95% of it','loose']};
function renderAlert(){
  /* v5 sized the digest with `61*(1-cut)/0.4`, a formula with nothing behind it.
     It now reads off the same curve the tune panel does: the share of the flagged
     set this bar keeps, and the precision measured at that bar. Both are measured
     on the analyst's own labelled papers, which the caption says, because a number
     whose provenance is not on screen gets read as a fact about the pool. */
  const OP=opPoints(), at=OP[AL.op], FLAG=61;
  const expect=Math.max(0,Math.round(FLAG*at.n/OP.loose.n));
  document.getElementById('alertForm').innerHTML=`
    <div class="fRow"><label>Who gets it</label>
      <input type="text" value="${AL.to}" oninput="AL.to=this.value">
      <div class="fNote">Comma-separated. They do not need an account.</div></div>
    <div class="fRow"><label>When</label>
      <div class="fInline">
        <select onchange="setAL('freq',this.value)">
          <option ${AL.freq==='daily'?'selected':''} value="daily">Every day</option>
          <option ${AL.freq==='weekly'?'selected':''} value="weekly">Every week</option>
          <option ${AL.freq==='monthly'?'selected':''} value="monthly">Every month</option></select>
        ${AL.freq!=='daily'?`<select onchange="setAL('day',this.value)">${
          ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
          .map(d=>`<option ${AL.day===d?'selected':''}>${d}</option>`).join('')}</select>`:''}
        <select onchange="setAL('time',this.value)">${['06:00','08:00','12:00','17:00','21:00']
          .map(t=>`<option ${AL.time===t?'selected':''}>${t}</option>`).join('')}</select>
      </div></div>
    <div class="fRow"><label>Subject line</label>
      <input type="text" value="${AL.subject}" oninput="AL.subject=this.value;renderAlert()">
      <div class="fNote">We append the count, so an empty week reads differently from a busy one.</div></div>
    <div class="fRow"><label>How much to send</label>
      <div class="fInline">
        <span class="grow"><select onchange="setAL('count',+this.value)">${[5,10,20,50]
          .map(c=>`<option ${AL.count===c?'selected':''} value="${c}">at most ${c} papers</option>`).join('')}</select></span>
        <span class="grow"><select onchange="setAL('op',this.value)" ${OP.collapsed?'disabled':''}>${
          ['tight','balanced','loose'].map(k=>`<option ${AL.op===k?'selected':''} value="${k}">${
            OPLBL[k][0]}</option>`).join('')}</select></span>
      </div>
      ${OP.collapsed?`<div class="fNote" style="color:var(--gold-ink)">All three settings land on the
        same ${at.n} papers with the labels you have so far — the bar cannot be moved usefully yet,
        so it is fixed until there is more to separate.</div>`:''}
      <div class="fNote">At this bar it would have sent <b>${expect}</b> of the ${FLAG} it flagged
        today, and about <b>${(at.pr*100).toFixed(0)}%</b> of those are usually worth reading
        (${(at.rc*100).toFixed(0)}% of everything relevant reaches you).
        <span class="im" data-tip="Both figures are measured on the papers you have labelled yourself, by leaving each one out of the model that scores it. They are not measured on the pool — nobody knows the pool's answers, which is the whole reason you labelled anything.">i</span>
        The same three settings live in <b>Tune your model</b>; this is that choice, applied on a
        schedule.</div></div>
    <div class="fRow"><label>What to include for each paper</label>
      <div class="fChecks">${Object.entries({title:'Title',journal:'Journal',year:'Year',
        link:'Link / DOI',confidence:'How sure it is',abstract:'Full abstract'})
        .map(([k,l])=>`<button class="fChk ${AL.meta[k]?'on':''}" onclick="togMeta('${k}')">${l}</button>`).join('')}</div></div>
    <div class="fRow"><label>Summarise each one against your description</label>
      <div class="fInline">
        <span class="tgl ${AL.summarise&&aiOK()?'on':''}" onclick="${aiOK()?"setAL('summarise',!AL.summarise)":''}"
          style="cursor:${aiOK()?'pointer':'not-allowed'};opacity:${aiOK()?1:.45}">
          <span class="sw"><i></i></span><span>${!aiOK()?'Unavailable':AL.summarise?'Yes':'No'}</span></span>
        ${AL.summarise&&aiOK()?`<select onchange="setAL('model',this.value)">${MODELS_AVAIL
          .map(m=>`<option ${AL.model===m?'selected':''}>${m}</option>`).join('')}</select>`:''}
      </div>
      ${!aiOK()?`<div class="fNote">AI features are off for this workspace, so the alert can only send
        what is already stored — title, journal, year, link and how sure it is. Turn them on in
        <b>AI Providers</b> on the description screen.</div>`:''}
      ${AL.summarise?`<div class="fNote">⚠️ This is the one thing here that sends your data on a
        schedule rather than when you press something: each paper's title and abstract, plus your
        description, go to ${AL.model} every time the alert runs. Nothing else does.</div>`:''}
    </div>
    <div class="fRow"><label>If it finds nothing</label>
      <div class="fInline">
        <span class="tgl ${AL.sendEmpty?'on':''}" onclick="setAL('sendEmpty',!AL.sendEmpty)" style="cursor:pointer">
          <span class="sw"><i></i></span><span>${AL.sendEmpty?'Send anyway':'Stay quiet'}</span></span></div>
      <div class="fNote">“We looked and found nothing” and “we did not look” are different facts.
        Sending the empty one is how you can tell which happened.</div></div>`;

  const shown=Math.min(AL.count,expect);
  document.getElementById('alertPrev').innerHTML=`
    <div class="email">
      <div class="eh"><div class="sub">${AL.subject} — ${shown||'nothing'} ${shown===1?'paper':'papers'}</div>
        <div class="meta">to ${AL.to.split(',')[0]}${AL.to.includes(',')?' +others':''} ·
          ${AL.freq==='daily'?'every day':AL.freq==='weekly'?`${AL.day}s`:'monthly'} at ${AL.time}</div></div>
      <div class="eb">${shown? PRED.slice(0,Math.min(shown,3)).map(r=>`<div class="ep">
        ${AL.meta.title?`<div class="t">${r.t}</div>`:''}
        ${(AL.meta.journal||AL.meta.year||AL.meta.confidence)?`<div class="m">${
          [AL.meta.journal?r.v:null,AL.meta.year?r.y:null,
           AL.meta.confidence?`${(r.c*100).toFixed(0)}% match`:null].filter(Boolean).join(' · ')}</div>`:''}
        ${AL.summarise?`<div class="s">${r.sum}</div>`:(AL.meta.abstract?`<div class="s">${r.ab.split('|')[0]}</div>`:'')}
        ${AL.meta.link?`<div class="m">doi.org/10.1016/…</div>`:''}</div>`).join('')
        +(shown>3?`<div class="ep" style="color:var(--ink3)">…and ${shown-3} more</div>`:'')
        : `<div class="ep" style="color:var(--ink3)">Nothing cleared your bar this week. We looked at
           38 new papers.</div>`}</div>
      <div class="ef">Sorted by how well each matches your description · unsubscribe · change settings</div>
    </div>`;

  document.getElementById('alertFoot').innerHTML=`
    <span class="warn">${AL.summarise?`Summarising sends abstracts to ${AL.model} on a schedule.`
      :'Nothing leaves this machine on a schedule with summarising off.'}</span>
    <span class="sp"></span>
    <button class="btn ghost" onclick="document.getElementById('alertModal').classList.remove('on')">Cancel</button>
    <button class="btn ghost" onclick="document.getElementById('alertModal').classList.remove('on');openEmailPrev()">See the whole email</button>
    <button class="btn" onclick="alert('Alert saved. It would run '+(AL.freq==='daily'?'daily':AL.day+'s')+' at '+AL.time+'.');document.getElementById('alertModal').classList.remove('on')">Turn it on</button>`;
}
function openExport(){ document.getElementById('expModal').classList.add('on'); renderExport() }
function renderExport(){
  document.getElementById('expForm').innerHTML=`
    <div class="fRow"><label>What are you doing with it?</label>
      <div class="presets" style="margin-top:9px">
        <button class="${expPreset===0?'sel':''}" onclick="expPreset=0;renderExport()"><span class="pt">Share a shortlist</span><span class="pd">A report someone can read</span></button>
        <button class="${expPreset===1?'sel':''}" onclick="expPreset=1;renderExport()"><span class="pt">Feed a model</span><span class="pd">Rows, with the vectors</span></button>
        <button class="${expPreset===2?'sel':''}" onclick="expPreset=2;renderExport()"><span class="pt">Reference manager</span><span class="pd">Zotero, EndNote</span></button></div></div>
    <div class="fRow"><div class="det" style="font-size:13px;color:var(--ink2);line-height:1.6">${
      ['A PDF or Markdown report: what you were looking for, what you kept and why, with the criteria and the counts.',
       'JSONL or Parquet, one row per paper, with the embedding vectors. Every row carries which use case and version it came from, the app version, the embedding model, and who decided what and when — not optional, because a training set nobody can attribute is one nobody can check.',
       'RIS or BibTeX. Deliberately plain: just the citations, no stamps.'][expPreset]}</div></div>
    <div class="fRow"><label>What to include</label>
      <div class="fChecks">
        <button class="fChk on">Your ${nPos()} relevant + ${nNeg()} not</button>
        <button class="fChk ${expPreset===1?'on':''}">The 61 guesses, marked as guesses</button>
        <button class="fChk ${expPreset===1?'on':''}">Embedding vectors</button>
        <button class="fChk on">Your description and its version</button></div>
      <div class="fNote">Guesses are always labelled as guesses in the file. A row that does not say
        who decided it is a row nobody downstream can check.</div></div>`;
}

/* ═══════════════════════════════════════════════════════════════════════
   NAV
   ═══════════════════════════════════════════════════════════════════════ */
/* No 'land' here any more: the landing page is its own document, so arriving at
   the app IS arriving at the workspace. The one thing that had to survive the
   split is the welcome modal — it used to be triggered by `launch()` on the way
   out of the landing page, and is now triggered on first load of this page from
   the same localStorage flag, so a visitor who has already seen it does not see
   it twice. */
const SCREENS=['home','describe','find','sort','use','emailprev'];
function go(s){ S.screen=s;
  SCREENS.forEach(k=>document.getElementById(k).classList.toggle('on',k===s));
  if(s==='describe') renderDescribe();
  if(s==='sort'){ if(!S.deck.length) S.deck=dealBatch(); draw() }
  if(s==='emailprev') renderEmailPrev();
  paint(); scrollTo(0,0);
}
function toggleKeys(){ document.getElementById('keys').classList.toggle('on') }
function jump(t){ S.ghostPos=Math.max(0,t-nPos()); S.ghostNeg=Math.max(0,(t>=20?63:Math.round(t*3.4))-nNeg());
  checkGates(); paint(); if(S.screen==='sort')draw() }
/* A detector with nothing to detect is untestable, and gold-labelling the deck
   never contradicts itself. This mislabels exactly the four things the four
   detectors look for, so the ladder can be walked. Nothing here is a fixture
   value pretending to be a measurement — it is the demo doing what a confused
   analyst does. */
function demoConflict(){
  /* Disjoint by construction. The first version picked each mistake independently
     and they collided — the LC3 paper is both a seed and a spec-satisfying paper,
     so the "rejected a paper matching your spec" flip silently undid the
     duplicate pair and two detectors reported the same paper. Each mistake now
     claims its papers and the next one works around them. */
  const used=new Set();
  const take=(arr,k)=>arr.filter(p=>!used.has(p.id)).slice(0,k).map(p=>(used.add(p.id),p));
  const near=(S.ans.near||['dental cements'])[0];

  /* ⑤ the same study twice, both ways */
  const dup=PAPERS.find(p=>p.v==='arXiv (preprint)');
  if(dup){
    const twin=PAPERS.find(p=>p.id!==dup.id&&jaccard(titleToks(p.id),titleToks(dup.id))>=CONF_DUP_MIN);
    if(twin){ used.add(dup.id); used.add(twin.id); S.lab[dup.id]='neg'; S.lab[twin.id]='pos' }
  }
  /* ⑥ a seed, turned down */
  take(PAPERS.filter(p=>p.seed&&!p.nab),1).forEach(p=>S.lab[p.id]='neg');
  /* ② a near-miss, kept */
  take(PAPERS.filter(p=>!p.nab&&paperHasStrong(p.id,near)),1).forEach(p=>S.lab[p.id]='pos');
  /* ② papers that satisfy the whole description, turned down */
  take(PAPERS.filter(p=>!p.nab
    && (S.ans.systems||[]).some(t=>paperHas(p.id,t))
    && (S.ans.outcome||[]).some(t=>paperHas(p.id,t))),3).forEach(p=>S.lab[p.id]='neg');

  S.cRes={}; S.cSeen={}; S.lastChk=0;
  checkGates(); paint(); if(S.screen==='sort') draw();
}
function autoplay(){
  if(window._auto){clearInterval(window._auto);window._auto=null;return}
  window._auto=setInterval(()=>{ const c=curCard();
    if(!c||c.k!=='paper'){clearInterval(window._auto);window._auto=null;return}
    label(PAPERS[c.id].g==='pos'?'pos':'neg') },170);
}
addEventListener('keydown',e=>{
  if(e.target?.matches?.('input,textarea,select')) return;
  const k=e.key.toLowerCase();
  if(k==='?'){ toggleKeys(); return }
  if(k==='escape'){ document.getElementById('keys').classList.remove('on'); return }
  if(S.screen!=='sort') return;
  const c=curCard();
  if(c&&c.k==='suggestion'){ if(k==='y')sugYes(S.i); else if(k==='n')sugNo(S.i); return }
  /* deliberately NO single-key answer on a check card. J/K/Y/N are reflexes by
     this point in a sitting, and the whole reason this card is a different colour
     is that it must not be answerable by reflex. */
  if(c&&c.k==='check'){ return }
  if(c&&c.k==='milestone'){ if(k==='enter')next(); return }
  if(k==='j'){ label('pos'); e.preventDefault() }
  else if(k==='k'){ label('neg'); e.preventDefault() }
  else if(k==='u'){ undo(); e.preventDefault() }
  else if(k==='f'){ S.lens=!S.lens; draw(); e.preventDefault() }
});
/* Derived, not typed: the strip described "72 papers" while the fixture grew to 76,
   which is the smallest possible version of a caption drifting from its own data. */
document.getElementById('demoNote').textContent =
  `fixture: 4,120-paper pool at 2.1% relevant · ${DEALABLE.length} dealt in batches of ${BATCH}`
  + ` (${DEALABLE.filter(p=>p.g==='pos').length} of them relevant)`;

/* ── boot ──────────────────────────────────────────────────────────────────
   `renderDescribe()` runs before the first paint because the interview screen
   is built from QS on every render and the dock's gate reads it; leaving it to
   the first `go('describe')` meant the dock measured an unrendered form. */
renderDescribe();
go('home');
maybeWelcome();
