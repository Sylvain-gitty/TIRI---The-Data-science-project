/* ═══════════════════════════════════════════════════════════════════════
   STATE
   ═══════════════════════════════════════════════════════════════════════ */
const S={ screen:'home', arch:false, lab:{}, deck:[], i:0, hist:[], lens:false, pending:false,
  aside:[], seenTerms:new Set(), declined:0, excluded:0, filtered:0, got:new Set(),
  ghostPos:0, ghostNeg:0, batch:0, ans:{}, focusQ:0, aiBrief:null, dropped:new Set(),
  aiOpen:null, otherOpen:false, dragOver:false, tplFrom:null,
  /* ONE place for the AI settings, because the answer has to hold everywhere:
     `llmOff` disables every AI affordance in the app, and `model` is the default
     every model picker starts on. Two screens each remembering their own choice
     is how a privacy setting becomes advisory. */
  llmOff:false, model:'Claude Sonnet 4.6',
  cRes:{}, cSeen:{}, notes:[], lastChk:0, cAll:[],
  regions:{eu:true,us:false,cn:false,row:false} };
const MODELS_AVAIL=['Claude Sonnet 4.6','Nemotron 3 Super 120B','Gemma 4 31B','Qwen3 32B'];
/* one predicate, asked everywhere an AI button exists */
const aiOK=()=>!S.llmOff && Object.values(S.regions).some(Boolean);

const seen=()=>Object.keys(S.lab).length;
const nPos=()=>Object.values(S.lab).filter(v=>v==='pos').length+S.ghostPos;
const nNeg=()=>Object.values(S.lab).filter(v=>v==='neg').length+S.ghostNeg;
const ctrlSeen=()=>Object.keys(S.lab).filter(id=>PAPERS[id].arm==='control').length;

/* the dealable pool: never a card with no abstract */
const DEALABLE=PAPERS.filter(p=>!p.nab);
/* The fixture starts BLANK so the empty state is what you see first (Warren asked
   to see it). `demoFill()` on the strip loads a worked answer set. */
S.ans={};
