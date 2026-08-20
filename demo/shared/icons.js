/* ═══════════════════════════════════════════════════════════════════════
   ICONS
   Drawn to the Lucide/Feather contract — 24x24 grid, 1.75 stroke, round caps
   and joins, no fills, currentColor — so the real package drops in one-for-one
   and every name below is the Lucide name. They are hand-drawn here only
   because a single HTML file cannot import an icon package, and saying "these
   are Lucide" when they are approximations would be the same class of lie as a
   fabricated measurement. Both licences are permissive (Lucide ISC, Feather
   MIT); in the app it is `lucide-react`, one import, no sprite sheet.

   Rule for using them: an icon marks a HEADING or an ACTION and never carries
   meaning on its own. Every one here sits beside a word.
   ═══════════════════════════════════════════════════════════════════════ */
const IC={
 radar:'<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 12l6.4-4.2"/>',
 layers:'<path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="M3 13l9 5 9-5"/>',
 pie:'<circle cx="12" cy="12" r="9"/><path d="M12 12V3"/><path d="M12 12l7.8 4.5"/>',
 shapes:'<circle cx="7.5" cy="16.5" r="4"/><rect x="13.5" y="12.5" width="7.5" height="7.5"/><path d="M12 3l4 7H8l4-7Z"/>',
 tag:'<path d="M20.5 12.5l-8 8-9-9V3.5h8l9 9Z"/><circle cx="7.6" cy="7.6" r="1.3"/>',
 gauge:'<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4.2-5.4"/>',
 target:'<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1"/>',
 warn:'<path d="M12 4 2.6 20h18.8L12 4Z"/><path d="M12 10v4.4"/><path d="M12 17.4h.01"/>',
 filePlus:'<path d="M13.5 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8.5L13.5 3Z"/><path d="M13.5 3v5.5H19"/><path d="M12 12.5v5"/><path d="M9.5 15h5"/>',
 sparkles:'<path d="M11 4l1.5 4.1L16.6 9.6 12.5 11 11 15.2 9.5 11 5.4 9.6 9.5 8.1 11 4Z"/><path d="M18 15.5l.65 1.7 1.7.65-1.7.65-.65 1.7-.65-1.7-1.7-.65 1.7-.65.65-1.7Z"/>',
 shield:'<path d="M12 3l8 3v5.8c0 4.9-3.3 7.9-8 9.2-4.7-1.3-8-4.3-8-9.2V6l8-3Z"/><path d="M9 12l2.2 2.2L15.4 10"/>',
 check:'<path d="M4.5 12.5l5 5L20 6.5"/>',
 x:'<path d="M6 6l12 12"/><path d="M18 6L6 18"/>',
 chevDown:'<path d="M6 9.5l6 6 6-6"/>',
 chevRight:'<path d="M9 6l6 6-6 6"/>',
 arrowRight:'<path d="M4 12h15.5"/><path d="M13.5 6l6 6-6 6"/>',
 arrowDown:'<path d="M12 4.5V20"/><path d="M6 14l6 6 6-6"/>',
 bars:'<path d="M3 20.5h18"/><path d="M6.5 20.5v-6.5"/><path d="M11.5 20.5V5"/><path d="M16.5 20.5v-10"/>',
 trend:'<path d="M3 17l6-6 4 4 8-8"/><path d="M17 7h4v4"/>',
 award:'<circle cx="12" cy="9" r="5.8"/><path d="M9 14L7.5 21l4.5-2.6L16.5 21 15 14"/>',
 map:'<path d="M3 6l6-2 6 2 6-2v14l-6 2-6-2-6 2V6Z"/><path d="M9 4v14"/><path d="M15 6v14"/>',
 archive:'<rect x="3" y="4" width="18" height="4.2"/><path d="M5 8.2V20h14V8.2"/><path d="M10 12h4"/>',
 circleCheck:'<circle cx="12" cy="12" r="9"/><path d="M8 12.4l2.6 2.6L16 9.6"/>',
 mail:'<rect x="3" y="5" width="18" height="14"/><path d="M3 7l9 6 9-6"/>',
 sliders:'<path d="M3 8h10"/><path d="M18 8h3"/><circle cx="15.5" cy="8" r="2.2"/><path d="M3 16h4"/><path d="M12 16h9"/><circle cx="9.5" cy="16" r="2.2"/>',
 download:'<path d="M12 4v11"/><path d="M8 11l4 4 4-4"/><path d="M4 19.5h16"/>',
 upload:'<path d="M12 15V4"/><path d="M8 8l4-4 4 4"/><path d="M4 19.5h16"/>',
 eye:'<path d="M2.5 12S6 5.6 12 5.6 21.5 12 21.5 12 18 18.4 12 18.4 2.5 12 2.5 12Z"/><circle cx="12" cy="12" r="3"/>',
 search:'<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.4 15.4L21 21"/>',
 info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7.6h.01"/>',
 compare:'<circle cx="6" cy="18" r="3"/><circle cx="18" cy="6" r="3"/><path d="M6 15V9.5A3.5 3.5 0 0 1 9.5 6H13"/><path d="M18 9v5.5a3.5 3.5 0 0 1-3.5 3.5H11"/>',
 flow:'<path d="M4 4v16"/><path d="M4 8.5h6.5a3.5 3.5 0 0 1 3.5 3.5v1a3.5 3.5 0 0 0 3.5 3.5H21"/><path d="M4 15h4.5"/>',
 users:'<circle cx="9" cy="8" r="3.6"/><path d="M2.8 20a6.2 6.2 0 0 1 12.4 0"/><path d="M16 4.8a3.6 3.6 0 0 1 0 7.2"/><path d="M17 14.6a6.2 6.2 0 0 1 4.2 5.4"/>',
 flask:'<path d="M9 3v6L4 20h16L15 9V3"/><path d="M8 3h8"/><path d="M6.6 15h10.8"/>',
 filter:'<path d="M3 5h18l-7 8.2V20l-4-2.2v-4.6L3 5Z"/>',
 clock:'<circle cx="12" cy="12" r="9"/><path d="M12 7.5V12l3.4 2"/>',
 file:'<path d="M13.5 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8.5L13.5 3Z"/><path d="M13.5 3v5.5H19"/>',
 plus:'<path d="M12 5v14"/><path d="M5 12h14"/>'
};
const icon=(n,cls)=>`<svg class="ic${cls?' '+cls:''}" viewBox="0 0 24 24" fill="none" stroke="currentColor"
  stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${IC[n]||''}</svg>`;
