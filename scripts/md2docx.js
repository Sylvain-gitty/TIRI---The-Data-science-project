// Markdown -> DOCX for the TIRI whitepaper.
// Handles: headings, paragraphs, bullets, ordered lists, tables (with :--- alignment),
// blockquotes, images, horizontal rules, and inline **bold** / *italic* / `code`.
const fs = require('fs');
const path = require('path');
const D = require('docx');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, WidthType, BorderStyle, ShadingType, PageOrientation,
  Footer, PageNumber, LevelFormat, convertInchesToTwip,
} = D;

const MD = process.argv[2];
const OUT = process.argv[3];
const SRC_DIR = path.dirname(path.resolve(MD));

// A4 portrait, 1" margins  ->  content width in DXA
const PAGE_W = 11906, MARGIN = 1440;
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9026

const INK = '1A1A1A', MUTED = '555555', RULE = 'BFBFBF', HEADSHADE = 'EFEFEF', ACCENT = '1F4E79';

// ---------- PNG dimensions (IHDR) ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// ---------- inline formatting ----------
function inline(text, base = {}) {
  const runs = [];
  // tokenise on **bold**, *italic*, `code`
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0, m;
  const push = (t, opts) => { if (t) runs.push(new TextRun(Object.assign({ text: t, color: INK }, base, opts))); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith('**')) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith('`')) push(tok.slice(1, -1), { font: 'Consolas', size: 18, color: '9C2A00' });
    else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return runs.length ? runs : [new TextRun(Object.assign({ text: '', color: INK }, base))];
}

// ---------- table helpers ----------
function splitRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(s => s.trim());
}
function alignOf(spec) {
  const s = spec.trim();
  if (s.endsWith(':') && s.startsWith(':')) return AlignmentType.CENTER;
  if (s.endsWith(':')) return AlignmentType.RIGHT;
  return AlignmentType.LEFT;
}
function buildTable(headers, aligns, rows) {
  const n = headers.length;
  // proportional widths from mean content length, clamped
  const weights = [];
  for (let c = 0; c < n; c++) {
    let tot = headers[c].replace(/[*`]/g, '').length;
    for (const r of rows) tot += ((r[c] || '').replace(/[*`]/g, '').length);
    weights.push(Math.max(6, Math.min(60, tot / (rows.length + 1))));
  }
  const sum = weights.reduce((a, b) => a + b, 0);
  let widths = weights.map(w => Math.max(700, Math.round(CONTENT_W * w / sum)));
  const drift = CONTENT_W - widths.reduce((a, b) => a + b, 0);
  widths[widths.indexOf(Math.max(...widths))] += drift;

  const mkRow = (cells, isHead) => new TableRow({
    tableHeader: isHead || undefined,
    children: cells.map((t, i) => {
      return new TableCell({
        width: { size: widths[i], type: WidthType.DXA },
        shading: isHead ? { type: ShadingType.CLEAR, fill: HEADSHADE, color: 'auto' } : undefined,
        margins: { top: 60, bottom: 60, left: 110, right: 110 },
        children: [new Paragraph({
          alignment: aligns[i] || AlignmentType.LEFT,
          spacing: { before: 20, after: 20 },
          children: inline(t, { size: 18, bold: isHead || undefined }),
        })],
      });
    }),
  });

  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      right: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: RULE },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: RULE },
    },
    rows: [mkRow(headers, true), ...rows.map(r => mkRow(r, false))],
  });
}

// ---------- main parse ----------
const lines = fs.readFileSync(MD, 'utf8').replace(/\r\n/g, '\n').split('\n');
const body = [];
let i = 0;
let inFrontMatter = true; // treat pre-first-"---" title block specially

function para(text, opts = {}) {
  return new Paragraph(Object.assign({
    spacing: { before: 60, after: 120, line: 276 },
    children: inline(text),
  }, opts));
}

while (i < lines.length) {
  let line = lines[i];

  // blank
  if (!line.trim()) { i++; continue; }

  // horizontal rule
  if (/^---+\s*$/.test(line)) {
    body.push(new Paragraph({
      spacing: { before: 160, after: 160 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 1 } },
      children: [new TextRun('')],
    }));
    i++; continue;
  }

  // image  ![alt](file.png)
  let im = line.trim().match(/^!\[([^\]]*)\]\(([^)]+)\)\s*$/);
  if (im) {
    const file = path.resolve(SRC_DIR, im[2]);
    const { w, h } = pngSize(file);
    const maxPx = 600;                       // ~6.25in at 96dpi
    const scale = Math.min(1, maxPx / w);
    body.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 180, after: 60 },
      children: [new ImageRun({
        type: 'png',
        data: fs.readFileSync(file),
        transformation: { width: Math.round(w * scale), height: Math.round(h * scale) },
      })],
    }));
    i++; continue;
  }

  // heading
  let hm = line.match(/^(#{1,4})\s+(.*)$/);
  if (hm) {
    const lvl = hm[1].length, txt = hm[2].trim();
    if (lvl === 1) {
      body.push(new Paragraph({
        heading: HeadingLevel.TITLE,
        spacing: { before: 0, after: 160 },
        children: inline(txt, { size: 40, bold: true, color: ACCENT }),
      }));
    } else if (lvl === 2) {
      body.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 400, after: 140 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 } },
        children: inline(txt, { size: 28, bold: true, color: ACCENT }),
      }));
    } else if (lvl === 3) {
      body.push(new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 280, after: 100 },
        children: inline(txt, { size: 24, bold: true, color: ACCENT }),
      }));
    } else {
      body.push(new Paragraph({
        heading: HeadingLevel.HEADING_3,
        spacing: { before: 220, after: 90 },
        children: inline(txt, { size: 21, bold: true, color: INK }),
      }));
    }
    i++; continue;
  }

  // table
  if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|?\s*$/.test(lines[i + 1])) {
    const headers = splitRow(lines[i]);
    const aligns = splitRow(lines[i + 1]).map(alignOf);
    i += 2;
    const rows = [];
    while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(splitRow(lines[i])); i++; }
    body.push(buildTable(headers, aligns, rows));
    body.push(new Paragraph({ spacing: { before: 0, after: 160 }, children: [new TextRun('')] }));
    continue;
  }

  // blockquote (may span lines, may contain a blank ">" separator)
  if (/^>\s?/.test(line)) {
    const chunks = [];
    let cur = [];
    while (i < lines.length && /^>/.test(lines[i])) {
      const t = lines[i].replace(/^>\s?/, '');
      if (!t.trim()) { if (cur.length) { chunks.push(cur.join(' ')); cur = []; } }
      else cur.push(t.trim());
      i++;
    }
    if (cur.length) chunks.push(cur.join(' '));
    chunks.forEach((c, k) => {
      body.push(new Paragraph({
        spacing: { before: k === 0 ? 160 : 60, after: k === chunks.length - 1 ? 180 : 60, line: 276 },
        indent: { left: 340 },
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 12 } },
        shading: { type: ShadingType.CLEAR, fill: 'F2F6FA', color: 'auto' },
        children: inline(c, { size: 21 }),
      }));
    });
    continue;
  }

  // bullet list
  if (/^\s*[-*]\s+/.test(line)) {
    while (i < lines.length && (/^\s*[-*]\s+/.test(lines[i]) || /^\s{2,}\S/.test(lines[i]))) {
      if (/^\s*[-*]\s+/.test(lines[i])) {
        const indent = Math.floor((lines[i].match(/^\s*/)[0].length) / 2);
        let txt = lines[i].replace(/^\s*[-*]\s+/, '');
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*[-*]\s+/.test(lines[i])) {
          txt += ' ' + lines[i].trim(); i++;
        }
        body.push(new Paragraph({
          numbering: { reference: 'tiri-bullets', level: Math.min(indent, 1) },
          spacing: { before: 40, after: 60, line: 276 },
          children: inline(txt),
        }));
      } else i++;
    }
    body.push(new Paragraph({ spacing: { before: 0, after: 100 }, children: [new TextRun('')] }));
    continue;
  }

  // ordered list
  if (/^\s*\d+\.\s+/.test(line)) {
    while (i < lines.length && (/^\s*\d+\.\s+/.test(lines[i]) || /^\s{2,}\S/.test(lines[i]))) {
      if (/^\s*\d+\.\s+/.test(lines[i])) {
        let txt = lines[i].replace(/^\s*\d+\.\s+/, '');
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*\d+\.\s+/.test(lines[i])) {
          txt += ' ' + lines[i].trim(); i++;
        }
        body.push(new Paragraph({
          numbering: { reference: 'tiri-numbers', level: 0 },
          spacing: { before: 40, after: 60, line: 276 },
          children: inline(txt),
        }));
      } else i++;
    }
    body.push(new Paragraph({ spacing: { before: 0, after: 100 }, children: [new TextRun('')] }));
    continue;
  }

  // plain paragraph (join wrapped lines)
  let buf = [line.trim()];
  i++;
  while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|>|\s*[-*]\s|\s*\d+\.\s|\s*\||---+\s*$|!\[)/.test(lines[i])) {
    buf.push(lines[i].trim()); i++;
  }
  const text = buf.join(' ');
  // a whole-line italic paragraph directly after an image is a figure caption
  const capm = text.match(/^\*(.+)\*$/s);
  const prev = body[body.length - 1];
  const afterImage = prev instanceof Paragraph && JSON.stringify(prev).includes('ImageRun') === false;
  if (capm && /^Figure /.test(capm[1])) {
    body.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 220 },
      children: inline(capm[1], { size: 17, italics: true, color: MUTED }),
    }));
  } else {
    body.push(para(text));
  }
}

const doc = new Document({
  creator: 'Sylvain Fossier & Warren Fauvel',
  title: 'TIRI — A Radar for Interesting Science',
  description: 'Whitepaper on the TIRI relevance-screening workflow and results',
  styles: {
    default: {
      document: { run: { font: 'Calibri', size: 21, color: INK }, paragraph: { spacing: { line: 276 } } },
    },
  },
  numbering: {
    config: [
      {
        reference: 'tiri-bullets',
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 360, hanging: 220 } } } },
          { level: 1, format: LevelFormat.BULLET, text: '◦', alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 220 } } } },
        ],
      },
      {
        reference: 'tiri-numbers',
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 400, hanging: 260 } } } },
        ],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: 16838, orientation: PageOrientation.PORTRAIT },
        margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: ['TIRI whitepaper  ·  ', PageNumber.CURRENT], size: 16, color: MUTED })],
        })],
      }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log('wrote', OUT, b.length, 'bytes'); });
