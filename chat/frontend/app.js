/* Agibot X2 채팅 앱 프론트엔드 (프레임워크 없는 단일 파일 — 빌드 불필요) */
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const conn = document.getElementById('conn');
const emojiOverlay = document.getElementById('emoji-overlay');
const ctlMsg = document.getElementById('ctl-msg');

const EMOJI_MAP = {
  happy: '😊', sad: '😢', blink: '😉', thinking: '🤔', calm: '😌',
  cute: '🥰', angry: '😠', surprised: '😮', sleepy: '😴',
};

function addMsg(kind, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + kind;
  if (kind === 'say') text = '🔊 ' + text;
  if (kind === 'event') text = '⚙ ' + text;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

/* ---------- 상태 배지 ---------- */
function setBadge(id, text, cls) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = 'badge' + (cls ? ' ' + cls : '');
}
function renderStatus(s) {
  setBadge('b-mode', '모드: ' + (s.mode ?? '?'),
           s.mode && s.mode.startsWith('LOCOMOTION') ? 'warn' : 'ok');
  if (s.upright !== undefined)
    setBadge('b-upright', s.upright ? '자세: 직립' : '자세: 이상!',
             s.upright ? 'ok' : 'err');
  if (s.position)
    setBadge('b-pos', `위치: (${s.position.x}, ${s.position.y})`);
  if (s.battery_pct !== undefined) {
    const p = s.battery_pct;
    setBadge('b-batt', `배터리: ${p}%`,
             p > 30 ? 'ok' : p > 15 ? 'warn' : 'err');
  }
}

/* ---------- WebSocket ---------- */
let ws = null;
function connect() {
  ws = new WebSocket(
    (location.protocol === 'https:' ? 'wss://' : 'ws://')
    + location.host + '/ws');
  ws.onopen = () => { conn.textContent = '온라인'; conn.className = 'badge ok'; };
  ws.onclose = () => {
    conn.textContent = '재연결 중…'; conn.className = 'badge err';
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.kind === 'status') { renderStatus(m.state); return; }
    if (m.kind === 'emoji') {
      emojiOverlay.textContent = EMOJI_MAP[m.text] || '🙂';
      setTimeout(() => (emojiOverlay.textContent = ''), 5000);
      addMsg('sys', `😀 표정: ${m.text}`);
      return;
    }
    addMsg(m.kind, m.text);
  };
}
connect();

/* ---------- 입력 ---------- */
document.getElementById('input-row').addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || !ws || ws.readyState !== 1) return;
  addMsg('user', text);
  ws.send(JSON.stringify({ kind: 'user', text }));
  input.value = '';
});

/* ---------- 버튼 (하드와이어 정지 / 리셋) ---------- */
document.getElementById('btn-stop').addEventListener('click', () => {
  fetch('/stop', { method: 'POST' })
    .then(r => r.text()).then(t => {
      ctlMsg.textContent = t;
      addMsg('sys', '🛑 즉시 정지 (LLM 독립 경로)');
      setTimeout(() => (ctlMsg.textContent = ''), 3000);
    });
});
document.getElementById('btn-reset').addEventListener('click', () => {
  ctlMsg.textContent = '초기화 중…';
  fetch('/reset', { method: 'POST' })
    .then(r => r.text()).then(t => {
      ctlMsg.textContent = t;
      addMsg('sys', '🔄 시뮬 초기화됨');
      setTimeout(() => (ctlMsg.textContent = ''), 4000);
    });
});

/* ---------- 이력 로드 ---------- */
fetch('/history').then(r => r.json()).then(rows => {
  rows.forEach(r => addMsg(r.kind === 'user' ? 'user'
    : r.kind === 'say' ? 'say' : 'reply', r.text));
  addMsg('sys', '— 이전 대화 불러옴 —');
}).catch(() => {});
