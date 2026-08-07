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

/* ---------- 로봇 음성 (브라우저 TTS — Mac 스피커로 재생) ---------- */
let voiceOn = true;
let koVoice = null;
function pickVoice() {
  const vs = speechSynthesis.getVoices();
  koVoice = vs.find(v => v.lang.startsWith('ko') && /Yuna|유나/i.test(v.name))
    || vs.find(v => v.lang.startsWith('ko')) || null;
}
if ('speechSynthesis' in window) {
  pickVoice();
  speechSynthesis.onvoiceschanged = pickVoice;
}
function robotSpeak(text) {
  if (!voiceOn || !('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'ko-KR';
  if (koVoice) u.voice = koVoice;
  u.rate = 1.05;
  u.pitch = 1.1;              // 로봇 느낌 살짝
  speechSynthesis.cancel();   // 겹치면 최신 발화 우선
  speechSynthesis.speak(u);
}
document.getElementById('btn-voice').addEventListener('click', (e) => {
  voiceOn = !voiceOn;
  if (!voiceOn) speechSynthesis.cancel();
  e.target.textContent = voiceOn ? '🔊 음성 ON' : '🔇 음성 OFF';
  e.target.classList.toggle('off', !voiceOn);
});

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
    if (m.kind === 'say') robotSpeak(m.text);
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

/* ---------- 기능 레지스트리 기반 UI 노출 (A-0) ----------
   sim 전용 컨트롤(월드 드롭다운/리셋)은 백엔드 /features가
   허용할 때만 표시 — 실기 모드에서 자동으로 사라진다. */
fetch('/features').then(r => r.json()).then(f => {
  if (!f.ui.world_switch) {
    const el = document.getElementById('world-sel');
    if (el) el.style.display = 'none';
  }
  if (!f.ui.sim_reset) {
    const el = document.getElementById('btn-reset');
    if (el) el.style.display = 'none';
  }
}).catch(() => {});

/* ---------- 월드 선택 ---------- */
const worldSel = document.getElementById('world-sel');
function loadWorlds() {
  fetch('/worlds').then(r => r.json()).then(d => {
    worldSel.innerHTML = '';
    for (const [name, desc] of Object.entries(d.worlds)) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = `🌍 ${desc}`;
      if (name === d.current) opt.selected = true;
      worldSel.appendChild(opt);
    }
  });
}
loadWorlds();
worldSel.addEventListener('change', () => {
  ctlMsg.textContent = '월드 전환 중…';
  fetch('/world', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: worldSel.value }),
  }).then(r => r.text()).then(t => {
    ctlMsg.textContent = t;
    addMsg('sys', `🌍 월드 전환: ${worldSel.value}`);
    setTimeout(() => (ctlMsg.textContent = ''), 4000);
  });
});

/* ---------- 이력 로드 ---------- */
fetch('/history').then(r => r.json()).then(rows => {
  rows.forEach(r => addMsg(r.kind === 'user' ? 'user'
    : r.kind === 'say' ? 'say' : 'reply', r.text));
  addMsg('sys', '— 이전 대화 불러옴 —');
}).catch(() => {});
