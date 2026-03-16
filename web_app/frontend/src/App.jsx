import { useState, useEffect, useRef, useCallback } from 'react';

const API = '/api/predict';

const LABEL_NAMES = [
  'clean','racism','sexism','profanity','cyberbullying',
  'toxicity','hate_speech','implicit_hate','threat','sarcasm'
];

const CAT = {
  clean: {
    color:'#16a34a', sev:'safe', desc:'Safe content',
    explain:'This message contains no harmful content and reflects healthy communication.',
    forParent:'No action needed. Encourage your child to continue communicating respectfully.',
    forChild:'Great job! Respectful conversations build strong friendships.',
  },
  racism: {
    color:'#dc2626', sev:'danger', desc:'Racial discrimination',
    explain:'Language that targets or demeans someone based on their race, ethnicity, or national origin.',
    forParent:'Have an open conversation about why racial stereotypes and slurs cause real harm. Help your child understand that everyone deserves equal respect regardless of background.',
    forChild:'Words that target someone\'s race can cause deep pain. Everyone is equal — treat others the way you want to be treated.',
  },
  sexism: {
    color:'#e11d48', sev:'danger', desc:'Gender-based hate',
    explain:'Messages containing gender-based discrimination, stereotypes, or harassment.',
    forParent:'Discuss gender equality with your child. Explain that putting someone down because of their gender is never acceptable.',
    forChild:'Everyone deserves respect no matter their gender. Putting someone down because of who they are is never okay.',
  },
  profanity: {
    color:'#ea580c', sev:'warn', desc:'Obscene language',
    explain:'Use of vulgar, obscene, or inappropriate language in conversation.',
    forParent:'Talk about appropriate vs. inappropriate language. Explain that profanity can hurt others and reflects poorly on the speaker.',
    forChild:'Using bad words can hurt people\'s feelings and make them uncomfortable. Try expressing yourself with kinder words.',
  },
  cyberbullying: {
    color:'#9333ea', sev:'danger', desc:'Personal attacks',
    explain:'Targeted personal attacks, harassment, or intimidation directed at an individual.',
    forParent:'Take this seriously. Ask your child if they are being bullied or bullying others. Save evidence, report to the platform, and contact the school if needed.',
    forChild:'If someone is being mean to you online, don\'t respond — tell a parent or trusted adult right away. You are not alone.',
  },
  toxicity: {
    color:'#7c3aed', sev:'warn', desc:'Toxic language',
    explain:'Generally negative, hostile, or harmful language that creates an unhealthy communication environment.',
    forParent:'Monitor the conversation pattern. Occasional negativity is normal, but persistent toxicity may indicate deeper issues. Encourage positive expression.',
    forChild:'Negative words can spread like a virus. Try to be the person who lifts others up, not brings them down.',
  },
  hate_speech: {
    color:'#b91c1c', sev:'danger', desc:'Hate speech',
    explain:'Targeted hatred against a group based on race, religion, gender, sexual orientation, or disability.',
    forParent:'This is a serious concern. Discuss the impact of hate speech with your child. Consider reporting to the platform and, if at school, to administrators.',
    forChild:'Hating someone for who they are is wrong. If you see hate speech, tell an adult — standing up against hate makes the world safer.',
  },
  implicit_hate: {
    color:'#c2410c', sev:'warn', desc:'Indirect / coded hate',
    explain:'Uses coded language, dog whistles, or indirect expressions to discriminate against a group.',
    forParent:'This is harder to detect. Learn common coded phrases and discuss with your child how some "jokes" can carry hidden harmful meanings.',
    forChild:'Some words seem harmless but actually hurt people in hidden ways. If something feels wrong, it probably is — ask an adult.',
  },
  threat: {
    color:'#991b1b', sev:'danger', desc:'Threats or violence',
    explain:'Messages containing direct or implied threats of physical harm or violence.',
    forParent:'Take all threats seriously, even if they seem like "jokes." Save the message, report it to the platform, and contact school officials or law enforcement if you feel the threat is credible.',
    forChild:'If someone threatens you — even as a joke — tell a parent or teacher immediately. Your safety always comes first.',
  },
  sarcasm: {
    color:'#4f46e5', sev:'info', desc:'Sarcasm / irony',
    explain:'Detected sarcastic or ironic tone. While sarcasm itself is not harmful, it can sometimes mask hostility.',
    forParent:'Sarcasm is a normal part of communication, but excessive sarcasm directed at someone can be a form of subtle bullying. Context matters.',
    forChild:'Sarcasm can be funny, but it can also hurt. Make sure the other person is laughing with you, not being hurt by your words.',
  },
};

const KB_LETTERS = [
  ['q','w','e','r','t','y','u','i','o','p'],
  ['a','s','d','f','g','h','j','k','l'],
  ['shift','z','x','c','v','b','n','m','backspace'],
  ['123','globe','space','return'],
];
const KB_NUMBERS = [
  ['1','2','3','4','5','6','7','8','9','0'],
  ['-','/',':',';','(',')','$','&','@','"'],
  ['#+=','.',',','?','!','\'','backspace'],
  ['ABC','globe','space','return'],
];
const KB_SYMBOLS = [
  ['[',']','{','}','#','%','^','*','+','='],
  ['_','\\','|','~','<','>','€','£','¥','•'],
  ['123','.',',','?','!','\'','backspace'],
  ['ABC','globe','space','return'],
];

const CONTROL_KEYS = new Set([
  'shift','backspace','space','return','globe','123','ABC','#+=',
]);

const INIT_MSGS = [
  { id:1, text:'Hey! How are you?', from:'them', analysis:null },
  { id:2, text:'Pretty good, just finished homework!', from:'me', analysis:null },
  { id:3, text:'Nice! Want to play games later?', from:'them', analysis:null },
];

export default function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState(INIT_MSGS);
  const [liveDetection, setLiveDetection] = useState(null);
  const [selectedMsgId, setSelectedMsgId] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [shift, setShift] = useState(false);
  const [kbMode, setKbMode] = useState('abc');
  const [model, setModel] = useState('distilbert');
  const [activeKey, setActiveKey] = useState(null);
  const [loading, setLoading] = useState(false);
  const msgEndRef = useRef(null);
  const nextId = useRef(4);

  const layout = kbMode === 'abc' ? KB_LETTERS
               : kbMode === 'num' ? KB_NUMBERS
               : KB_SYMBOLS;

  const selectedMsg = messages.find(m => m.id === selectedMsgId);
  const dashAnalysis = selectedMsg?.analysis || liveDetection;

  useEffect(() => {
    msgEndRef.current?.scrollIntoView({ behavior:'smooth' });
  }, [messages]);

  useEffect(() => {
    setAlerts([]);
    setLiveDetection(null);
    setMessages(prev => {
      const cleared = prev.map(m => ({ ...m, analysis: null }));
      cleared.forEach(m => {
        fetch(API, {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ text: m.text, model }),
        })
        .then(r => r.json())
        .then(d => {
          if (d.error) return;
          setMessages(p => p.map(msg => msg.id === m.id ? { ...msg, analysis: d } : msg));
          if (d.label !== 'clean') {
            setAlerts(p => {
              if (p.some(a => a.msgId === m.id)) return p;
              return [{ msgId:m.id, text:m.text, label:d.label, sev:CAT[d.label]?.sev, ts:Date.now() }, ...p];
            });
          }
        })
        .catch(() => {});
      });
      return cleared;
    });
  }, [model]);

  // Live typing detection
  useEffect(() => {
    if (!input.trim()) { setLiveDetection(null); return; }
    setSelectedMsgId(null);
    const t = setTimeout(() => analyzeLive(input), 350);
    return () => clearTimeout(t);
  }, [input, model]);

  const callApi = async (text) => {
    const r = await fetch(API, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ text, model }),
    });
    return r.json();
  };

  const analyzeLive = useCallback(async (text) => {
    setLoading(true);
    try {
      const d = await callApi(text);
      if (!d.error) setLiveDetection(d);
    } catch {}
    setLoading(false);
  }, [model]);

  const analyzeMessage = async (msgId, text) => {
    try {
      const d = await callApi(text);
      if (d.error) return;
      setMessages(prev => prev.map(m =>
        m.id === msgId ? { ...m, analysis: d } : m
      ));
      if (d.label !== 'clean') {
        setAlerts(prev => {
          if (prev.some(a => a.msgId === msgId)) return prev;
          return [{ msgId, text, label:d.label, sev:CAT[d.label]?.sev, ts:Date.now() }, ...prev];
        });
      }
    } catch {}
  };

  const pressKey = (key) => {
    setActiveKey(key);
    setTimeout(() => setActiveKey(null), 180);

    switch (key) {
      case 'shift':     setShift(s => !s); return;
      case 'backspace': setInput(s => s.slice(0,-1)); return;
      case 'space':     setInput(s => s + ' '); return;
      case 'return':    sendMsg(); return;
      case 'globe':     return;
      case '123':       setKbMode('num'); return;
      case 'ABC':       setKbMode('abc'); setShift(false); return;
      case '#+=':       setKbMode('sym'); return;
      default: break;
    }

    if (kbMode === 'abc') {
      const ch = shift ? key.toUpperCase() : key;
      setInput(s => s + ch);
      if (shift) setShift(false);
    } else {
      setInput(s => s + key);
    }
  };

  const sendMsg = () => {
    const text = input.trim();
    if (!text) return;
    const id = nextId.current++;
    setMessages(prev => [...prev, { id, text, from:'me', analysis:null }]);
    setInput('');
    setLiveDetection(null);
    setSelectedMsgId(id);
    analyzeMessage(id, text);
  };

  const handleMsgClick = (msgId) => {
    setSelectedMsgId(prev => prev === msgId ? null : msgId);
  };

  const safetyFromAnalysis = (a) => {
    if (!a) return { cls:'safe', text:'SafeType — Monitoring' };
    const s = CAT[a.label]?.sev || 'safe';
    if (s === 'safe')   return { cls:'safe',   text:'Content Safe' };
    if (s === 'danger') return { cls:'danger', text:`Harmful — ${a.label.replace(/_/g,' ')}` };
    if (s === 'warn')   return { cls:'warn',   text:`Caution — ${a.label.replace(/_/g,' ')}` };
    return { cls:'info', text:a.label.replace(/_/g,' ') };
  };

  const ss = safetyFromAnalysis(dashAnalysis);

  const renderKey = (k, ri) => {
    const isControl = CONTROL_KEYS.has(k);
    const isActive = activeKey === k;
    const display = getKeyDisplay(k, shift, kbMode);
    const cls = [
      'key',
      isControl ? k.replace(/[^a-zA-Z]/g,'') : '',
      isControl ? 'special' : '',
      isActive  ? 'pressed' : '',
      k === 'return' && input.trim() ? 'go' : '',
    ].filter(Boolean).join(' ');

    return (
      <button key={k + ri} className={cls}
        onMouseDown={() => pressKey(k)}
        onTouchStart={(e) => { e.preventDefault(); pressKey(k); }}>
        {k === 'shift' ? (
          <svg width="20" height="20" viewBox="0 0 20 20" className={shift?'shift-on':''}>
            <path d="M10 3L3 12h4v5h6v-5h4L10 3z"
              fill={shift?"currentColor":"none"} stroke="currentColor" strokeWidth="1.5"/>
          </svg>
        ) : k === 'backspace' ? (
          <svg width="22" height="18" viewBox="0 0 22 18">
            <path d="M7 1l-6 8 6 8h14V1H7z" fill="none" stroke="currentColor" strokeWidth="1.3"/>
            <path d="M11 6l4 6M15 6l-4 6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
          </svg>
        ) : k === 'globe' ? (
          <svg width="18" height="18" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" strokeWidth="1.3"/>
            <ellipse cx="10" cy="10" rx="4" ry="8" fill="none" stroke="currentColor" strokeWidth="1.2"/>
            <line x1="2" y1="10" x2="18" y2="10" stroke="currentColor" strokeWidth="1.2"/>
          </svg>
        ) : (
          <span>{display}</span>
        )}
        {!isControl && isActive && (
          <div className="key-popup">{display}</div>
        )}
      </button>
    );
  };

  const dashTitle = selectedMsg
    ? `Analysis — "${selectedMsg.text.length > 24 ? selectedMsg.text.slice(0,24)+'...' : selectedMsg.text}"`
    : input.trim() ? 'Live Detection' : 'Detection';

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg className="brand-shield" width="20" height="22" viewBox="0 0 20 22" fill="none">
            <path d="M10 1L2 5v6c0 5.25 3.4 10.2 8 11.1C14.6 21.2 18 16.25 18 11V5L10 1z" stroke="currentColor" strokeWidth="1.5" fill="currentColor" opacity=".12"/>
            <path d="M7 11l2.5 2.5L13.5 9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          SafeType
        </div>
        <div className="topbar-right">
          <div className="status-dot" />
          <span className="model-label">Model</span>
          <select value={model} onChange={e => setModel(e.target.value)}>
            <option value="distilbert">DistilBERT</option>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="naive_baseline">Naive Baseline</option>
          </select>
        </div>
      </header>

      <section className="hero">
        <h1>Keyboard Safety Monitor</h1>
        <p>Real-time content safety detection for children's messaging. Tap a message to inspect its analysis.</p>
      </section>

      <div className="main-grid">
        <div className="phone-col">
          <div className="iphone">
            <div className="iphone-notch" />
            <div className="iphone-screen">
              <div className="ios-statusbar">
                <span>9:41</span>
                <span className="sb-icons">
                  <svg width="16" height="12" viewBox="0 0 16 12"><path d="M1 9h2v3H1zM5 6h2v6H5zM9 3h2v9H9zM13 0h2v12h-2z" fill="currentColor"/></svg>
                  <svg width="16" height="12" viewBox="0 0 16 12"><path d="M8 3a7 7 0 015.5 2.7.5.5 0 01-.7.7A6 6 0 008 4a6 6 0 00-4.8 2.4.5.5 0 01-.7-.7A7 7 0 018 3zm0 3a4 4 0 013.2 1.6.5.5 0 01-.7.7A3 3 0 008 7a3 3 0 00-2.5 1.3.5.5 0 01-.7-.7A4 4 0 018 6zm0 3a1.5 1.5 0 011.1.5.5.5 0 01-.7.7.5.5 0 00-.8 0 .5.5 0 01-.7-.7A1.5 1.5 0 018 9z" fill="currentColor"/></svg>
                  <svg width="24" height="12" viewBox="0 0 24 12"><rect x="0" y="1" width="20" height="10" rx="2" fill="none" stroke="currentColor" strokeWidth="1.2"/><rect x="21" y="3.5" width="2" height="5" rx="1" fill="currentColor"/><rect x="1.5" y="2.5" width="15" height="7" rx="1" fill="currentColor"/></svg>
                </span>
              </div>

              <div className="chat-hdr">
                <div className="chat-back">‹</div>
                <div className="chat-avatar">A</div>
                <div className="chat-info">
                  <div className="chat-name">Alex</div>
                  <div className="chat-sub">online</div>
                </div>
              </div>

              <div className={`safety-bar ${ss.cls}`}>
                <div className="safety-dot" />
                <span>{ss.text}</span>
              </div>

              <div className="messages">
                <div className="msg-time">Today 9:38 AM</div>
                {messages.map(m => {
                  const a = m.analysis;
                  const sev = a ? CAT[a.label]?.sev : null;
                  const dotColor = a ? CAT[a.label]?.color : '#d1d5db';
                  const isSelected = selectedMsgId === m.id;
                  return (
                    <div key={m.id}
                      className={`msg ${m.from} ${isSelected ? 'selected' : ''}`}
                      onClick={() => handleMsgClick(m.id)}>
                      {m.text}
                      <span className="msg-dot"
                        style={{ background: dotColor, borderColor: sev === 'danger' ? '#fff' : 'transparent' }}
                        title={a ? `${a.label} (${(a.confidence*100).toFixed(0)}%)` : 'Analyzing...'}
                      />
                    </div>
                  );
                })}
                <div ref={msgEndRef} />
              </div>

              <div className="input-bar">
                <div className="input-field">
                  {input || <span className="input-ph">iMessage</span>}
                  <span className="cursor-blink">|</span>
                </div>
                <button className={`send-circle ${input.trim()?'active':''}`}
                        onClick={sendMsg} disabled={!input.trim()}>
                  <svg width="16" height="16" viewBox="0 0 16 16">
                    <path d="M8 2l0 12M3 7l5-5 5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
              </div>

              <div className={`keyboard kb-${kbMode}`}>
                {layout.map((row, ri) => (
                  <div key={`${kbMode}-${ri}`} className={`kb-row row-${ri}`}>
                    {row.map(k => renderKey(k, ri))}
                  </div>
                ))}
                <div className="kb-bottom-bar" />
              </div>
            </div>
          </div>
        </div>

        {/* Dashboard */}
        <div className="dashboard">
          <div className="card">
            <h3>{dashTitle}</h3>
            {dashAnalysis ? (
              <div className="live-result">
                <div className="live-dot" style={{background: CAT[dashAnalysis.label]?.color}} />
                <div className="live-label" style={{color: CAT[dashAnalysis.label]?.color}}>
                  {dashAnalysis.label.replace(/_/g,' ')}
                </div>
                <div className="live-conf">{(dashAnalysis.confidence*100).toFixed(1)}% confidence</div>
                {selectedMsg && (
                  <div className="live-text">"{selectedMsg.text}"</div>
                )}
              </div>
            ) : (
              <div className="live-empty">
                {loading ? 'Analyzing...' : 'Click a message or type to see analysis'}
              </div>
            )}
          </div>

          <div className="card">
            <h3>Probabilities</h3>
            <div className="prob-list">
              {LABEL_NAMES.map(n => {
                const p = dashAnalysis?.probabilities?.[n] || 0;
                return (
                  <div key={n} className="prob-row">
                    <span className="prob-dot" style={{background: CAT[n].color}} />
                    <span className="prob-name">{n.replace(/_/g,' ')}</span>
                    <div className="prob-track">
                      <div className="prob-fill" style={{width:`${p*100}%`, background:CAT[n].color}} />
                    </div>
                    <span className="prob-pct">{p ? (p*100).toFixed(1)+'%' : '—'}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card">
            <h3>Alerts</h3>
            <div className="alert-log">
              {alerts.length === 0 ? (
                <div className="alert-empty">No alerts. Flagged messages will appear here.</div>
              ) : alerts.map((a, i) => (
                <div key={i} className="alert-item" onClick={() => setSelectedMsgId(a.msgId)}>
                  <span className="a-dot" style={{background: CAT[a.label]?.color}} />
                  <span className="a-text">{a.text.length > 40 ? a.text.slice(0,40)+'...' : a.text}</span>
                  <span className={`a-badge ${a.sev}`}>{a.label.replace(/_/g,' ')}</span>
                </div>
              ))}
            </div>
          </div>

          {dashAnalysis && (
            <div className={`card edu-card edu-${CAT[dashAnalysis.label]?.sev || 'safe'}`}>
              <h3>Safety Guide</h3>
              <div className="edu-header">
                <span className="edu-badge" style={{background: CAT[dashAnalysis.label]?.color}}>
                  {dashAnalysis.label.replace(/_/g,' ')}
                </span>
                <span className="edu-desc">{CAT[dashAnalysis.label]?.desc}</span>
              </div>
              <div className="edu-explain">{CAT[dashAnalysis.label]?.explain}</div>
              <div className="edu-section">
                <div className="edu-block">
                  <div className="edu-block-title">For Parents</div>
                  <p>{CAT[dashAnalysis.label]?.forParent}</p>
                </div>
                <div className="edu-block">
                  <div className="edu-block-title">For Children</div>
                  <p>{CAT[dashAnalysis.label]?.forChild}</p>
                </div>
              </div>
            </div>
          )}

          <div className="card">
            <h3>How It Works</h3>
            <div className="steps">
              {[
                ['Keyboard Layer','SafeType runs as a custom iOS keyboard extension, monitoring text input across all messaging apps.'],
                ['Real-Time NLP','Every message is analyzed by fine-tuned DistilBERT or Logistic Regression models with sub-second latency.'],
                ['10 Safety Categories','Detects racism, threats, cyberbullying, profanity, sexism, hate speech, toxicity, implicit hate, sarcasm, and clean content.'],
                ['Parent Dashboard','Parents receive real-time alerts when harmful content is detected, with educational guidance on how to respond.'],
              ].map(([title,desc], i) => (
                <div key={i} className="step">
                  <div className="step-num">{i+1}</div>
                  <div><strong>{title}</strong><br/><span className="step-desc">{desc}</span></div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h3>Category Reference</h3>
            <div className="cat-ref">
              {LABEL_NAMES.map(n => (
                <div key={n} className="cat-ref-row">
                  <span className="cat-ref-dot" style={{background: CAT[n].color}} />
                  <span className="cat-ref-name">{n.replace(/_/g,' ')}</span>
                  <span className={`cat-ref-sev ${CAT[n].sev}`}>{CAT[n].sev}</span>
                  <span className="cat-ref-desc">{CAT[n].desc}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <footer className="footer">
        SafeType — AI Keyboard Safety for Children &nbsp;|&nbsp; MSAI 540
      </footer>
    </div>
  );
}

function getKeyDisplay(k, shifted, mode) {
  if (k === 'space')     return 'space';
  if (k === 'return')    return 'return';
  if (k === '123')       return '123';
  if (k === 'ABC')       return 'ABC';
  if (k === '#+=')       return '#+=';
  if (k === 'shift' || k === 'backspace' || k === 'globe') return '';
  if (mode === 'abc')    return shifted ? k.toUpperCase() : k;
  return k;
}
