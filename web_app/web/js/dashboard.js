var SAFETYPE_API = 'https://dl-project-2-second-version.onrender.com/api/predict';

var CAT = {
  clean:         { color:'#16a34a', sev:'safe',   desc:'Safe content',
    explain:'This message contains no harmful content and reflects healthy communication.',
    forParent:'No action needed. Encourage your child to continue communicating respectfully.',
    forChild:'Great job! Respectful conversations build strong friendships.' },
  racism:        { color:'#dc2626', sev:'danger', desc:'Racial discrimination',
    explain:'Language that targets or demeans someone based on their race, ethnicity, or national origin.',
    forParent:'Have an open conversation about why racial stereotypes and slurs cause real harm.',
    forChild:'Words that target someone\'s race can cause deep pain. Everyone is equal.' },
  sexism:        { color:'#e11d48', sev:'danger', desc:'Gender-based hate',
    explain:'Messages containing gender-based discrimination, stereotypes, or harassment.',
    forParent:'Discuss gender equality with your child. Explain that putting someone down because of their gender is never acceptable.',
    forChild:'Everyone deserves respect no matter their gender.' },
  profanity:     { color:'#ea580c', sev:'warn',   desc:'Obscene language',
    explain:'Use of vulgar, obscene, or inappropriate language in conversation.',
    forParent:'Talk about appropriate vs. inappropriate language.',
    forChild:'Using bad words can hurt people\'s feelings. Try expressing yourself with kinder words.' },
  cyberbullying: { color:'#9333ea', sev:'danger', desc:'Personal attacks',
    explain:'Targeted personal attacks, harassment, or intimidation directed at an individual.',
    forParent:'Take this seriously. Ask your child if they are being bullied or bullying others. Save evidence and report.',
    forChild:'If someone is being mean to you online, don\'t respond — tell a parent or trusted adult right away.' },
  toxicity:      { color:'#7c3aed', sev:'warn',   desc:'Toxic language',
    explain:'Generally negative, hostile, or harmful language that creates an unhealthy environment.',
    forParent:'Monitor the conversation pattern. Encourage positive expression.',
    forChild:'Negative words can spread like a virus. Try to be the person who lifts others up.' },
  hate_speech:   { color:'#b91c1c', sev:'danger', desc:'Hate speech',
    explain:'Targeted hatred against a group based on race, religion, gender, or other identity.',
    forParent:'This is a serious concern. Discuss the impact of hate speech with your child. Consider reporting.',
    forChild:'Hating someone for who they are is wrong. If you see hate speech, tell an adult.' },
  implicit_hate: { color:'#c2410c', sev:'warn',   desc:'Indirect / coded hate',
    explain:'Uses coded language or indirect expressions to discriminate against a group.',
    forParent:'This is harder to detect. Learn common coded phrases and discuss with your child.',
    forChild:'Some words seem harmless but actually hurt people in hidden ways. Ask an adult if something feels wrong.' },
  threat:        { color:'#991b1b', sev:'danger', desc:'Threats or violence',
    explain:'Messages containing direct or implied threats of physical harm or violence.',
    forParent:'Take all threats seriously. Save the message, report it, and contact authorities if credible.',
    forChild:'If someone threatens you — even as a joke — tell a parent or teacher immediately.' },
  sarcasm:       { color:'#4f46e5', sev:'info',   desc:'Sarcasm / irony',
    explain:'Detected sarcastic or ironic tone. While not always harmful, it can sometimes mask hostility.',
    forParent:'Sarcasm is normal, but excessive sarcasm directed at someone can be subtle bullying.',
    forChild:'Sarcasm can be funny, but make sure the other person is laughing with you, not being hurt.' }
};

var LABEL_NAMES = ['clean','racism','sexism','profanity','cyberbullying','toxicity','hate_speech','implicit_hate','threat','sarcasm'];

var SafeTypeDashboard = {
  client: null,
  messages: [],
  pageSize: 50,
  currentOffset: 0,
  realtimeChannel: null,
  _analysisInterval: null,
  selectedModel: 'distilbert',

  init: function (supabaseClient) {
    this.client = supabaseClient;
  },

  analyzeMessageDirect: function (text) {
    return fetch(SAFETYPE_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, model: this.selectedModel })
    }).then(function (r) { return r.json(); });
  },

  fetchMessages: function (filters, append) {
    var self = this;
    var query = this.client
      .from('messages')
      .select('*')
      .order('timestamp', { ascending: false })
      .range(
        append ? this.currentOffset : 0,
        (append ? this.currentOffset : 0) + this.pageSize - 1
      );

    if (filters.app && filters.app !== 'all') {
      if (filters.app === 'keyboard') {
        query = query.eq('source_layer', 'keyboard');
      } else {
        query = query.eq('app_source', filters.app);
      }
    }
    if (filters.flag === 'flagged') {
      query = query.eq('is_flagged', true);
    } else if (filters.flag === 'safe') {
      query = query.eq('is_flagged', false);
    }
    if (filters.source && filters.source !== 'all') {
      query = query.eq('source_layer', filters.source);
    }

    return query.then(function (result) {
      if (result.error) throw result.error;
      if (append) {
        self.messages = self.messages.concat(result.data || []);
        self.currentOffset += (result.data || []).length;
      } else {
        self.messages = result.data || [];
        self.currentOffset = self.messages.length;
      }
      return self.messages;
    });
  },

  fetchStats: function () {
    var self = this;
    var todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);

    return this.client
      .from('messages')
      .select('id, is_flagged, app_source', { count: 'exact' })
      .gte('timestamp', todayStart.getTime())
      .then(function (result) {
        var messages = result.data || [];
        var total = messages.length;
        var flagged = messages.filter(function (m) { return m.is_flagged; }).length;
        var appSet = {};
        messages.forEach(function (m) { appSet[m.app_source] = true; });

        return self.client
          .from('messages')
          .select('created_at')
          .order('created_at', { ascending: false })
          .limit(1)
          .then(function (lastResult) {
            var lastMsg = lastResult.data;
            var lastSync = lastMsg && lastMsg.length > 0
              ? self.timeAgo(new Date(lastMsg[0].created_at))
              : '--';
            return { total: total, flagged: flagged, apps: Object.keys(appSet).length, lastSync: lastSync };
          });
      });
  },

  fetchFlaggedAlerts: function () {
    return this.client
      .from('messages')
      .select('*')
      .eq('is_flagged', true)
      .order('timestamp', { ascending: false })
      .limit(10)
      .then(function (result) { return result.data || []; });
  },

  analyzeMessages: function () {
    return this.client.functions.invoke('analyze-messages', { body: {} })
      .then(function (result) {
        if (result.error) throw result.error;
        return result.data;
      });
  },

  subscribeRealtime: function (onNewMessage) {
    this.realtimeChannel = this.client
      .channel('messages-realtime')
      .on('postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'messages' },
        function (payload) { onNewMessage(payload.new); }
      )
      .subscribe();
  },

  unsubscribeRealtime: function () {
    if (this.realtimeChannel) {
      this.client.removeChannel(this.realtimeChannel);
      this.realtimeChannel = null;
    }
  },

  startAutoAnalysis: function (onComplete) {
    var self = this;
    this.analyzeMessages().then(onComplete).catch(function () {});
    this._analysisInterval = setInterval(function () {
      self.analyzeMessages().then(onComplete).catch(function () {});
    }, 60000);
  },

  stopAutoAnalysis: function () {
    if (this._analysisInterval) {
      clearInterval(this._analysisInterval);
      this._analysisInterval = null;
    }
  },

  // ── Rendering ──

  getAppBadge: function (appSource) {
    var map = {
      'com.whatsapp': { label:'WhatsApp', cls:'whatsapp' },
      'com.whatsapp.w4b': { label:'WA Biz', cls:'whatsapp' },
      'com.google.android.apps.messaging': { label:'Messages', cls:'messages' },
      'com.samsung.android.messaging': { label:'Samsung', cls:'messages' },
      'com.android.mms': { label:'MMS', cls:'messages' },
      'com.instagram.android': { label:'Instagram', cls:'instagram' },
      'com.snapchat.android': { label:'Snapchat', cls:'snapchat' },
      'keyboard': { label:'Keyboard', cls:'keyboard' }
    };
    var info = map[appSource] || { label:(appSource||'').split('.').pop(), cls:'other' };
    return '<span class="app-badge ' + info.cls + '">' + info.label + '</span>';
  },

  renderMessageCard: function (msg) {
    var text = this.escapeHtml(msg.text || '');
    var time = new Date(msg.timestamp).toLocaleString();
    var dir = (msg.direction || 'unknown').toLowerCase();
    var dirLabels = { incoming:'IN', outgoing:'OUT', unknown:'--' };

    var dotColor = '#d1d5db';
    var flagHtml = '<span class="flag-badge pending">--</span>';
    var cardClass = 'msg-card';

    if (msg.is_flagged === true) {
      dotColor = '#dc2626';
      var reason = msg.flag_reason || 'flagged';
      var label = reason.split(' (')[0];
      flagHtml = '<span class="flag-badge danger">' + this.escapeHtml(label) + '</span>';
      cardClass += ' flagged-card';
    } else if (msg.is_flagged === false) {
      dotColor = '#16a34a';
      flagHtml = '<span class="flag-badge safe">safe</span>';
    }

    return '<div class="' + cardClass + '" data-msg-id="' + msg.id + '" data-msg-text="' + this.escapeAttr(msg.text || '') + '">' +
      '<span class="msg-sev-dot" style="background:' + dotColor + '"></span>' +
      '<div class="msg-card-body">' +
        '<div class="msg-card-text">' + text + '</div>' +
        '<div class="msg-card-meta">' +
          this.getAppBadge(msg.app_source) +
          '<span class="dir-badge ' + dir + '">' + (dirLabels[dir] || '--') + '</span>' +
          flagHtml +
          '<span>' + this.escapeHtml(msg.sender || '') + '</span>' +
          '<span>' + time + '</span>' +
        '</div>' +
      '</div>' +
    '</div>';
  },

  renderAlertItem: function (msg) {
    var text = this.escapeHtml(msg.text || '');
    var reason = msg.flag_reason || 'flagged';
    var label = reason.split(' (')[0];
    return '<div class="alert-item" data-msg-text="' + this.escapeAttr(msg.text || '') + '">' +
      '<span class="a-dot"></span>' +
      '<span class="a-text">' + (text.length > 60 ? text.slice(0,60) + '...' : text) + '</span>' +
      '<span class="a-reason">' + this.escapeHtml(label) + '</span>' +
    '</div>';
  },

  renderDetection: function (data) {
    if (!data || data.error) return '<div class="live-empty">Click a message to see detailed analysis</div>';
    var cat = CAT[data.label] || {};
    return '<div class="live-result">' +
      '<div class="live-dot" style="background:' + (cat.color || '#d1d5db') + '"></div>' +
      '<div class="live-label" style="color:' + (cat.color || '#374151') + '">' + (data.label || '').replace(/_/g,' ') + '</div>' +
      '<div class="live-conf">' + ((data.confidence || 0) * 100).toFixed(1) + '% confidence</div>' +
    '</div>';
  },

  renderProbabilities: function (data) {
    if (!data || !data.probabilities) {
      return LABEL_NAMES.map(function (n) {
        var c = CAT[n] || {};
        return '<div class="prob-row">' +
          '<span class="prob-dot" style="background:' + (c.color||'#ccc') + '"></span>' +
          '<span class="prob-name">' + n.replace(/_/g,' ') + '</span>' +
          '<div class="prob-track"><div class="prob-fill" style="width:0;background:' + (c.color||'#ccc') + '"></div></div>' +
          '<span class="prob-pct">&mdash;</span></div>';
      }).join('');
    }
    return LABEL_NAMES.map(function (n) {
      var p = data.probabilities[n] || 0;
      var c = CAT[n] || {};
      return '<div class="prob-row">' +
        '<span class="prob-dot" style="background:' + (c.color||'#ccc') + '"></span>' +
        '<span class="prob-name">' + n.replace(/_/g,' ') + '</span>' +
        '<div class="prob-track"><div class="prob-fill" style="width:' + (p*100) + '%;background:' + (c.color||'#ccc') + '"></div></div>' +
        '<span class="prob-pct">' + (p ? (p*100).toFixed(1) + '%' : '&mdash;') + '</span></div>';
    }).join('');
  },

  renderSafetyGuide: function (data) {
    if (!data || data.error) return '';
    var cat = CAT[data.label] || {};
    if (!cat.explain) return '';
    var sevClass = 'edu-' + (cat.sev || 'safe');
    return '<div class="card edu-card ' + sevClass + '">' +
      '<h3>Safety Guide</h3>' +
      '<div class="edu-header">' +
        '<span class="edu-badge" style="background:' + (cat.color||'#666') + '">' + (data.label||'').replace(/_/g,' ') + '</span>' +
        '<span class="edu-desc">' + (cat.desc||'') + '</span>' +
      '</div>' +
      '<div class="edu-explain">' + cat.explain + '</div>' +
      '<div class="edu-section">' +
        '<div class="edu-block"><div class="edu-block-title">For Parents</div><p>' + (cat.forParent||'') + '</p></div>' +
        '<div class="edu-block"><div class="edu-block-title">For Children</div><p>' + (cat.forChild||'') + '</p></div>' +
      '</div>' +
    '</div>';
  },

  renderCategoryRef: function () {
    return LABEL_NAMES.map(function (n) {
      var c = CAT[n] || {};
      return '<div class="cat-ref-row">' +
        '<span class="cat-ref-dot" style="background:' + (c.color||'#ccc') + '"></span>' +
        '<span class="cat-ref-name">' + n.replace(/_/g,' ') + '</span>' +
        '<span class="cat-ref-sev ' + (c.sev||'safe') + '">' + (c.sev||'safe') + '</span>' +
        '<span class="cat-ref-desc">' + (c.desc||'') + '</span>' +
      '</div>';
    }).join('');
  },

  // ── Utils ──

  escapeHtml: function (str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },

  escapeAttr: function (str) {
    return str.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  },

  timeAgo: function (date) {
    var s = Math.floor((Date.now() - date.getTime()) / 1000);
    if (s < 60) return 'just now';
    if (s < 3600) return Math.floor(s/60) + 'm ago';
    if (s < 86400) return Math.floor(s/3600) + 'h ago';
    return Math.floor(s/86400) + 'd ago';
  }
};
