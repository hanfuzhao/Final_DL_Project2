/**
 * Main app controller with hash-based routing.
 */
(function () {
    var screens = ['landing', 'setup', 'login', 'dashboard'];
    var client = SafeTypeConfig.getSupabaseClient();

    function navigateTo(screen) {
        screens.forEach(function (s) {
            document.getElementById(s + '-screen').classList.remove('active');
        });
        document.getElementById(screen + '-screen').classList.add('active');
        if (window.location.hash !== '#' + screen) {
            window.location.hash = screen;
        }
        window.scrollTo(0, 0);
    }

    function handleRoute() {
        var hash = window.location.hash.slice(1);
        if (hash === 'dashboard') {
            var session = SafeTypeAuth.getSession();
            if (session) {
                navigateTo('dashboard');
                loadDashboard();
            } else {
                navigateTo('login');
            }
        } else if (screens.indexOf(hash) !== -1) {
            navigateTo(hash);
        } else {
            navigateTo('landing');
        }
    }

    function boot() {
        SafeTypeAuth.init();
        SafeTypeDashboard.init(client);

        document.getElementById('cat-ref').innerHTML = SafeTypeDashboard.renderCategoryRef();
        document.getElementById('prob-bars').innerHTML = SafeTypeDashboard.renderProbabilities(null);

        document.querySelectorAll('a[href^="#"]').forEach(function (link) {
            link.addEventListener('click', function (e) {
                var target = this.getAttribute('href').slice(1);
                if (screens.indexOf(target) !== -1) {
                    e.preventDefault();
                    if (target === 'dashboard') {
                        var session = SafeTypeAuth.getSession();
                        if (session) {
                            navigateTo('dashboard');
                            loadDashboard();
                        } else {
                            navigateTo('login');
                        }
                    } else {
                        navigateTo(target);
                    }
                }
            });
        });

        window.addEventListener('hashchange', handleRoute);
        handleRoute();
    }

    // ── Analysis UI ──

    function showAnalyzeStatus(text, isError) {
        var el = document.getElementById('analyze-status');
        el.textContent = text;
        el.className = 'analyze-status' + (isError ? ' error' : ' success');
        setTimeout(function () { el.textContent = ''; }, 4000);
    }

    function runAnalysis() {
        var btn = document.getElementById('btn-analyze');
        btn.disabled = true;
        btn.textContent = 'Analyzing...';
        SafeTypeDashboard.analyzeMessages()
            .then(function (result) {
                if (result && result.analyzed !== undefined) {
                    showAnalyzeStatus('Analyzed ' + result.analyzed + ', flagged ' + result.flagged, false);
                    if (result.flagged > 0) loadDashboard();
                } else {
                    showAnalyzeStatus('Analysis complete', false);
                }
            })
            .catch(function () {
                showAnalyzeStatus('Analysis failed', true);
            })
            .finally(function () {
                btn.disabled = false;
                btn.textContent = 'Analyze Now';
            });
    }

    function analyzeAndShowMessage(text) {
        var detEl = document.getElementById('detection-result');
        var probEl = document.getElementById('prob-bars');
        var guideEl = document.getElementById('safety-guide-container');

        detEl.innerHTML = '<div class="live-empty">Analyzing...</div>';

        SafeTypeDashboard.analyzeMessageDirect(text)
            .then(function (data) {
                detEl.innerHTML = SafeTypeDashboard.renderDetection(data);
                probEl.innerHTML = SafeTypeDashboard.renderProbabilities(data);
                guideEl.innerHTML = SafeTypeDashboard.renderSafetyGuide(data);
            })
            .catch(function () {
                detEl.innerHTML = '<div class="live-empty">Analysis failed. API may be waking up — try again in a moment.</div>';
            });
    }

    // ── Dashboard ──

    var dashboardLoaded = false;
    var selectedMsgId = null;

    function loadDashboard() {
        SafeTypeDashboard.fetchStats()
            .then(function (stats) {
                document.getElementById('stat-total').textContent = stats.total;
                document.getElementById('stat-flagged').textContent = stats.flagged;
                document.getElementById('stat-apps').textContent = stats.apps;
                document.getElementById('stat-last-sync').textContent = stats.lastSync;

                var badge = document.getElementById('device-status');
                if (stats.lastSync !== '--' && stats.lastSync.indexOf('d ago') === -1) {
                    badge.textContent = 'Online';
                    badge.className = 'status-pill online';
                } else {
                    badge.textContent = 'Offline';
                    badge.className = 'status-pill offline';
                }
            });

        SafeTypeDashboard.fetchFlaggedAlerts()
            .then(function (flagged) {
                var section = document.getElementById('alerts-section');
                var list = document.getElementById('alerts-list');
                if (flagged.length > 0) {
                    section.style.display = 'block';
                    list.innerHTML = flagged.map(function (m) {
                        return SafeTypeDashboard.renderAlertItem(m);
                    }).join('');
                    bindAlertClicks();
                } else {
                    section.style.display = 'none';
                }
            });

        loadMessages();

        if (!dashboardLoaded) {
            dashboardLoaded = true;

            SafeTypeDashboard.subscribeRealtime(function (newMsg) {
                var container = document.getElementById('messages-list');
                var empty = container.querySelector('.msg-empty');
                if (empty) container.innerHTML = '';
                container.insertAdjacentHTML('afterbegin', SafeTypeDashboard.renderMessageCard(newMsg));
                bindMsgClicks();

                var totalEl = document.getElementById('stat-total');
                totalEl.textContent = parseInt(totalEl.textContent) + 1;
                if (newMsg.is_flagged) {
                    var flagEl = document.getElementById('stat-flagged');
                    flagEl.textContent = parseInt(flagEl.textContent) + 1;
                }
                document.getElementById('stat-last-sync').textContent = 'just now';
                var db = document.getElementById('device-status');
                db.textContent = 'Online';
                db.className = 'status-pill online';
            });

            SafeTypeDashboard.startAutoAnalysis(function (result) {
                if (result && result.flagged > 0) loadDashboard();
            });
        }
    }

    function loadMessages(append) {
        var filters = {
            app: document.getElementById('filter-app').value,
            flag: document.getElementById('filter-flag').value,
            source: document.getElementById('filter-source').value,
        };

        SafeTypeDashboard.fetchMessages(filters, append)
            .then(function (messages) {
                var container = document.getElementById('messages-list');
                var loadMoreBtn = document.getElementById('btn-load-more');

                if (messages.length === 0 && !append) {
                    container.innerHTML = '<div class="msg-empty">No messages yet. Waiting for device sync...</div>';
                } else {
                    container.innerHTML = messages.map(function (m) {
                        return SafeTypeDashboard.renderMessageCard(m);
                    }).join('');
                }

                loadMoreBtn.style.display = messages.length >= SafeTypeDashboard.pageSize ? 'inline-block' : 'none';
                bindMsgClicks();
            });
    }

    function bindMsgClicks() {
        document.querySelectorAll('.msg-card').forEach(function (card) {
            card.addEventListener('click', function () {
                document.querySelectorAll('.msg-card.selected').forEach(function (c) { c.classList.remove('selected'); });
                this.classList.add('selected');
                var text = this.getAttribute('data-msg-text');
                selectedMsgId = this.getAttribute('data-msg-id');
                if (text) analyzeAndShowMessage(text);
            });
        });
    }

    function bindAlertClicks() {
        document.querySelectorAll('.alert-item').forEach(function (item) {
            item.addEventListener('click', function () {
                var text = this.getAttribute('data-msg-text');
                if (text) analyzeAndShowMessage(text);
            });
        });
    }

    // ── Event listeners ──

    document.getElementById('login-form').addEventListener('submit', function (e) {
        e.preventDefault();
        SafeTypeAuth.signIn(
            document.getElementById('email').value,
            document.getElementById('password').value
        ).then(function () {
            navigateTo('dashboard');
            loadDashboard();
        }).catch(function (err) {
            var el = document.getElementById('login-error');
            el.textContent = err.message || 'Sign in failed';
            el.style.display = 'block';
            setTimeout(function () { el.style.display = 'none'; }, 5000);
        });
    });

    document.getElementById('btn-logout').addEventListener('click', function () {
        SafeTypeDashboard.unsubscribeRealtime();
        SafeTypeDashboard.stopAutoAnalysis();
        dashboardLoaded = false;
        SafeTypeAuth.signOut();
        navigateTo('landing');
    });

    ['filter-app', 'filter-flag', 'filter-source'].forEach(function (id) {
        document.getElementById(id).addEventListener('change', function () { loadMessages(); });
    });

    document.getElementById('btn-refresh').addEventListener('click', function () { loadDashboard(); });
    document.getElementById('btn-analyze').addEventListener('click', function () { runAnalysis(); });
    document.getElementById('btn-load-more').addEventListener('click', function () { loadMessages(true); });

    document.getElementById('model-select').addEventListener('change', function () {
        SafeTypeDashboard.selectedModel = this.value;
        if (selectedMsgId) {
            var selectedCard = document.querySelector('.msg-card.selected');
            if (selectedCard) {
                var text = selectedCard.getAttribute('data-msg-text');
                if (text) analyzeAndShowMessage(text);
            }
        }
    });

    boot();
})();
