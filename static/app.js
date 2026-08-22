/* ==========================================================================
   PyHarness Web UI — Frontend Logic
   ========================================================================== */

(function () {
    'use strict';

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    const state = {
        ws: null,
        sessionId: null,
        planId: null,
        sessions: [],
        currentSessionId: null,
        isStreaming: false,
        currentStreamElement: null,
    };

    // ------------------------------------------------------------------
    // DOM refs
    // ------------------------------------------------------------------
    const dom = {
        sessionList: document.getElementById('session-list'),
        chatMessages: document.getElementById('chat-messages'),
        chatHeader: document.getElementById('chat-header'),
        userInput: document.getElementById('user-input'),
        btnSend: document.getElementById('btn-send'),
        btnNewSession: document.getElementById('btn-new-session'),
        btnSearch: document.getElementById('btn-search'),
        searchModal: document.getElementById('search-modal'),
        searchInput: document.getElementById('search-input'),
        btnDoSearch: document.getElementById('btn-do-search'),
        searchResults: document.getElementById('search-results'),
        btnCloseSearch: document.getElementById('btn-close-search'),
        planEmpty: document.getElementById('plan-empty'),
        planContent: document.getElementById('plan-content'),
        planProgressFill: document.getElementById('plan-progress-fill'),
        planSteps: document.getElementById('plan-steps'),
        subagentEmpty: document.getElementById('subagent-empty'),
        subagentList: document.getElementById('subagent-list'),
    };

    // ------------------------------------------------------------------
    // WebSocket
    // ------------------------------------------------------------------
    function connectWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        state.ws = new WebSocket(`${protocol}//${location.host}/ws/events`);

        state.ws.onopen = () => {
            console.log('WebSocket connected');
            addSystemMessage('已连接到服务器');
        };

        state.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleServerMessage(msg);
        };

        state.ws.onclose = () => {
            console.log('WebSocket disconnected');
            addSystemMessage('连接已断开，尝试重连...');
            setTimeout(connectWebSocket, 3000);
        };

        state.ws.onerror = (err) => {
            console.error('WebSocket error:', err);
        };
    }

    function sendMessage(type, data = {}) {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ type, ...data }));
        }
    }

    // ------------------------------------------------------------------
    // Message handling
    // ------------------------------------------------------------------
    function handleServerMessage(msg) {
        const eventType = msg.type;
        const data = msg.data || {};

        switch (eventType) {
            case 'user.message':
                appendUserMessage(data.text || '');
                break;

            case 'assistant.finished':
                finishStreaming();
                break;

            case 'tool.called':
                showToolCallStart(data);
                break;

            case 'tool.result':
                showToolResult(data);
                break;

            case 'llm_stream_chunk':
                handleStreamChunk(data);
                break;

            case 'plan_created':
                initPlanPanel(data);
                break;

            case 'plan_step_start':
                updateStepStatus(data.step_id, 'running', data.step_title);
                break;

            case 'plan_step_complete':
                updateStepStatus(data.step_id, data.step_status, data.step_title);
                break;

            case 'plan_completed':
                showPlanSummary(data);
                break;

            case 'session.started':
                state.sessionId = data.session_id;
                updateChatHeader(data.session_id);
                break;

            case 'session.finished':
            case 'session_end':
                addSystemMessage(`会话结束 (${data.total_turns || '?'} 轮)`);
                refreshSessionList();
                break;

            case 'error':
                addSystemMessage(`错误: ${data.message}`);
                finishStreaming();
                break;

            default:
                // Unknown event, ignore
                break;
        }
    }

    // ------------------------------------------------------------------
    // Chat UI
    // ------------------------------------------------------------------
    function appendUserMessage(text) {
        removeWelcome();
        const div = document.createElement('div');
        div.className = 'message user';
        div.textContent = text;
        dom.chatMessages.appendChild(div);
        scrollToBottom();
    }

    function appendAssistantMessage(text) {
        removeWelcome();
        const div = document.createElement('div');
        div.className = 'message assistant';
        div.textContent = text;
        dom.chatMessages.appendChild(div);
        scrollToBottom();
        return div;
    }

    function addSystemMessage(text) {
        const div = document.createElement('div');
        div.className = 'message system';
        div.textContent = text;
        dom.chatMessages.appendChild(div);
        scrollToBottom();
    }

    function removeWelcome() {
        const welcome = dom.chatMessages.querySelector('.welcome');
        if (welcome) welcome.remove();
    }

    function scrollToBottom() {
        dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    }

    // ------------------------------------------------------------------
    // Streaming
    // ------------------------------------------------------------------
    function startStreaming() {
        state.isStreaming = true;
        removeWelcome();
        const div = document.createElement('div');
        div.className = 'message assistant';
        const contentSpan = document.createElement('span');
        contentSpan.className = 'typing-cursor';
        div.appendChild(contentSpan);
        dom.chatMessages.appendChild(div);
        state.currentStreamElement = contentSpan;
        scrollToBottom();
    }

    function handleStreamChunk(data) {
        if (!state.isStreaming) {
            startStreaming();
        }
        if (state.currentStreamElement && data.content) {
            state.currentStreamElement.textContent += data.content;
            scrollToBottom();
        }
        if (data.is_finished) {
            finishStreaming();
        }
    }

    function finishStreaming() {
        state.isStreaming = false;
        if (state.currentStreamElement) {
            state.currentStreamElement.classList.remove('typing-cursor');
            state.currentStreamElement = null;
        }
        scrollToBottom();
    }

    // ------------------------------------------------------------------
    // Tool Calls
    // ------------------------------------------------------------------
    function showToolCallStart(data) {
        removeWelcome();
        const card = document.createElement('div');
        card.className = 'tool-call-card';
        card.innerHTML = `
            <div class="tool-call-header">🔧 调用工具: ${escapeHtml(data.tool || data.name || '?')}</div>
            <div class="tool-call-args">${escapeHtml(data.arguments || '')}</div>
        `;
        dom.chatMessages.appendChild(card);
        scrollToBottom();
    }

    function showToolResult(data) {
        const isError = data.is_success === false || data.status === 'error';
        const card = document.createElement('div');
        card.className = 'tool-result-card' + (isError ? ' tool-result-error' : '');
        card.innerHTML = `
            <div class="tool-result-header">${isError ? '❌' : '✅'} ${escapeHtml(data.tool || data.name || '?')}</div>
            <div class="tool-call-args">${escapeHtml(data.output || data.result || '')}</div>
        `;
        dom.chatMessages.appendChild(card);
        scrollToBottom();
    }

    // ------------------------------------------------------------------
    // Plan Panel
    // ------------------------------------------------------------------
    function initPlanPanel(data) {
        state.planId = data.plan_id;
        dom.planEmpty.style.display = 'none';
        dom.planContent.style.display = 'block';
        renderPlanSteps(data.steps || [], data.progress || 0);
    }

    function renderPlanSteps(steps, progress) {
        dom.planProgressFill.style.width = `${Math.round((progress || 0) * 100)}%`;
        dom.planSteps.innerHTML = '';
        steps.forEach(step => {
            const div = document.createElement('div');
            div.className = 'plan-step';
            div.id = `plan-step-${step.id}`;
            const icon = getStepIcon(step.status);
            div.innerHTML = `
                <span class="plan-step-icon">${icon}</span>
                <span class="plan-step-title">${escapeHtml(step.title || step.id)}</span>
                <span class="plan-step-status ${step.status}">${step.status}</span>
            `;
            dom.planSteps.appendChild(div);
        });
    }

    function updateStepStatus(stepId, status, title) {
        const stepEl = document.getElementById(`plan-step-${stepId}`);
        if (!stepEl) {
            // Step not yet rendered, re-render all steps if we have a plan
            // For simplicity, just update the panel
            return;
        }
        const icon = getStepIcon(status);
        stepEl.querySelector('.plan-step-icon').textContent = icon;
        const statusEl = stepEl.querySelector('.plan-step-status');
        statusEl.className = `plan-step-status ${status}`;
        statusEl.textContent = status;
        if (title) {
            stepEl.querySelector('.plan-step-title').textContent = title;
        }
    }

    function getStepIcon(status) {
        switch (status) {
            case 'completed': return '✅';
            case 'running': return '🔄';
            case 'pending': return '⏳';
            case 'failed': return '❌';
            case 'skipped': return '⏭️';
            default: return '⏳';
        }
    }

    function showPlanSummary(data) {
        const icon = data.final_status === 'completed' ? '🎉' : '⚠️';
        addSystemMessage(`${icon} 计划 ${data.final_status || 'completed'} (进度: ${Math.round((data.progress || 1) * 100)}%)`);
    }

    // ------------------------------------------------------------------
    // Subagent Panel
    // ------------------------------------------------------------------
    function updateSubagentStatus(data) {
        dom.subagentEmpty.style.display = 'none';
        const item = document.createElement('div');
        item.className = 'subagent-item';
        const statusClass = data.status === 'ok' ? 'running' : 'failed';
        item.innerHTML = `
            <span class="subagent-status ${statusClass}"></span>
            <span class="subagent-name">${escapeHtml(data.spec?.name || data.name || 'subagent')}</span>
        `;
        dom.subagentList.appendChild(item);
    }

    // ------------------------------------------------------------------
    // Sessions
    // ------------------------------------------------------------------
    async function refreshSessionList() {
        try {
            const res = await fetch('/api/sessions');
            const data = await res.json();
            state.sessions = data.sessions || [];
            renderSessionList();
        } catch (e) {
            console.error('Failed to load sessions:', e);
        }
    }

    function renderSessionList() {
        dom.sessionList.innerHTML = '';
        if (state.sessions.length === 0) {
            dom.sessionList.innerHTML = '<div class="loading">暂无会话</div>';
            return;
        }
        state.sessions.forEach(session => {
            const div = document.createElement('div');
            div.className = 'session-item' + (session.session_id === state.currentSessionId ? ' active' : '');
            const date = new Date(session.created_at).toLocaleString('zh-CN');
            div.innerHTML = `
                <div class="session-item-title">会话 ${session.session_id.slice(0, 8)}</div>
                <div class="session-item-meta">${date} · ${session.message_count || 0} 条消息</div>
            `;
            div.addEventListener('click', () => loadSession(session.session_id));
            dom.sessionList.appendChild(div);
        });
    }

    async function loadSession(sessionId) {
        state.currentSessionId = sessionId;
        renderSessionList();
        try {
            const res = await fetch(`/api/sessions/${sessionId}`);
            const data = await res.json();
            if (data.error) {
                addSystemMessage(`加载会话失败: ${data.error}`);
                return;
            }
            dom.chatMessages.innerHTML = '';
            (data.messages || []).forEach(msg => {
                if (msg.role === 'user') {
                    appendUserMessage(msg.content);
                } else if (msg.role === 'assistant') {
                    appendAssistantMessage(msg.content);
                } else if (msg.role === 'tool') {
                    showToolCallStart({ name: msg.name, arguments: '' });
                    showToolResult({ name: msg.name, output: msg.content, is_success: true });
                }
            });
            updateChatHeader(sessionId);
        } catch (e) {
            console.error('Failed to load session:', e);
        }
    }

    function updateChatHeader(sessionId) {
        dom.chatHeader.innerHTML = `<span>会话: ${sessionId ? sessionId.slice(0, 8) + '...' : '新会话'}</span>`;
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------
    function openSearch() {
        dom.searchModal.style.display = 'flex';
        dom.searchInput.focus();
    }

    function closeSearch() {
        dom.searchModal.style.display = 'none';
        dom.searchInput.value = '';
        dom.searchResults.innerHTML = '';
    }

    async function doSearch() {
        const query = dom.searchInput.value.trim();
        const sessionId = state.currentSessionId;
        if (!query) return;

        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, session_id: sessionId, limit: 10 }),
            });
            const data = await res.json();
            renderSearchResults(data);
        } catch (e) {
            console.error('Search failed:', e);
        }
    }

    function renderSearchResults(data) {
        dom.searchResults.innerHTML = '';
        if (data.count === 0) {
            dom.searchResults.innerHTML = '<div class="loading">未找到结果</div>';
            return;
        }
        data.results.forEach(r => {
            const div = document.createElement('div');
            div.className = 'search-result-item';
            div.innerHTML = `
                <div class="role">${escapeHtml(r.role)}</div>
                <div class="content">${escapeHtml(r.content)}</div>
                ${r.snippet ? `<div class="snippet">${escapeHtml(r.snippet)}</div>` : ''}
            `;
            dom.searchResults.appendChild(div);
        });
    }

    // ------------------------------------------------------------------
    // Event Listeners
    // ------------------------------------------------------------------
    function setupEventListeners() {
        dom.btnSend.addEventListener('click', sendUserMessage);
        dom.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendUserMessage();
            }
        });

        dom.btnNewSession.addEventListener('click', () => {
            state.currentSessionId = null;
            dom.chatMessages.innerHTML = `
                <div class="welcome">
                    <h2>👋 欢迎使用 PyHarness</h2>
                    <p>发送消息开始与 Agent 对话</p>
                </div>
            `;
            dom.planEmpty.style.display = 'block';
            dom.planContent.style.display = 'none';
            dom.subagentEmpty.style.display = 'block';
            dom.subagentList.innerHTML = '';
            updateChatHeader(null);
            renderSessionList();
        });

        dom.btnSearch.addEventListener('click', openSearch);
        dom.btnCloseSearch.addEventListener('click', closeSearch);
        dom.btnDoSearch.addEventListener('click', doSearch);
        dom.searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') doSearch();
        });
        dom.searchModal.addEventListener('click', (e) => {
            if (e.target === dom.searchModal) closeSearch();
        });
    }

    function sendUserMessage() {
        const text = dom.userInput.value.trim();
        if (!text || state.isStreaming) return;

        dom.userInput.value = '';
        sendMessage('user_message', {
            content: text,
            session_id: state.currentSessionId,
        });
    }

    // ------------------------------------------------------------------
    // Utils
    // ------------------------------------------------------------------
    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------
    function init() {
        setupEventListeners();
        connectWebSocket();
        refreshSessionList();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
