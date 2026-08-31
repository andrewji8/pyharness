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
        btnClearSessions: document.getElementById('btn-clear-sessions'),
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
                if (data.internal) break;
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
                if (data.internal) break;
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

            case 'subagent_start':
            case 'subagent_complete':
                updateSubagentStatus(data);
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
                <span class="plan-step-title">${escapeHtml(step.title || step.description || step.id)}</span>
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
        const name = data.spec?.name || data.name || 'subagent';
        const escapedName = escapeHtml(name);
        const statusClass = data.status === 'ok' ? 'running' : 'failed';
        const statusText = data.status === 'ok' ? 'running' : (data.status === 'timeout' ? 'timeout' : 'failed');

        let item = Array.from(dom.subagentList.children).find(
            el => el.dataset.subagentName === name
        );

        if (!item) {
            item = document.createElement('div');
            item.className = 'subagent-item';
            item.dataset.subagentName = name;
            item.innerHTML = `
                <span class="subagent-status ${statusClass}"></span>
                <span class="subagent-name">${escapedName}</span>
            `;
            dom.subagentList.appendChild(item);
        } else {
            const statusEl = item.querySelector('.subagent-status');
            if (statusEl) {
                statusEl.className = 'subagent-status ' + statusClass;
                statusEl.title = statusText;
            }
        }
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
                <button class="session-item-menu-btn" data-session-id="${session.session_id}">⋯</button>
            `;
            const menuBtn = div.querySelector('.session-item-menu-btn');
            menuBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                showSessionContextMenu(e, session);
            });
            div.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                showSessionContextMenu(e, session);
            });
            div.addEventListener('click', (e) => {
                if (e.target.classList.contains('session-item-menu-btn')) return;
                loadSession(session.session_id);
            });
            dom.sessionList.appendChild(div);
        });
    }

    function showSessionContextMenu(event, session) {
        hideSessionContextMenu();
        const menu = document.createElement('div');
        menu.className = 'session-context-menu';
        menu.style.left = event.clientX + 'px';
        menu.style.top = event.clientY + 'px';
        menu.innerHTML = `
            <div class="session-context-menu-item" data-action="download-md">下载 Markdown</div>
            <div class="session-context-menu-item" data-action="export-json">导出 JSON</div>
            <div class="session-context-menu-item" data-action="copy">复制对话</div>
            <div class="session-context-menu-item" data-action="rename">重命名</div>
            <div class="session-context-menu-item danger" data-action="delete">删除</div>
        `;
        document.body.appendChild(menu);
        menu.querySelectorAll('.session-context-menu-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                const action = item.dataset.action;
                hideSessionContextMenu();
                handleSessionAction(action, session);
            });
        });
        document.addEventListener('click', hideSessionContextMenu);
        document.addEventListener('contextmenu', hideSessionContextMenu);
    }

    function hideSessionContextMenu() {
        const existing = document.querySelector('.session-context-menu');
        if (existing) existing.remove();
        document.removeEventListener('click', hideSessionContextMenu);
        document.removeEventListener('contextmenu', hideSessionContextMenu);
    }

    async function handleSessionAction(action, session) {
        switch (action) {
            case 'download-md':
                await downloadSessionMarkdown(session);
                break;
            case 'export-json':
                exportSessionJson(session);
                break;
            case 'copy':
                await copySessionToClipboard(session);
                break;
            case 'rename':
                await renameSession(session);
                break;
            case 'delete':
                await deleteSession(session);
                break;
        }
    }

    async function downloadSessionMarkdown(session) {
        try {
            const res = await fetch(`/api/sessions/${session.session_id}`);
            const data = await res.json();
            if (data.error) {
                addSystemMessage(`加载会话失败: ${data.error}`);
                return;
            }
            const lines = [`# 会话 ${session.session_id.slice(0, 8)}`, ''];
            (data.messages || []).forEach(msg => {
                if (msg.role === 'user') {
                    lines.push('## 用户');
                    lines.push(msg.content);
                } else if (msg.role === 'assistant') {
                    lines.push('## 助手');
                    lines.push(msg.content);
                } else if (msg.role === 'tool') {
                    lines.push('## 工具: ' + (msg.name || ''));
                    lines.push(msg.content);
                }
                lines.push('');
            });
            const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const date = new Date().toISOString().slice(0, 10);
            a.download = `pyharness_${session.session_id.slice(0, 8)}_${date}.md`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error('Failed to download markdown:', e);
        }
    }

    function exportSessionJson(session) {
        const a = document.createElement('a');
        a.href = `/api/sessions/${session.session_id}`;
        a.download = `pyharness_${session.session_id.slice(0, 8)}.json`;
        a.click();
    }

    async function copySessionToClipboard(session) {
        try {
            const res = await fetch(`/api/sessions/${session.session_id}`);
            const data = await res.json();
            if (data.error) {
                addSystemMessage(`加载会话失败: ${data.error}`);
                return;
            }
            const lines = [`# 会话 ${session.session_id.slice(0, 8)}`, ''];
            (data.messages || []).forEach(msg => {
                if (msg.role === 'user') {
                    lines.push('## 用户');
                    lines.push(msg.content);
                } else if (msg.role === 'assistant') {
                    lines.push('## 助手');
                    lines.push(msg.content);
                } else if (msg.role === 'tool') {
                    lines.push('## 工具: ' + (msg.name || ''));
                    lines.push(msg.content);
                }
                lines.push('');
            });
            const text = lines.join('\n');
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                const textarea = document.createElement('textarea');
                textarea.value = text;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
            }
            showToast('已复制');
        } catch (e) {
            console.error('Failed to copy:', e);
        }
    }

    async function renameSession(session) {
        const title = prompt('输入新名称:', session.title || '');
        if (title === null) return;
        const trimmed = title.trim();
        if (!trimmed) return;
        try {
            const res = await fetch(`/api/sessions/${session.session_id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: trimmed }),
            });
            const data = await res.json();
            if (data.error) {
                addSystemMessage(`重命名失败: ${data.error}`);
                return;
            }
            addSystemMessage(`已重命名为: ${trimmed}`);
            refreshSessionList();
        } catch (e) {
            console.error('Failed to rename:', e);
        }
    }

    async function deleteSession(session) {
        if (!confirm('确定要删除此会话吗？此操作不可恢复。')) return;
        try {
            const res = await fetch(`/api/sessions/${session.session_id}`, {
                method: 'DELETE',
            });
            const data = await res.json();
            if (data.deleted) {
                addSystemMessage('会话已删除');
                if (state.currentSessionId === session.session_id) {
                    state.currentSessionId = null;
                    dom.chatMessages.innerHTML = '';
                    updateChatHeader(null);
                }
                refreshSessionList();
            } else {
                addSystemMessage(data.error || '删除失败');
            }
        } catch (e) {
            console.error('Failed to delete:', e);
        }
    }

    function showToast(message) {
        let toast = document.querySelector('.toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.className = 'toast';
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 2000);
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

        dom.btnClearSessions.addEventListener('click', clearAllSessions);
    }

    async function clearAllSessions() {
        if (!confirm('确定要清空所有历史记录吗？此操作不可恢复！')) return;
        try {
            const res = await fetch('/api/sessions', { method: 'DELETE' });
            const data = await res.json();
            if (data.status === 'ok') {
                showToast(`已清空 ${data.cleared_count} 条会话`);
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
                refreshSessionList();
            } else {
                showToast('清空失败');
            }
        } catch (e) {
            console.error('Failed to clear sessions:', e);
        }
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
