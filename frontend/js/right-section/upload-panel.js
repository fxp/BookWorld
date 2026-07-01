/**
 * UploadPanel — lets users paste or upload a story text to generate a new world.
 *
 * Usage: new UploadPanel(onSuccessCallback)
 * The callback receives the preset filename when processing is done.
 */
class UploadPanel {
    constructor(onSuccess) {
        this.onSuccess = onSuccess || (() => {});
        this._pollTimer = null;
        this._buildDOM();
        this._bindEvents();
    }

    /* ---- DOM construction ---- */

    _buildDOM() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'upload-modal-overlay';
        this.overlay.innerHTML = `
<div class="upload-modal">
  <h3 data-i18n="uploadStoryTitle">上传故事 / Upload Story</h3>

  <div class="upload-modal-row">
    <div>
      <label data-i18n="uploadStoryTitleLabel">故事标题 / Title</label>
      <input type="text" id="up-title" placeholder="e.g. 红楼梦 / Dream of Red Mansions" />
    </div>
    <div style="flex:0 0 110px">
      <label data-i18n="uploadLang">语言 / Language</label>
      <select id="up-lang">
        <option value="zh">中文</option>
        <option value="en">English</option>
      </select>
    </div>
  </div>

  <div>
    <label data-i18n="uploadTextLabel">粘贴故事内容 / Paste story text</label>
    <textarea id="up-text" placeholder="粘贴小说正文（前几章即可）…"></textarea>
    <div class="upload-modal-hint" data-i18n="uploadTextHint">
      建议粘贴前 5,000–10,000 字，LLM 会自动提取角色、地点和世界观。
    </div>
  </div>

  <div>
    <label data-i18n="uploadFileLabel">或上传 .txt 文件 / Or upload a .txt file</label>
    <label class="upload-file-area" id="up-file-label">
      <input type="file" id="up-file" accept=".txt,.md" />
      <i class="fas fa-file-upload" style="margin-right:6px"></i>
      <span data-i18n="uploadFileDrop">点击选择文件 / Click to choose</span>
    </label>
  </div>

  <div class="upload-progress-wrap" id="up-progress-wrap">
    <div class="upload-progress-bar-track">
      <div class="upload-progress-bar-fill" id="up-bar"></div>
    </div>
    <div class="upload-progress-label" id="up-step">初始化…</div>
  </div>

  <div class="upload-success" id="up-success"></div>

  <div class="upload-modal-actions">
    <button class="upload-cancel-btn" id="up-cancel">取消 / Cancel</button>
    <button class="upload-submit-btn" id="up-submit">
      <i class="fas fa-magic" style="margin-right:4px"></i>
      生成世界 / Generate World
    </button>
  </div>
</div>`;
        document.body.appendChild(this.overlay);

        // Shortcuts to key elements
        this.titleInput = this.overlay.querySelector('#up-title');
        this.langSelect = this.overlay.querySelector('#up-lang');
        this.textArea   = this.overlay.querySelector('#up-text');
        this.fileInput  = this.overlay.querySelector('#up-file');
        this.progressWrap = this.overlay.querySelector('#up-progress-wrap');
        this.bar        = this.overlay.querySelector('#up-bar');
        this.stepLabel  = this.overlay.querySelector('#up-step');
        this.successBox = this.overlay.querySelector('#up-success');
        this.submitBtn  = this.overlay.querySelector('#up-submit');
        this.cancelBtn  = this.overlay.querySelector('#up-cancel');
    }

    _bindEvents() {
        this.cancelBtn.addEventListener('click', () => this.hide());
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) this.hide();
        });

        // File → textarea
        this.fileInput.addEventListener('change', () => {
            const file = this.fileInput.files[0];
            if (!file) return;
            if (!this.titleInput.value) {
                this.titleInput.value = file.name.replace(/\.[^/.]+$/, '');
            }
            const reader = new FileReader();
            reader.onload = (ev) => { this.textArea.value = ev.target.result; };
            reader.readAsText(file, 'utf-8');
        });

        this.submitBtn.addEventListener('click', () => this._submit());
    }

    /* ---- public ---- */

    show() {
        this._resetUI();
        this.overlay.classList.add('active');
    }

    hide() {
        this.overlay.classList.remove('active');
        if (this._pollTimer) { clearInterval(this._pollTimer); this._pollTimer = null; }
    }

    /* ---- internals ---- */

    _resetUI() {
        this.titleInput.value = '';
        this.textArea.value = '';
        this.fileInput.value = '';
        this.progressWrap.classList.remove('active');
        this.successBox.classList.remove('active');
        this.successBox.textContent = '';
        this.bar.style.width = '0%';
        this.stepLabel.textContent = '初始化…';
        this.submitBtn.disabled = false;
    }

    async _submit() {
        const title = this.titleInput.value.trim() || 'Custom Story';
        const language = this.langSelect.value;
        const text = this.textArea.value.trim();

        if (!text) {
            alert('请先粘贴故事内容或上传文件。\nPlease paste story text or upload a file first.');
            return;
        }

        this.submitBtn.disabled = true;
        this.progressWrap.classList.add('active');
        this.successBox.classList.remove('active');
        this._setProgress(5, '提交故事文本… / Submitting…');

        let taskId;
        try {
            const res = await fetch('/api/upload-story', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, language, text }),
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Upload failed');
            }
            const data = await res.json();
            taskId = data.task_id;
        } catch (err) {
            this._showError(err.message);
            return;
        }

        // Poll for completion
        this._pollTimer = setInterval(() => this._poll(taskId), 2000);
    }

    async _poll(taskId) {
        try {
            const res = await fetch(`/api/upload-story/${taskId}`);
            if (!res.ok) return;
            const data = await res.json();

            if (data.status === 'processing') {
                this._setProgress(data.progress || 0, data.step || '处理中…');
            } else if (data.status === 'done') {
                clearInterval(this._pollTimer);
                this._pollTimer = null;
                this._setProgress(100, '完成！/ Done!');
                this._showSuccess(data);
            } else if (data.status === 'error') {
                clearInterval(this._pollTimer);
                this._pollTimer = null;
                this._showError(data.error || 'Unknown error');
            }
        } catch (_) { /* network hiccup, retry on next tick */ }
    }

    _setProgress(pct, label) {
        this.bar.style.width = `${pct}%`;
        this.stepLabel.textContent = label;
    }

    _showSuccess(data) {
        this.successBox.classList.add('active');
        this.successBox.innerHTML = `
✅ 世界已生成！/ World generated!<br>
<strong>${data.world_name || data.preset}</strong><br>
角色 / Characters: ${data.character_count} &nbsp;|&nbsp; 地点 / Locations: ${data.location_count}<br>
<em style="font-size:0.8em;color:#555">${data.preset}</em>`;
        this.submitBtn.disabled = false;
        this.onSuccess(data.preset);
    }

    _showError(msg) {
        clearInterval(this._pollTimer);
        this._pollTimer = null;
        this.progressWrap.classList.remove('active');
        this.successBox.classList.add('active');
        this.successBox.style.background = '#fff5f5';
        this.successBox.style.borderColor = '#ffcdd2';
        this.successBox.style.color = '#c62828';
        this.successBox.textContent = `❌ 错误 / Error: ${msg}`;
        this.submitBtn.disabled = false;
    }
}

window.UploadPanel = UploadPanel;
