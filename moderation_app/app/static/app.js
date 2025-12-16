(() => {
  function readLabels() {
    if (Array.isArray(window.APP_LABELS)) return window.APP_LABELS;
    const el = document.getElementById('labelsJson');
    if (!el) return [];
    try {
      const parsed = JSON.parse(el.textContent || '[]');
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
      return [];
    }
  }

  const LABELS = readLabels();

  const toastEl = document.getElementById('toast');
  const modalEl = document.getElementById('reportModal');
  const modBadModalEl = document.getElementById('moderateBadModal');
  const reportForm = document.getElementById('reportForm');
  const reportCommentId = document.getElementById('reportCommentId');
  const reportNote = document.getElementById('reportNote');

  const modBadForm = document.getElementById('moderateBadForm');
  const modBadCommentId = document.getElementById('moderateBadCommentId');
  const modBadNote = document.getElementById('moderateBadNote');

  function toast(msg) {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    window.setTimeout(() => toastEl.classList.remove('show'), 2200);
  }

  function openModal(commentId) {
    if (!modalEl) return;
    reportCommentId.value = String(commentId);
    reportNote.value = '';
    modalEl.setAttribute('aria-hidden', 'false');

    // reset checks
    for (const cb of modalEl.querySelectorAll('input[type="checkbox"][name="labels"]')) {
      cb.checked = false;
    }
  }

  function closeModal() {
    if (!modalEl) return;
    modalEl.setAttribute('aria-hidden', 'true');
  }

  function openModBadModal(commentId) {
    if (!modBadModalEl) return;
    modBadCommentId.value = String(commentId);
    if (modBadNote) modBadNote.value = '';
    modBadModalEl.setAttribute('aria-hidden', 'false');
    for (const cb of modBadModalEl.querySelectorAll('input[type="checkbox"][name="labels"]')) {
      cb.checked = false;
    }
  }

  function closeModBadModal() {
    if (!modBadModalEl) return;
    modBadModalEl.setAttribute('aria-hidden', 'true');
  }

  // Render score badges with intensity
  function renderBadges() {
    for (const el of document.querySelectorAll('.badges[data-scores]')) {
      let scores = {};
      try { scores = JSON.parse(el.getAttribute('data-scores') || '{}'); } catch (_) { scores = {}; }

      el.textContent = '';
      for (const label of LABELS) {
        const v = Math.max(0, Math.min(1, Number(scores[label] ?? 0)));
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.textContent = `${label}: ${v.toFixed(2)}`;

        // blue-ish intensity
        const alpha = 0.10 + 0.55 * v;
        badge.style.background = `rgba(110,168,254,${alpha.toFixed(3)})`;
        badge.style.borderColor = `rgba(110,168,254,${Math.max(0.18, alpha).toFixed(3)})`;

        el.appendChild(badge);
      }
    }
  }

  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-report-button]');
    if (btn) {
      openModal(btn.getAttribute('data-comment-id'));
      return;
    }

    if (e.target.closest('[data-modal-close]')) {
      closeModal();
    }

    const modBadBtn = e.target.closest('[data-moderate-bad]');
    if (modBadBtn) {
      openModBadModal(modBadBtn.getAttribute('data-comment-id'));
      return;
    }

    if (e.target.closest('[data-moderate-close]')) {
      closeModBadModal();
    }
  });

  if (reportForm) {
    reportForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const id = Number(reportCommentId.value);
      const labels = [];
      for (const cb of reportForm.querySelectorAll('input[type="checkbox"][name="labels"]')) {
        if (cb.checked) labels.push(cb.value);
      }

      const payload = { labels, note: reportNote.value || '' };

      try {
        const res = await fetch(`/report/${id}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          if (res.status === 429) {
            toast('Rate limited. Try again later.');
          } else {
            toast(err.error || 'Report failed.');
          }
          return;
        }

        closeModal();
        toast('Report submitted');
      } catch (_) {
        toast('Network error');
      }
    });
  }

  if (modBadForm) {
    modBadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const id = Number(modBadCommentId.value);
      const labels = [];
      for (const cb of modBadForm.querySelectorAll('input[type="checkbox"][name="labels"]')) {
        if (cb.checked) labels.push(cb.value);
      }

      const payload = { labels, note: (modBadNote && modBadNote.value) ? modBadNote.value : '' };

      try {
        const res = await fetch(`/moderation/${id}/bad`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          toast(err.error || 'Moderation update failed.');
          return;
        }

        closeModBadModal();
        toast('Marked bad and trained');
        window.setTimeout(() => window.location.reload(), 400);
      } catch (_) {
        toast('Network error');
      }
    });
  }

  renderBadges();
})();
