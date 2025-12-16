(() => {
  const LABELS = Array.isArray(window.APP_LABELS) ? window.APP_LABELS : [];

  const toastEl = document.getElementById('toast');
  const modalEl = document.getElementById('reportModal');
  const reportForm = document.getElementById('reportForm');
  const reportCommentId = document.getElementById('reportCommentId');
  const reportNote = document.getElementById('reportNote');

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

  renderBadges();
})();
