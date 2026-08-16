// Application number - a small realistic touch
document.getElementById('app-number').textContent =
  String(Math.floor(100000 + Math.random() * 899999));

// Slip date
document.getElementById('slip-date').textContent =
  new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });

// CIBIL score live hint
const cibilInput = document.getElementById('cibil-input');
const cibilHint = document.getElementById('cibil-hint');

cibilInput.addEventListener('input', () => {
  const val = Number(cibilInput.value);
  if (!val) {
    cibilHint.textContent = 'Credit score used by Indian lenders. 750+ is generally considered strong.';
    cibilHint.classList.remove('warn');
    return;
  }
  if (val < 300 || val > 900) {
    cibilHint.textContent = 'CIBIL score must be between 300 and 900.';
    cibilHint.classList.add('warn');
  } else if (val < 550) {
    cibilHint.textContent = 'Poor range \u2014 approval is unlikely regardless of other factors.';
    cibilHint.classList.add('warn');
  } else if (val < 700) {
    cibilHint.textContent = 'Fair range \u2014 other factors will weigh in more heavily.';
    cibilHint.classList.remove('warn');
  } else {
    cibilHint.textContent = 'Strong range \u2014 favorable for approval.';
    cibilHint.classList.remove('warn');
  }
});

// Panels
const slipEmpty = document.getElementById('slip-empty');
const slipResult = document.getElementById('slip-result');
const slipLoading = document.getElementById('slip-loading');
const slipError = document.getElementById('slip-error');

function showPanel(panel) {
  [slipEmpty, slipResult, slipLoading, slipError].forEach(p => p.classList.add('hidden'));
  panel.classList.remove('hidden');
}

const form = document.getElementById('loan-form');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  showPanel(slipLoading);

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    renderResult(data);
  } catch (err) {
    showPanel(slipError);
    document.getElementById('error-text').textContent = err.message;
  }
});

function renderResult(data) {
  const stamp = document.getElementById('stamp');
  const confidenceValue = document.getElementById('confidence-value');
  const confidenceFill = document.getElementById('confidence-fill');
  const slipNote = document.getElementById('slip-note');

  // Reset stamp animation by cloning the node
  const newStamp = stamp.cloneNode(true);
  stamp.parentNode.replaceChild(newStamp, stamp);

  const isApproved = data.status === 'APPROVED';
  newStamp.classList.remove('approved', 'rejected');
  newStamp.classList.add(isApproved ? 'approved' : 'rejected');
  newStamp.querySelector('#stamp-text').textContent = data.status;

  confidenceValue.textContent = `${data.probability}%`;
  confidenceFill.style.width = '0%';

  showPanel(slipResult);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confidenceFill.style.width = `${data.probability}%`;
    });
  });

  if (!isApproved && data.cibilScore < 550) {
    slipNote.textContent = 'Primary factor: CIBIL score is in the poor range.';
  } else if (isApproved) {
    slipNote.textContent = 'Application meets the model\u2019s approval criteria.';
  } else {
    slipNote.textContent = 'Application falls short of the model\u2019s approval threshold.';
  }
}
