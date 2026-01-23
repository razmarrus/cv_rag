// Placeholder for future frontend logic
// TODO: Add loading spinner during question processing
// TODO: Add client-side rate limit tracking
// TODO: Add question history (localStorage)

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('questionForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const questionInput = document.getElementById('questionInput');
    
    if (form && submitBtn) {
        form.addEventListener('submit', function() {
            submitBtn.disabled = true;
            btnText.textContent = 'Thinking...';
            btnSpinner.classList.add('active');
            questionInput.disabled = true;
        });
    }
});