// Placeholder for future frontend logic
// TODO: Add loading spinner during question processing
// TODO: Add client-side rate limit tracking
// TODO: Add question history (localStorage)

document.addEventListener('DOMContentLoaded', function() {
    console.log('CV RAG System loaded');
    const form = document.getElementById('questionForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const questionInput = document.getElementById('questionInput');
    
    form.addEventListener('submit', function() {
        submitBtn.disabled = true;
        btnText.textContent = 'Processing...';
        btnSpinner.classList.remove('d-none');
        questionInput.disabled = true;
    });
});