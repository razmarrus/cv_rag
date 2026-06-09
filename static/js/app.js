document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('questionForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const questionInput = document.getElementById('questionInput');
    const answerDisplay = document.querySelector('.answer-display');
    
    // AJAX form submission - no page refresh
    if (form && submitBtn) {
        form.addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent page refresh
            
            const formData = new FormData(form);
            const question = formData.get('question');
            
            // Show loading state
            submitBtn.disabled = true;
            btnText.textContent = '...';
            btnSpinner.classList.add('active');
            
            // Show loading in answer box
            if (answerDisplay) {
                answerDisplay.innerHTML = '<p class="answer-placeholder">Thinking...</p>';
            }
            
            // Send AJAX request
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: new URLSearchParams(formData)
            })
            .then(response => response.json())
            .then(data => {
                // Reset button state
                submitBtn.disabled = false;
                btnText.textContent = '→';
                btnSpinner.classList.remove('active');
                
                // Display answer
                if (data.answer && answerDisplay) {
                    let html = '<p>' + data.answer + '</p>';
                    
                    // Add metadata if available
                    if (data.sources || data.num_chunks || data.execution_time) {
                        html += '<div class="metadata">';
                        if (data.sources) {
                            html += '<span class="tag">Sources: ' + data.sources.join(', ') + '</span>';
                        }
                        if (data.num_chunks) {
                            html += '<span class="tag">' + data.num_chunks + ' chunks</span>';
                        }
                        if (data.execution_time) {
                            html += '<span class="tag">' + data.execution_time + 's</span>';
                        }
                        html += '</div>';
                    }
                    
                    answerDisplay.innerHTML = html;
                } else if (data.error && answerDisplay) {
                    answerDisplay.innerHTML = '<p style="color: #C33;">Error: ' + data.error + '</p>';
                }
                
                // Update quota if provided
                if (data.remaining_requests !== undefined) {
                    const quotaNumber = document.querySelector('.quota-number');
                    if (quotaNumber && data.daily_limit) {
                        quotaNumber.textContent = data.remaining_requests + '/' + data.daily_limit;
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);
                submitBtn.disabled = false;
                btnText.textContent = '→';
                btnSpinner.classList.remove('active');
                
                if (answerDisplay) {
                    answerDisplay.innerHTML = '<p style="color: #C33;">An error occurred. Please try again.</p>';
                }
            });
        });
    }
    
});

// Global function for onclick handlers
function togglePortfolioAccordion(accordionId) {
    const accordion = document.getElementById(accordionId);
    if (accordion) {
        accordion.classList.toggle('active');
    }
}

const OFF_SCRIPT_QUESTIONS = [
    'What are your hobbies?',
    'Have you participated in any half marathons?',
    'Who is your favorite film director?',
    'Do you enjoy sports?',
    'What do you do for fun?',
    'What is your favorite movie?',
    'Do you like pasta?',
    'Do you like cats?',
    'What games do you play?',
    'What is your favorite Fallout game?',
    'Do you collect vinyl records?',
];

function fillQuestion(question) {
    const input = document.getElementById('questionInput');
    if (input) {
        input.value = question;
        input.focus();
    }
}

function fillOffScriptQuestion() {
    const question = OFF_SCRIPT_QUESTIONS[
        Math.floor(Math.random() * OFF_SCRIPT_QUESTIONS.length)
    ];
    fillQuestion(question);
}

function toggleAccordion(accordionId) {
    const accordion = document.getElementById(accordionId);
    if (accordion) {
        accordion.classList.toggle('active');
    }
}