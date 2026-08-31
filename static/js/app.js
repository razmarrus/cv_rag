document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('questionForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const questionInput = document.getElementById('questionInput');
    const answerDisplay = document.querySelector('.answer-display');

    if (form && submitBtn) {
        form.addEventListener('submit', function(event) {
            event.preventDefault();

            const formData = new FormData(form);

            submitBtn.disabled = true;
            btnText.textContent = '...';
            btnSpinner.classList.add('active');

            if (answerDisplay) {
                answerDisplay.innerHTML = '<p class="answer-placeholder">Thinking...</p>';
            }

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
                submitBtn.disabled = false;
                btnText.textContent = '→';
                btnSpinner.classList.remove('active');

                if (data.answer && answerDisplay) {
                    let html = '<p>' + escapeHtml(data.answer) + '</p>';
                    if (data.sources || data.num_chunks || data.execution_time) {
                        html += '<div class="metadata">';
                        if (data.sources) {
                            html += '<span class="tag">Sources: ' + escapeHtml(data.sources.join(', ')) + '</span>';
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
                    answerDisplay.innerHTML = '<p style="color: #C33;">Error: ' + escapeHtml(data.error) + '</p>';
                }

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

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function togglePortfolioAccordion(accordionId) {
    const accordion = document.getElementById(accordionId);
    if (accordion) {
        accordion.classList.toggle('active');
    }
}

// Keep in sync with _PRESET_OFF_SCRIPT_QUESTIONS in main.py, which drives
// preset-aware prompting server-side.
const OFF_SCRIPT_QUESTIONS = [
    'What are your hobbies?',
    'What do you do for fun?',
    'What do you do outside of work?',
    'How do you unwind after work?',
    'What makes you happy?',
    "What's on your mind?",
    'Tell me something that is not on your CV.',
    'Have you participated in any half marathons?',
    'Do you enjoy sports?',
    'Do you like pasta?',
    'What is your favorite food?',
    'Do you like cooking?',
    'Do you collect vinyl records?',
    'What are your favorite bands?',
    'Who is your favorite musician?',
    'What is your favorite album?',
    'What music do you listen to while coding?',
    'What do you listen to before sleep?',
    'Can you play piano?',
    'Can you play an instrument?',
    'What is your favorite film?',
    'Who is your favorite film director?',
    'What films have you watched recently?',
    'What games do you play?',
    'Do you play video games?',
    'Are you a mentor?',
    'Do you like teaching?',
    'Do you give presentations?',
    'Can you explain AI to non-technical people?',
    'Why do you work in AI and software engineering?',
    'What do you think of Rick Rubin?',
    'Do you lie down at parties?',
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
