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

function clearRickRubinMode() {
    const mode = document.getElementById('rickRubinMode');
    if (mode) mode.value = '';
}

function togglePortfolioAccordion(accordionId) {
    const accordion = document.getElementById(accordionId);
    if (accordion) {
        accordion.classList.toggle('active');
    }
}

const OFF_SCRIPT_QUESTIONS = [
    'What are your hobbies?',
    'Have you participated in any half marathons?',
    'Do you enjoy sports?',
    'What makes you happy?',
    'What do you do for fun?',
    'Do you like pasta?',
    'Do you collect vinyl records?',
    "What's on your mind?",
    'What are your favorite bands?',
    'What is your favorite film?',
    'Why do you work in AI and software engineering?',
    'Can you explain AI to non-technical people?',
    'Are you a mentor?',
    'Can you play piano?',
    'What is your favorite food?',
    'Who is your favorite film director?',
];

const RICK_RUBIN_QUESTIONS = [
    'What do you think of Rick Rubin?',
    'Do you like pasta?',
    'What are your favorite bands?',
    'Can you play an instrument?',
    'What do you listen to before sleep?',
    'Do you lie down at parties?',
    // 'Thanks for Korn — what do you think?',
    'What are your favorite bands?',
    'Do you give presentations?',
    'Can you explain AI concepts to non-technical people?',
    'What\'s on your mind?',
];

function fillQuestion(question) {
    clearRickRubinMode();
    const input = document.getElementById('questionInput');
    if (input) {
        input.value = question;
        input.focus();
    }
}

function fillOffScriptQuestion() {
    clearRickRubinMode();
    const question = OFF_SCRIPT_QUESTIONS[
        Math.floor(Math.random() * OFF_SCRIPT_QUESTIONS.length)
    ];
    fillQuestion(question);
}

function fillRickRubinQuestion() {
    clearRickRubinMode();
    const question = RICK_RUBIN_QUESTIONS[
        Math.floor(Math.random() * RICK_RUBIN_QUESTIONS.length)
    ];
    const input = document.getElementById('questionInput');
    const mode = document.getElementById('rickRubinMode');
    if (input) {
        input.value = question;
        input.focus();
    }
    if (mode) {
        mode.value = '1';
    }
}

function toggleAccordion(accordionId) {
    const accordion = document.getElementById(accordionId);
    if (accordion) {
        accordion.classList.toggle('active');
    }
}
