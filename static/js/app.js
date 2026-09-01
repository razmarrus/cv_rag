const PRESET_QUESTIONS = {
    projects: [
        'What is your most recent project?',
        'Have you built RAG systems?',
        'What was your biggest LLM project?',
        'Tell me about your RAG chatbot project.',
        'Have you built recommendation systems?',
        'Do you have experience with sales forecasting?',
        'What forecasting accuracy have you achieved?',
        'Have you worked on customer segmentation?',
        'What real-time data projects have you done?',
        'What document Q&A systems have you built?',
        'What industries have your projects covered?',
        'Tell me about a project you deployed to production.',
    ],
    stack: [
        'What is your tech stack and tools you use?',
        'What is your primary programming language?',
        'Do you have Spark experience?',
        'What cloud platforms do you use?',
        'Do you work with Azure?',
        'Do you use Terraform?',
        'What LLM frameworks do you know?',
        'What Python libraries do you use?',
        'Do you have MLOps experience?',
        'What databases do you work with?',
        'Do you use Docker in production?',
        'What do you use for CI/CD?',
    ],
    offScript: [
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
    ],
};

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function toggleAccordion(id) {
    const el = document.getElementById(id);
    if (el) {
        el.classList.toggle('active');
    }
}

function fillPreset(kind) {
    const pool = PRESET_QUESTIONS[kind];
    const input = document.getElementById('questionInput');
    const presetFlag = document.getElementById('presetFlag');
    if (!pool || !pool.length || !input) {
        return;
    }
    input.value = pool[Math.floor(Math.random() * pool.length)];
    input.focus();
    if (presetFlag) {
        presetFlag.value = kind;
    }
}

function setFormBusy(busy) {
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    if (submitBtn) {
        submitBtn.disabled = busy;
    }
    if (btnText) {
        btnText.textContent = busy ? '...' : '→';
    }
    if (btnSpinner) {
        btnSpinner.classList.toggle('active', busy);
    }
}

function renderAnswer(data) {
    const answerDisplay = document.querySelector('.answer-display');
    if (!answerDisplay) {
        return;
    }
    if (data.answer) {
        let html = '<p>' + escapeHtml(data.answer) + '</p>';
        const tags = [];
        if (data.sources && data.sources.length) {
            tags.push('Sources: ' + escapeHtml(data.sources.join(', ')));
        }
        if (data.num_chunks) {
            tags.push(data.num_chunks + ' chunks');
        }
        if (data.execution_time) {
            tags.push(data.execution_time + 's');
        }
        if (tags.length) {
            html += '<div class="metadata">' +
                tags.map(function (tag) {
                    return '<span class="tag">' + tag + '</span>';
                }).join('') +
                '</div>';
        }
        answerDisplay.innerHTML = html;
        return;
    }
    const message = data.error || 'An error occurred. Please try again.';
    answerDisplay.innerHTML = '<p style="color: #C33;">' + escapeHtml(message) + '</p>';
}

function updateQuota(data) {
    if (data.remaining_requests === undefined || !data.daily_limit) {
        return;
    }
    const quotaNumber = document.querySelector('.quota-number');
    if (quotaNumber) {
        quotaNumber.textContent = data.remaining_requests + '/' + data.daily_limit;
    }
}

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('questionForm');
    const questionInput = document.getElementById('questionInput');
    const presetFlag = document.getElementById('presetFlag');
    const answerDisplay = document.querySelector('.answer-display');

    if (questionInput && presetFlag) {
        questionInput.addEventListener('input', function () {
            presetFlag.value = '';
        });
    }

    if (!form) {
        return;
    }

    form.addEventListener('submit', async function (event) {
        event.preventDefault();
        setFormBusy(true);
        if (answerDisplay) {
            answerDisplay.innerHTML = '<p class="answer-placeholder">Thinking...</p>';
        }
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { Accept: 'application/json' },
                body: new URLSearchParams(new FormData(form)),
            });
            const data = await response.json();
            renderAnswer(data);
            updateQuota(data);
        } catch (err) {
            console.error(err);
            renderAnswer({ error: 'An error occurred. Please try again.' });
        } finally {
            setFormBusy(false);
        }
    });
});
