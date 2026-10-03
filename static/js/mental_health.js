/**
 * mental_health.js
 * Controller for Mental Health & Stress Score Predictor
 * Provides interactive questionnaire controls, ML API communication, dual animated dials,
 * Chart.js psychometric radar visualization, and an interactive 4-7-8 breathing calm widget.
 */

let mhRadarChartInstance = null;
let currentMentalPresets = {};
let breathingTimer = null;
let breathingPhase = 'idle'; // idle, inhale, hold, exhale
let breathingCycleCount = 0;
let currentMode = 'mental'; // 'mental' or 'physical'

// Initialize Mental Health Module
document.addEventListener('DOMContentLoaded', () => {
    initMentalPresets();
    initModeSwitcher();
    // Default active tab is mental health
    switchAppMode('mental');
});

// App Mode Switcher (Mental Health vs Physical Health)
function initModeSwitcher() {
    const mentalTabBtn = document.getElementById('tab-btn-mental');
    const physicalTabBtn = document.getElementById('tab-btn-physical');

    if (mentalTabBtn) {
        mentalTabBtn.addEventListener('click', () => switchAppMode('mental'));
    }
    if (physicalTabBtn) {
        physicalTabBtn.addEventListener('click', () => switchAppMode('physical'));
    }
}

function switchAppMode(mode) {
    currentMode = mode;
    const mentalTabBtn = document.getElementById('tab-btn-mental');
    const physicalTabBtn = document.getElementById('tab-btn-physical');

    const heroMental = document.getElementById('hero-mental-content');
    const heroPhysical = document.getElementById('hero-physical-content');

    const mentalFormSec = document.getElementById('mental-input-section');
    const physicalFormSec = document.getElementById('input-section');

    const mentalResSec = document.getElementById('mental-results-section');
    const physicalResSec = document.getElementById('results-section');
    const physicalPlanSec = document.getElementById('action-plan-section');

    const navResultsMental = document.getElementById('nav-mental-results-link');
    const navResultsPhysical = document.getElementById('nav-results-link');
    const navPlanPhysical = document.getElementById('nav-plan-link');

    if (mode === 'mental') {
        if (mentalTabBtn) mentalTabBtn.classList.add('active-tab');
        if (physicalTabBtn) physicalTabBtn.classList.remove('active-tab');

        if (heroMental) heroMental.classList.remove('hidden');
        if (heroPhysical) heroPhysical.classList.add('hidden');

        if (mentalFormSec) mentalFormSec.classList.remove('hidden');
        if (physicalFormSec) physicalFormSec.classList.add('hidden');

        if (physicalResSec) physicalResSec.classList.add('hidden');
        if (physicalPlanSec) physicalPlanSec.classList.add('hidden');

        if (navResultsPhysical) navResultsPhysical.classList.add('hidden');
        if (navPlanPhysical) navPlanPhysical.classList.add('hidden');

        // Check if mental results were previously rendered
        if (mentalResSec && !mentalResSec.dataset.empty && navResultsMental) {
            navResultsMental.classList.remove('hidden');
        }
    } else {
        if (mentalTabBtn) mentalTabBtn.classList.remove('active-tab');
        if (physicalTabBtn) physicalTabBtn.classList.add('active-tab');

        if (heroMental) heroMental.classList.add('hidden');
        if (heroPhysical) heroPhysical.classList.remove('hidden');

        if (mentalFormSec) mentalFormSec.classList.add('hidden');
        if (physicalFormSec) physicalFormSec.classList.remove('hidden');

        if (mentalResSec) mentalResSec.classList.add('hidden');
        if (navResultsMental) navResultsMental.classList.add('hidden');

        if (physicalResSec && !physicalResSec.classList.contains('hidden')) {
            if (navResultsPhysical) navResultsPhysical.classList.remove('hidden');
            if (navPlanPhysical) navPlanPhysical.classList.remove('hidden');
        }
    }
}

// Fetch Mental Health Presets
async function initMentalPresets() {
    try {
        const res = await fetch('/api/mental-health/presets');
        const json = await res.json();
        if (json.status === 'success') {
            currentMentalPresets = json.presets;
        }
    } catch (e) {
        console.warn("Could not load mental health presets:", e);
    }
}

// Interactive Questionnaire Option Selection
function selectMHOption(questionId, value) {
    const hiddenInput = document.getElementById(`mh-${questionId}`);
    if (hiddenInput) {
        hiddenInput.value = value;
    }

    // Update button states in that question container
    const container = document.getElementById(`opt-group-${questionId}`);
    if (container) {
        const buttons = container.querySelectorAll('.mh-opt-btn');
        buttons.forEach(btn => {
            if (btn.dataset.val === String(value)) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    updateQuestionnaireProgress();
}

// Update Questionnaire Completion Indicator
function updateQuestionnaireProgress() {
    const inputs = document.querySelectorAll('input[name^="q_"]');
    let filled = 0;
    inputs.forEach(inp => {
        if (inp.value !== "" && inp.value !== undefined) {
            filled++;
        }
    });

    const total = inputs.length;
    const pct = Math.round((filled / total) * 100);
    const progressEl = document.getElementById('mh-progress-bar');
    const countEl = document.getElementById('mh-answered-count');
    if (progressEl) progressEl.style.width = `${pct}%`;
    if (countEl) countEl.innerText = `${filled} of ${total} answered (${pct}%)`;
}

// Load Preset Persona
function loadMentalPersona(key) {
    if (!currentMentalPresets[key]) return;
    const d = currentMentalPresets[key].data;

    // Apply values to each question
    for (const [qid, val] of Object.entries(d)) {
        if (qid === 'sleep_hours') {
            const inp = document.getElementById('mh-sleep-hours');
            if (inp) inp.value = val;
            const disp = document.getElementById('mh-sleep-hours-val');
            if (disp) disp.innerText = `${val}h`;
        } else if (qid === 'work_screen_hours') {
            const inp = document.getElementById('mh-screen-hours');
            if (inp) inp.value = val;
            const disp = document.getElementById('mh-screen-hours-val');
            if (disp) disp.innerText = `${val}h`;
        } else {
            selectMHOption(qid, val);
        }
    }

    // Auto-trigger prediction
    document.getElementById('mental-health-form').dispatchEvent(new Event('submit'));
}

// Reset Mental Health Form
function resetMentalForm() {
    const defaultData = {
        q_anhedonia: 1,
        q_depressed: 1,
        q_optimism: 3,
        q_overwhelmed: 1,
        q_uncontrollable: 1,
        q_coping_confidence: 3,
        q_anxious: 1,
        q_worry: 1,
        q_somatic: 1,
        q_sleep_issues: 1,
        q_fatigue: 1,
        q_burnout: 1,
        q_concentration: 1,
        q_isolation: 1,
        q_coping_habits: 2,
        sleep_hours: 7.5,
        work_screen_hours: 6.5
    };

    for (const [qid, val] of Object.entries(defaultData)) {
        if (qid === 'sleep_hours') {
            const inp = document.getElementById('mh-sleep-hours');
            if (inp) inp.value = val;
            const disp = document.getElementById('mh-sleep-hours-val');
            if (disp) disp.innerText = `${val}h`;
        } else if (qid === 'work_screen_hours') {
            const inp = document.getElementById('mh-screen-hours');
            if (inp) inp.value = val;
            const disp = document.getElementById('mh-screen-hours-val');
            if (disp) disp.innerText = `${val}h`;
        } else {
            selectMHOption(qid, val);
        }
    }
}

// Form Submission & API Call
async function handleMentalFormSubmit(event) {
    if (event) event.preventDefault();

    const submitBtn = document.getElementById('mh-submit-btn');
    const btnText = document.getElementById('mh-submit-btn-text');
    const btnIcon = document.getElementById('mh-submit-btn-icon');
    const spinner = document.getElementById('mh-submit-spinner');

    // Loading UI state
    submitBtn.disabled = true;
    btnText.innerText = "Analyzing Psychological & Stress Biomarkers...";
    btnIcon.classList.add('hidden');
    spinner.classList.remove('hidden');

    const payload = {
        q_anhedonia: parseInt(document.getElementById('mh-q_anhedonia').value) || 0,
        q_depressed: parseInt(document.getElementById('mh-q_depressed').value) || 0,
        q_optimism: parseInt(document.getElementById('mh-q_optimism').value) || 0,
        q_overwhelmed: parseInt(document.getElementById('mh-q_overwhelmed').value) || 0,
        q_uncontrollable: parseInt(document.getElementById('mh-q_uncontrollable').value) || 0,
        q_coping_confidence: parseInt(document.getElementById('mh-q_coping_confidence').value) || 0,
        q_anxious: parseInt(document.getElementById('mh-q_anxious').value) || 0,
        q_worry: parseInt(document.getElementById('mh-q_worry').value) || 0,
        q_somatic: parseInt(document.getElementById('mh-q_somatic').value) || 0,
        q_sleep_issues: parseInt(document.getElementById('mh-q_sleep_issues').value) || 0,
        q_fatigue: parseInt(document.getElementById('mh-q_fatigue').value) || 0,
        q_burnout: parseInt(document.getElementById('mh-q_burnout').value) || 0,
        q_concentration: parseInt(document.getElementById('mh-q_concentration').value) || 0,
        q_isolation: parseInt(document.getElementById('mh-q_isolation').value) || 0,
        q_coping_habits: parseInt(document.getElementById('mh-q_coping_habits').value) || 0,
        sleep_hours: parseFloat(document.getElementById('mh-sleep-hours').value) || 7.0,
        work_screen_hours: parseFloat(document.getElementById('mh-screen-hours').value) || 7.0
    };

    try {
        const response = await fetch('/api/mental-health/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (result.status === 'success') {
            renderMentalResults(result.data);
        } else {
            alert("Error predicting mental health scores: " + (result.message || "Unknown error"));
        }
    } catch (err) {
        console.error(err);
        alert("Server communication error. Please ensure the local Flask server is running.");
    } finally {
        submitBtn.disabled = false;
        btnText.innerText = "Calculate Mental Health & Stress Scores";
        btnIcon.classList.remove('hidden');
        spinner.classList.add('hidden');
    }
}

let lastRenderedDomains = null;

// Observe dark/light mode toggles to update radar chart colors
const themeObserver = new MutationObserver(() => {
    if (mhRadarChartInstance && lastRenderedDomains) {
        renderMHRadarChart(lastRenderedDomains);
    }
});
if (typeof document !== 'undefined') {
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
}

// Render Results Dashboard
function renderMentalResults(data) {
    lastRenderedDomains = data.domains;
    const resultsSection = document.getElementById('mental-results-section');
    resultsSection.classList.remove('hidden');
    resultsSection.dataset.empty = "false";

    const navLink = document.getElementById('nav-mental-results-link');
    if (navLink) navLink.classList.remove('hidden');

    // 1. Animated Dual Dials
    animateMHDial('mh-score-display', 'mh-score-circle', data.mental_health_score, data.mental_health_tier.color);
    animateMHDial('stress-score-display', 'stress-score-circle', data.stress_score, data.stress_tier.color);

    // 2. Badges & Descriptions
    const mhTierEl = document.getElementById('mh-tier-badge');
    if (mhTierEl) {
        mhTierEl.innerText = data.mental_health_tier.title;
        mhTierEl.className = `px-3 py-1 rounded-full text-xs font-bold border uppercase tracking-wider ${data.mental_health_tier.badge_class}`;
    }
    const mhDescEl = document.getElementById('mh-tier-desc');
    if (mhDescEl) mhDescEl.innerText = data.mental_health_tier.description;

    const stTierEl = document.getElementById('stress-tier-badge');
    if (stTierEl) {
        stTierEl.innerText = data.stress_tier.title;
        stTierEl.className = `px-3 py-1 rounded-full text-xs font-bold border uppercase tracking-wider ${data.stress_tier.badge_class}`;
    }
    const stDescEl = document.getElementById('stress-tier-desc');
    if (stDescEl) stDescEl.innerText = data.stress_tier.description;

    // 3. Domain Breakdown Bars
    renderMHDomainBars(data.domains);

    // 4. Psychometric Radar Chart
    renderMHRadarChart(data.domains);

    // 5. Risks & Strengths
    renderMHRisksAndStrengths(data.risk_factors, data.strengths);

    // 6. Action Roadmap & Resources
    renderMHActionRoadmap(data.action_plan);

    // 7. Confetti celebration if mental health score is thriving
    if (data.mental_health_score >= 80 && typeof confetti === 'function') {
        confetti({
            particleCount: 65,
            spread: 60,
            origin: { y: 0.6 }
        });
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Animate Circular Gauge
function animateMHDial(numberId, circleId, targetScore, colorTheme) {
    const numEl = document.getElementById(numberId);
    const circleEl = document.getElementById(circleId);

    if (numEl) {
        let current = 0;
        const duration = 1200;
        const start = performance.now();

        function step(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const ease = 1 - Math.pow(1 - progress, 3);
            current = Math.round(ease * targetScore * 10) / 10;
            numEl.innerText = current.toFixed(1);

            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                numEl.innerText = targetScore.toFixed(1);
            }
        }
        requestAnimationFrame(step);
    }

    if (circleEl) {
        const circumference = 2 * Math.PI * 54; // radius 54 => ~339.29
        circleEl.style.strokeDasharray = `${circumference} ${circumference}`;
        const offset = circumference - (targetScore / 100) * circumference;
        circleEl.style.strokeDashoffset = offset;

        // Set color
        if (colorTheme === 'emerald') circleEl.style.stroke = '#10b981';
        else if (colorTheme === 'teal') circleEl.style.stroke = '#14b8a6';
        else if (colorTheme === 'amber') circleEl.style.stroke = '#f59e0b';
        else if (colorTheme === 'orange') circleEl.style.stroke = '#f97316';
        else circleEl.style.stroke = '#f43f5e';
    }
}

// Domain Progress Bars
function renderMHDomainBars(domains) {
    const container = document.getElementById('mh-domain-bars-container');
    if (!container) return;

    container.innerHTML = '';
    for (const [key, dom] of Object.entries(domains)) {
        let barColor = 'bg-emerald-500';
        if (dom.score < 40) barColor = 'bg-rose-500';
        else if (dom.score < 65) barColor = 'bg-amber-500';
        else if (dom.score < 80) barColor = 'bg-teal-500';

        const row = document.createElement('div');
        row.className = "space-y-1.5 p-3 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800";
        row.innerHTML = `
            <div class="flex items-center justify-between text-xs">
                <span class="font-semibold text-slate-800 dark:text-slate-200 flex items-center gap-2">
                    <i class="${dom.icon} text-indigo-500"></i> ${dom.name}
                </span>
                <span class="font-bold text-slate-700 dark:text-slate-300">${dom.score}% <span class="text-[11px] font-normal text-slate-500">(${dom.status})</span></span>
            </div>
            <div class="w-full h-2 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                <div class="h-full rounded-full ${barColor} transition-all duration-1000 ease-out" style="width: 0%" data-target="${dom.score}"></div>
            </div>
        `;
        container.appendChild(row);
    }

    // Trigger width animations
    setTimeout(() => {
        container.querySelectorAll('[data-target]').forEach(bar => {
            bar.style.width = `${bar.getAttribute('data-target')}%`;
        });
    }, 100);
}

// Render Psychometric Radar Chart
function renderMHRadarChart(domains) {
    const canvas = document.getElementById('mh-radar-chart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const isDark = document.documentElement.classList.contains('dark');
    const gridColor = isDark ? 'rgba(148, 163, 184, 0.2)' : 'rgba(203, 213, 225, 0.6)';
    const textColor = isDark ? '#cbd5e1' : '#475569';

    const labels = [
        'Mood Vitality',
        'Stress Resilience',
        'Nervous Calm',
        'Sleep Recovery',
        'Cognitive Vitality',
        'Social Coping'
    ];
    const dataVals = [
        domains.emotional_vitality.score,
        domains.stress_resilience.score,
        domains.anxiety_regulation.score,
        domains.sleep_recovery.score,
        domains.cognitive_vitality.score,
        domains.social_coping.score
    ];

    if (mhRadarChartInstance) {
        mhRadarChartInstance.destroy();
    }

    mhRadarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Your Resilience Profile',
                data: dataVals,
                backgroundColor: 'rgba(99, 102, 241, 0.2)',
                borderColor: '#6366f1',
                borderWidth: 2.5,
                pointBackgroundColor: '#4f46e5',
                pointBorderColor: '#ffffff',
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 25,
                        color: textColor,
                        backdropColor: 'transparent',
                        font: { size: 10 }
                    },
                    grid: { color: gridColor },
                    angleLines: { color: gridColor },
                    pointLabels: {
                        color: textColor,
                        font: { size: 11, weight: '600' }
                    }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Render Risks & Strengths Cards
function renderMHRisksAndStrengths(risks, strengths) {
    const risksContainer = document.getElementById('mh-risks-container');
    const strengthsContainer = document.getElementById('mh-strengths-container');

    if (risksContainer) {
        risksContainer.innerHTML = '';
        if (risks.length === 0) {
            risksContainer.innerHTML = `
                <div class="p-3.5 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-xs text-emerald-800 dark:text-emerald-300 flex items-center gap-2">
                    <i class="fa-solid fa-circle-check text-emerald-500 text-base"></i>
                    <span>No critical psychological red flags detected. Autonomic regulation is steady.</span>
                </div>
            `;
        } else {
            risks.forEach(r => {
                const item = document.createElement('div');
                item.className = "p-3 rounded-xl bg-rose-50/70 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/60 space-y-1";
                item.innerHTML = `
                    <div class="flex items-center gap-2 text-rose-700 dark:text-rose-400 font-bold text-xs">
                        <i class="${r.icon || 'fa-solid fa-triangle-exclamation'}"></i> ${r.title}
                    </div>
                    <p class="text-[11px] text-slate-600 dark:text-slate-300 leading-relaxed">${r.detail}</p>
                `;
                risksContainer.appendChild(item);
            });
        }
    }

    if (strengthsContainer) {
        strengthsContainer.innerHTML = '';
        strengths.forEach(s => {
            const item = document.createElement('div');
            item.className = "p-3 rounded-xl bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900/60 space-y-1";
            item.innerHTML = `
                <div class="flex items-center gap-2 text-emerald-700 dark:text-emerald-400 font-bold text-xs">
                    <i class="fa-solid fa-circle-check"></i> ${s.title}
                </div>
                <p class="text-[11px] text-slate-600 dark:text-slate-300 leading-relaxed">${s.detail}</p>
            `;
            strengthsContainer.appendChild(item);
        });
    }
}

// Render Action Roadmap & Helplines
function renderMHActionRoadmap(plan) {
    const p1Container = document.getElementById('mh-phase1-container');
    const p2Container = document.getElementById('mh-phase2-container');
    const p3Container = document.getElementById('mh-phase3-container');
    const resContainer = document.getElementById('mh-resources-container');
    const discEl = document.getElementById('mh-disclaimer-text');

    if (p1Container && plan.phase1) {
        p1Container.innerHTML = plan.phase1.actions.map(a => `
            <div class="flex items-start gap-2.5 text-xs text-slate-700 dark:text-slate-300">
                <i class="fa-solid fa-arrow-right text-indigo-500 mt-0.5"></i>
                <span>${a}</span>
            </div>
        `).join('');
    }

    if (p2Container && plan.phase2) {
        p2Container.innerHTML = plan.phase2.actions.map(a => `
            <div class="flex items-start gap-2.5 text-xs text-slate-700 dark:text-slate-300">
                <i class="fa-solid fa-arrow-right text-teal-500 mt-0.5"></i>
                <span>${a}</span>
            </div>
        `).join('');
    }

    if (p3Container && plan.phase3) {
        p3Container.innerHTML = plan.phase3.actions.map(a => `
            <div class="flex items-start gap-2.5 text-xs text-slate-700 dark:text-slate-300">
                <i class="fa-solid fa-arrow-right text-purple-500 mt-0.5"></i>
                <span>${a}</span>
            </div>
        `).join('');
    }

    if (resContainer && plan.resources) {
        resContainer.innerHTML = plan.resources.map(r => `
            <div class="p-2.5 rounded-xl bg-slate-100 dark:bg-slate-800 text-xs flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
                <span class="font-semibold text-slate-800 dark:text-slate-200">${r.name}</span>
                <span class="text-indigo-600 dark:text-indigo-400 font-bold">${r.contact}</span>
            </div>
        `).join('');
    }

    if (discEl && plan.disclaimer) {
        discEl.innerText = plan.disclaimer;
    }
}

// ==========================================
// 4-7-8 Breathing Calm Interactive Exercise
// ==========================================
let breathingInterval = null;
let breathingSecRemaining = 0;
let isBreathingActive = false;

function toggleBreathingExercise() {
    if (isBreathingActive) {
        stopBreathingExercise();
    } else {
        startBreathingExercise();
    }
}

function startBreathingExercise() {
    isBreathingActive = true;
    breathingCycleCount = 1;
    const btn = document.getElementById('breathing-toggle-btn');
    if (btn) {
        btn.innerHTML = '<i class="fa-solid fa-pause mr-1.5"></i> Pause Breathing';
        btn.className = "px-4 py-2 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-700 text-white transition-all shadow-md";
    }
    runBreathingPhase('inhale');
}

function stopBreathingExercise() {
    isBreathingActive = false;
    clearTimeout(breathingTimer);
    clearInterval(breathingInterval);

    const circle = document.getElementById('breathing-circle-inner');
    if (circle) circle.className = 'breathing-circle-inner';

    const textEl = document.getElementById('breathing-guide-text');
    if (textEl) textEl.innerText = "Ready to start?";

    const timerEl = document.getElementById('breathing-timer-disp');
    if (timerEl) timerEl.innerText = "--";

    const btn = document.getElementById('breathing-toggle-btn');
    if (btn) {
        btn.innerHTML = '<i class="fa-solid fa-play mr-1.5"></i> Start 4-7-8 Calm Exercise';
        btn.className = "px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white transition-all shadow-md";
    }
}

function runBreathingPhase(phase) {
    if (!isBreathingActive) return;

    const circle = document.getElementById('breathing-circle-inner');
    const textEl = document.getElementById('breathing-guide-text');
    const timerEl = document.getElementById('breathing-timer-disp');
    const cycleEl = document.getElementById('breathing-cycle-disp');

    if (cycleEl) cycleEl.innerText = `Cycle ${breathingCycleCount} of 4`;

    clearInterval(breathingInterval);

    if (phase === 'inhale') {
        breathingSecRemaining = 4;
        if (circle) circle.className = 'breathing-circle-inner inhale';
        if (textEl) textEl.innerText = "Breathe In (through nose)...";
        if (timerEl) timerEl.innerText = `${breathingSecRemaining}s`;

        breathingInterval = setInterval(() => {
            breathingSecRemaining--;
            if (timerEl) timerEl.innerText = `${breathingSecRemaining}s`;
            if (breathingSecRemaining <= 0) {
                clearInterval(breathingInterval);
                runBreathingPhase('hold');
            }
        }, 1000);

    } else if (phase === 'hold') {
        breathingSecRemaining = 7;
        if (circle) circle.className = 'breathing-circle-inner hold';
        if (textEl) textEl.innerText = "Hold gently & relax shoulders...";
        if (timerEl) timerEl.innerText = `${breathingSecRemaining}s`;

        breathingInterval = setInterval(() => {
            breathingSecRemaining--;
            if (timerEl) timerEl.innerText = `${breathingSecRemaining}s`;
            if (breathingSecRemaining <= 0) {
                clearInterval(breathingInterval);
                runBreathingPhase('exhale');
            }
        }, 1000);

    } else if (phase === 'exhale') {
        breathingSecRemaining = 8;
        if (circle) circle.className = 'breathing-circle-inner exhale';
        if (textEl) textEl.innerText = "Exhale slowly (through mouth)...";
        if (timerEl) timerEl.innerText = `${breathingSecRemaining}s`;

        breathingInterval = setInterval(() => {
            breathingSecRemaining--;
            if (timerEl) timerEl.innerText = `${breathingSecRemaining}s`;
            if (breathingSecRemaining <= 0) {
                clearInterval(breathingInterval);
                breathingCycleCount++;
                if (breathingCycleCount > 4) {
                    // Complete 4 cycles
                    if (textEl) textEl.innerText = "Great job! Parasympathetic vagal reset complete.";
                    stopBreathingExercise();
                } else {
                    runBreathingPhase('inhale');
                }
            }
        }, 1000);
    }
}
