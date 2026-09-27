/**
 * app.js
 * Frontend controller for Vitalis AI - Health Score Predictor & Future Action Planner
 */

let radarChartInstance = null;
let currentPresets = {};

// Theme management
function initTheme() {
    const themeToggleBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const savedTheme = localStorage.getItem('vitalis-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        document.documentElement.classList.add('dark');
        themeIcon.className = 'fa-solid fa-sun text-lg text-amber-400';
    } else {
        document.documentElement.classList.remove('dark');
        themeIcon.className = 'fa-solid fa-moon text-lg';
    }

    themeToggleBtn.addEventListener('click', () => {
        const isDark = document.documentElement.classList.toggle('dark');
        localStorage.setItem('vitalis-theme', isDark ? 'dark' : 'light');
        themeIcon.className = isDark ? 'fa-solid fa-sun text-lg text-amber-400' : 'fa-solid fa-moon text-lg';
        if (radarChartInstance) {
            updateRadarTheme();
        }
    });
}

// Live BMI Calculator
function updateLiveBMI() {
    const heightCm = parseFloat(document.getElementById('input-height').value) || 0;
    const weightKg = parseFloat(document.getElementById('input-weight').value) || 0;
    const badge = document.getElementById('live-bmi-badge');
    const valSpan = document.getElementById('live-bmi-val');
    const descSpan = document.getElementById('live-bmi-desc');

    if (heightCm > 0 && weightKg > 0) {
        const heightM = heightCm / 100.0;
        const bmi = (weightKg / (heightM * heightM)).toFixed(1);
        valSpan.innerText = bmi;

        let desc = "Normal";
        let colorClass = "text-emerald-500";
        if (bmi < 18.5) {
            desc = "Underweight";
            colorClass = "text-amber-500";
        } else if (bmi <= 24.9) {
            desc = "Optimal Range";
            colorClass = "text-emerald-500";
        } else if (bmi <= 29.9) {
            desc = "Overweight";
            colorClass = "text-amber-500";
        } else {
            desc = "Obesity Range";
            colorClass = "text-rose-500";
        }

        descSpan.innerText = desc;
        descSpan.className = `font-medium ${colorClass}`;
        badge.innerText = `BMI: ${bmi}`;
    }
}

// Live Blood Pressure Category
function updateLiveBP() {
    const sys = parseInt(document.getElementById('input-systolic').value) || 120;
    const dia = parseInt(document.getElementById('input-diastolic').value) || 80;
    const badge = document.getElementById('live-bp-badge');

    if (sys < 120 && dia < 80) {
        badge.innerText = "BP: Optimal";
        badge.className = "px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-400";
    } else if (sys <= 129 && dia < 80) {
        badge.innerText = "BP: Elevated";
        badge.className = "px-2 py-0.5 text-xs font-semibold rounded-full bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-400";
    } else if (sys <= 139 || dia <= 89) {
        badge.innerText = "BP: Stage 1 HTN";
        badge.className = "px-2 py-0.5 text-xs font-semibold rounded-full bg-orange-100 dark:bg-orange-950 text-orange-700 dark:text-orange-400";
    } else {
        badge.innerText = "BP: High / Stage 2";
        badge.className = "px-2 py-0.5 text-xs font-semibold rounded-full bg-rose-100 dark:bg-rose-950 text-rose-700 dark:text-rose-400";
    }
}

// Live Stress Slider Display
function updateStressDisplay() {
    const stress = parseInt(document.getElementById('input-stress').value);
    document.getElementById('stress-val').innerText = stress;
    const indicator = document.getElementById('stress-indicator');

    if (stress <= 3) {
        indicator.innerText = `Low (${stress}/10) — Balanced`;
        indicator.className = "text-xs font-semibold text-emerald-600 dark:text-emerald-400";
    } else if (stress <= 6) {
        indicator.innerText = `Moderate (${stress}/10) — Manageable`;
        indicator.className = "text-xs font-semibold text-amber-600 dark:text-amber-400";
    } else {
        indicator.innerText = `High (${stress}/10) — Elevated Load`;
        indicator.className = "text-xs font-semibold text-rose-600 dark:text-rose-400";
    }
}

// Load presets from server or fallback
async function fetchPresets() {
    try {
        const res = await fetch('/api/presets');
        const json = await res.json();
        if (json.status === 'success') {
            currentPresets = json.presets;
        }
    } catch (e) {
        console.warn("Could not fetch presets:", e);
    }
}

function loadPersona(key) {
    if (!currentPresets[key]) return;
    const d = currentPresets[key].data;

    document.getElementById('input-age').value = d.age;
    document.getElementById('input-gender').value = d.gender;
    document.getElementById('input-height').value = d.height;
    document.getElementById('input-weight').value = d.weight;
    document.getElementById('input-systolic').value = d.systolic_bp;
    document.getElementById('input-diastolic').value = d.diastolic_bp;
    document.getElementById('input-rhr').value = d.resting_hr;
    document.getElementById('input-glucose').value = d.glucose;
    document.getElementById('input-cholesterol').value = d.cholesterol;
    document.getElementById('input-activity').value = d.activity_mins;
    document.getElementById('input-steps').value = d.daily_steps;
    document.getElementById('input-sleep').value = d.sleep_hours;
    document.getElementById('input-sleep-qual').value = d.sleep_quality;
    document.getElementById('input-water').value = d.water_liters;
    document.getElementById('input-diet').value = d.diet_quality;
    document.getElementById('input-smoking').value = d.smoking_status;
    document.getElementById('input-alcohol').value = d.alcohol_intake;
    document.getElementById('input-stress').value = d.stress_level;
    document.getElementById('input-screen').value = d.screen_time;

    document.getElementById('input-family-cvd').checked = !!d.family_cvd;
    document.getElementById('input-family-diabetes').checked = !!d.family_diabetes;
    document.getElementById('input-family-hypertension').checked = !!d.family_hypertension;

    updateLiveBMI();
    updateLiveBP();
    updateStressDisplay();

    // Trigger calculation automatically for great UX
    document.getElementById('health-form').dispatchEvent(new Event('submit'));
}

function resetForm() {
    document.getElementById('health-form').reset();
    document.getElementById('input-age').value = 30;
    document.getElementById('input-height').value = 175;
    document.getElementById('input-weight').value = 70;
    document.getElementById('input-systolic').value = 118;
    document.getElementById('input-diastolic').value = 76;
    document.getElementById('input-stress').value = 3;
    updateLiveBMI();
    updateLiveBP();
    updateStressDisplay();
}

// Form Submission & API Call
async function handleFormSubmit(event) {
    if (event) event.preventDefault();

    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.getElementById('submit-btn-text');
    const btnIcon = document.getElementById('submit-btn-icon');
    const spinner = document.getElementById('submit-spinner');

    // Loading state
    submitBtn.disabled = true;
    btnText.innerText = "Analyzing Biomarkers...";
    btnIcon.classList.add('hidden');
    spinner.classList.remove('hidden');

    const payload = {
        age: parseInt(document.getElementById('input-age').value),
        gender: document.getElementById('input-gender').value,
        height: parseFloat(document.getElementById('input-height').value),
        weight: parseFloat(document.getElementById('input-weight').value),
        systolic_bp: parseFloat(document.getElementById('input-systolic').value),
        diastolic_bp: parseFloat(document.getElementById('input-diastolic').value),
        resting_hr: parseFloat(document.getElementById('input-rhr').value),
        glucose: parseFloat(document.getElementById('input-glucose').value),
        cholesterol: parseFloat(document.getElementById('input-cholesterol').value),
        activity_mins: parseFloat(document.getElementById('input-activity').value),
        daily_steps: parseFloat(document.getElementById('input-steps').value),
        sleep_hours: parseFloat(document.getElementById('input-sleep').value),
        sleep_quality: parseInt(document.getElementById('input-sleep-qual').value),
        water_liters: parseFloat(document.getElementById('input-water').value),
        diet_quality: parseInt(document.getElementById('input-diet').value),
        smoking_status: document.getElementById('input-smoking').value,
        alcohol_intake: document.getElementById('input-alcohol').value,
        stress_level: parseInt(document.getElementById('input-stress').value),
        screen_time: parseFloat(document.getElementById('input-screen').value),
        family_cvd: document.getElementById('input-family-cvd').checked,
        family_diabetes: document.getElementById('input-family-diabetes').checked,
        family_hypertension: document.getElementById('input-family-hypertension').checked
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (result.status === 'success') {
            renderResults(result.data);
        } else {
            alert("Error predicting health score: " + (result.message || "Unknown error"));
        }
    } catch (err) {
        console.error(err);
        alert("Server communication error. Please ensure the local Flask server is running.");
    } finally {
        submitBtn.disabled = false;
        btnText.innerText = "Generate Health Score & Action Plan";
        btnIcon.classList.remove('hidden');
        spinner.classList.add('hidden');
    }
}

// Render Results Dashboard
function renderResults(data) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.remove('hidden');

    // Make nav items active
    document.getElementById('nav-results-link').classList.remove('hidden');
    document.getElementById('nav-plan-link').classList.remove('hidden');

    // 1. Overall Score Dial
    const targetScore = Math.round(data.overall_score);
    animateScoreNumber(targetScore);
    animateScoreCircle(data.overall_score);

    // Status badge
    const statusBadge = document.getElementById('res-status-badge');
    statusBadge.innerText = data.status_tier;
    if (data.overall_score >= 88) {
        statusBadge.className = "mt-1 px-3 py-1 text-xs font-bold rounded-full bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300";
    } else if (data.overall_score >= 74) {
        statusBadge.className = "mt-1 px-3 py-1 text-xs font-bold rounded-full bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300";
    } else if (data.overall_score >= 58) {
        statusBadge.className = "mt-1 px-3 py-1 text-xs font-bold rounded-full bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300";
    } else {
        statusBadge.className = "mt-1 px-3 py-1 text-xs font-bold rounded-full bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300";
    }

    document.getElementById('res-score-quote').innerText = data.summary_quote;

    // Quick Vitals Badges
    document.getElementById('res-bmi-val').innerText = `${data.bmi} (${data.bmi_category.split(' ')[0]})`;
    document.getElementById('res-bp-val').innerText = `${data.input_echo.systolic_bp}/${data.input_echo.diastolic_bp} mmHg`;
    document.getElementById('res-glucose-val').innerText = `${data.input_echo.glucose} mg/dL`;

    // 2. Biological Age
    const bio = data.biological_age;
    document.getElementById('res-chrono-age').innerText = bio.chronological_age;
    document.getElementById('res-bio-age').innerText = bio.biological_age;
    
    const deltaBadge = document.getElementById('res-bio-delta');
    const diff = bio.difference;
    if (diff < 0) {
        deltaBadge.innerText = `${diff} yrs`;
        deltaBadge.className = "text-xs font-bold px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-900/60 dark:text-emerald-300";
    } else if (diff === 0) {
        deltaBadge.innerText = "±0 yrs";
        deltaBadge.className = "text-xs font-bold px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300";
    } else {
        deltaBadge.innerText = `+${diff} yrs`;
        deltaBadge.className = "text-xs font-bold px-1.5 py-0.5 rounded bg-rose-100 text-rose-700 dark:bg-rose-900/60 dark:text-rose-300";
    }

    document.getElementById('res-bio-status-badge').innerText = bio.status;
    document.getElementById('res-bio-comment').innerText = bio.comment;

    // 3. Risk Stratification Meters
    renderRiskMeters(data.risk_profile);

    // 4. Pillars of Health & Radar Chart
    renderPillarCards(data.pillars);
    renderRadarChart(data.pillars);

    // 5. Action Plan Rendering
    renderActionPlan(data.action_plan);

    // Trigger celebration confetti for high vitality scores!
    if (data.overall_score >= 85 && window.confetti) {
        confetti({
            particleCount: 70,
            spread: 60,
            origin: { y: 0.6 }
        });
    }

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Animate Score Counter
function animateScoreNumber(target) {
    const el = document.getElementById('res-score-number');
    let current = 0;
    const duration = 1000;
    const stepTime = 15;
    const steps = duration / stepTime;
    const increment = target / steps;

    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            el.innerText = target;
            clearInterval(timer);
        } else {
            el.innerText = Math.round(current);
        }
    }, stepTime);
}

// Animate SVG Gauge
function animateScoreCircle(score) {
    const circle = document.getElementById('score-circle-progress');
    const radius = 50;
    const circumference = 2 * Math.PI * radius; // ~314.159
    const offset = circumference - (score / 100) * circumference;
    circle.style.strokeDashoffset = offset;
}

// Render Risk Meters
function renderRiskMeters(risks) {
    const container = document.getElementById('risk-meters-container');
    container.innerHTML = '';

    risks.forEach(r => {
        let badgeColor = "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300";
        if (r.badge === 'warning') {
            badgeColor = "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300";
        } else if (r.badge === 'danger') {
            badgeColor = "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300";
        }

        const div = document.createElement('div');
        div.className = "p-3 rounded-2xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800 flex items-start justify-between gap-3";
        div.innerHTML = `
            <div>
                <div class="flex items-center gap-2">
                    <span class="text-xs font-bold text-slate-800 dark:text-slate-200">${r.name}</span>
                    <span class="px-2 py-0.5 text-[10px] font-bold rounded-full ${badgeColor}">${r.level} Risk</span>
                </div>
                <p class="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-normal">${r.description}</p>
            </div>
        `;
        container.appendChild(div);
    });
}

// Pillar Breakdown Cards
function renderPillarCards(pillars) {
    const container = document.getElementById('pillars-container');
    container.innerHTML = '';

    const pillarMeta = {
        cardiovascular: { label: "Cardiovascular Health", icon: "fa-heart-pulse", color: "from-rose-500 to-pink-500", desc: "Blood pressure, arterial tone, resting heart rate" },
        metabolic: { label: "Metabolic & Composition", icon: "fa-fire-flame-curved", color: "from-amber-500 to-orange-500", desc: "BMI, glycemic control, body mass balance" },
        fitness: { label: "Physical Fitness & Activity", icon: "fa-person-running", color: "from-emerald-500 to-teal-500", desc: "Active minutes, daily step volume, movement" },
        sleep_recovery: { label: "Sleep & Circadian Recovery", icon: "fa-moon", color: "from-indigo-500 to-purple-500", desc: "Sleep duration, restorative depth, timing" },
        mental_wellbeing: { label: "Mental Wellbeing & Stress", icon: "fa-brain", color: "from-teal-500 to-cyan-500", desc: "Perceived allostatic load, screen equilibrium" },
        habits_lifestyle: { label: "Habits & Clean Lifestyle", icon: "fa-droplet", color: "from-sky-500 to-blue-500", desc: "Hydration consistency, toxin avoidance" }
    };

    for (const [key, val] of Object.entries(pillars)) {
        const meta = pillarMeta[key] || { label: key, icon: "fa-circle", color: "from-emerald-500 to-teal-500", desc: "" };
        const score = Math.round(val);

        let badgeBg = "text-emerald-600 dark:text-emerald-400";
        if (score < 60) badgeBg = "text-rose-500";
        else if (score < 75) badgeBg = "text-amber-500";

        const card = document.createElement('div');
        card.className = "p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800/80 space-y-2";
        card.innerHTML = `
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-2">
                    <i class="fa-solid ${meta.icon} text-xs text-slate-400"></i>
                    <span class="text-xs font-bold text-slate-800 dark:text-slate-200">${meta.label}</span>
                </div>
                <span class="text-xs font-black ${badgeBg}">${score}/100</span>
            </div>
            <!-- Progress Bar -->
            <div class="w-full h-2 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                <div class="h-full rounded-full bg-gradient-to-r ${meta.color} transition-all duration-700" style="width: ${score}%;"></div>
            </div>
            <p class="text-[11px] text-slate-400 dark:text-slate-500">${meta.desc}</p>
        `;
        container.appendChild(card);
    }
}

// Chart.js Radar Chart
function renderRadarChart(pillars) {
    const ctx = document.getElementById('healthRadarChart').getContext('2d');
    const isDark = document.documentElement.classList.contains('dark');

    const labels = [
        'Cardiovascular',
        'Metabolic',
        'Physical Fitness',
        'Sleep & Recovery',
        'Mental Wellbeing',
        'Clean Habits'
    ];

    const values = [
        pillars.cardiovascular,
        pillars.metabolic,
        pillars.fitness,
        pillars.sleep_recovery,
        pillars.mental_wellbeing,
        pillars.habits_lifestyle
    ];

    if (radarChartInstance) {
        radarChartInstance.destroy();
    }

    const gridColor = isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.08)';
    const textColor = isDark ? '#94a3b8' : '#475569';

    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Your Health Profile',
                    data: values,
                    backgroundColor: 'rgba(16, 185, 129, 0.25)',
                    borderColor: '#10b981',
                    borderWidth: 2.5,
                    pointBackgroundColor: '#10b981',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#10b981',
                    pointRadius: 4
                },
                {
                    label: 'Optimal Longevity Benchmark',
                    data: [90, 90, 85, 90, 85, 95],
                    backgroundColor: 'rgba(99, 102, 241, 0.05)',
                    borderColor: 'rgba(99, 102, 241, 0.4)',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: gridColor },
                    grid: { color: gridColor },
                    pointLabels: {
                        color: textColor,
                        font: { size: 10, family: 'Inter', weight: 600 }
                    },
                    suggestedMin: 20,
                    suggestedMax: 100,
                    ticks: {
                        stepSize: 20,
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: textColor,
                        boxWidth: 12,
                        font: { size: 10, family: 'Inter' }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw} / 100`
                    }
                }
            }
        }
    });
}

function updateRadarTheme() {
    if (!radarChartInstance) return;
    const isDark = document.documentElement.classList.contains('dark');
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.08)';
    const textColor = isDark ? '#94a3b8' : '#475569';

    radarChartInstance.options.scales.r.angleLines.color = gridColor;
    radarChartInstance.options.scales.r.grid.color = gridColor;
    radarChartInstance.options.scales.r.pointLabels.color = textColor;
    radarChartInstance.options.plugins.legend.labels.color = textColor;
    radarChartInstance.update();
}

// Render Future Action Plan
function renderActionPlan(plan) {
    // 1. Quick Wins (Phase 1)
    const quickContainer = document.getElementById('quick-wins-container');
    quickContainer.innerHTML = '';
    plan.quick_wins.forEach((qw, idx) => {
        const item = document.createElement('div');
        item.className = "p-4 rounded-2xl bg-slate-800/80 border border-slate-700/80 hover:border-emerald-500/50 transition-all flex items-start gap-3.5";
        item.innerHTML = `
            <input type="checkbox" id="qw-${idx}" class="mt-1 w-4 h-4 rounded text-emerald-500 focus:ring-emerald-400 bg-slate-900 border-slate-600">
            <label for="qw-${idx}" class="cursor-pointer space-y-1">
                <div class="flex items-center gap-2">
                    <i class="fa-solid ${qw.icon} text-emerald-400 text-xs"></i>
                    <span class="text-xs font-bold text-white">${qw.title}</span>
                    <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">${qw.category}</span>
                </div>
                <p class="text-xs text-slate-300 leading-relaxed">${qw.action}</p>
            </label>
        `;
        quickContainer.appendChild(item);
    });

    // 2. Routine Architecture (Phase 2)
    const routineContainer = document.getElementById('routine-plan-container');
    routineContainer.innerHTML = '';
    plan.routine_plan.forEach(rp => {
        const item = document.createElement('div');
        item.className = "p-4 rounded-2xl bg-slate-800/80 border border-slate-700/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3";
        item.innerHTML = `
            <div class="space-y-1">
                <div class="flex items-center gap-2">
                    <span class="text-xs font-bold text-teal-300">${rp.pillar}</span>
                </div>
                <p class="text-xs text-slate-300 leading-relaxed">${rp.detail}</p>
            </div>
            <span class="shrink-0 text-xs font-semibold px-2.5 py-1 rounded-full bg-teal-950 border border-teal-800/60 text-teal-300 self-start sm:self-center">
                <i class="fa-regular fa-clock mr-1"></i> ${rp.frequency}
            </span>
        `;
        routineContainer.appendChild(item);
    });

    // 3. Long Term Goals (Phase 3)
    const longContainer = document.getElementById('longterm-plan-container');
    longContainer.innerHTML = '';
    plan.long_term_goals.forEach(lg => {
        const item = document.createElement('div');
        item.className = "p-4 rounded-2xl bg-slate-800/80 border border-slate-700/80 space-y-2";
        item.innerHTML = `
            <div class="flex items-center justify-between gap-2">
                <span class="text-xs font-bold text-indigo-300">${lg.milestone}</span>
                <span class="text-[10px] font-semibold px-2 py-0.5 rounded bg-indigo-950 border border-indigo-800 text-indigo-300">
                    ${lg.target_timeframe}
                </span>
            </div>
            <p class="text-xs font-medium text-white">${lg.metric}</p>
            <p class="text-[11px] text-emerald-400 flex items-center gap-1.5">
                <i class="fa-solid fa-chart-line"></i> ${lg.impact}
            </p>
        `;
        longContainer.appendChild(item);
    });

    // 4. Clinical Actions
    const clinContainer = document.getElementById('clinical-actions-container');
    clinContainer.innerHTML = '';
    plan.clinical_actions.forEach(ca => {
        const item = document.createElement('div');
        item.className = "p-3 rounded-xl bg-slate-900/80 border border-slate-700/60 flex flex-col sm:flex-row sm:items-center justify-between gap-2";
        item.innerHTML = `
            <div class="space-y-0.5">
                <div class="flex items-center gap-2">
                    <span class="text-xs font-bold text-rose-300">${ca.test}</span>
                </div>
                <p class="text-xs text-slate-300">${ca.reason}</p>
            </div>
            <span class="shrink-0 text-[11px] font-medium px-2 py-0.5 rounded bg-slate-800 text-slate-400 self-start sm:self-center">
                ${ca.urgency}
            </span>
        `;
        clinContainer.appendChild(item);
    });
}

// Initial bootstrap
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    updateLiveBMI();
    updateLiveBP();
    updateStressDisplay();
    fetchPresets();
});
