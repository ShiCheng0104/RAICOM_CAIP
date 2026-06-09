const state = {
  health: null,
  models: [],
  adapters: [],
  metrics: null,
  leaderboard: [],
  selectedStreamEvent: null,
  graphSummary: null,
  graphGroups: [],
  selectedGraphGroup: null,
  selectedGraphGroupDetail: null,
  demoActions: [],
  demoPollTimer: null,
  modelSort: { key: "pr_auc", direction: "desc" },
  fraudRingGraphData: null,
  fraudRingZoom: 1,
  selectedMetricsModel: null,
  graphMiningLoading: false,
  thresholdResult: null,
  graphSageMetrics: null,
};

const FRAUD_RING_VIEWBOX = {
  width: 820,
  height: 420,
  centerX: 410,
  centerY: 210,
  minX: 42,
  maxX: 778,
  minY: 42,
  maxY: 378,
};

const sampleRecord = {
  transaction_id: "sample_transaction_001",
  timestamp: "2026-03-20T10:15:03",
  source: "demo",
  amount: 8800,
  amount_bucket: "large",
  currency: "USD",
  txn_type: "transfer",
  channel: "app",
  payment_method: "bank_transfer",
  merchant_category: "digital_goods",
  payer_id: "demo_user_001",
  payee_id: "demo_payee_009",
  merchant_id: "demo_merchant_003",
  device_id: "demo_device_007",
  ip_id: "demo_ip_018",
  payer_country: "CN",
  payee_country: "US",
  hour: 10,
  day_of_week: 4,
  is_weekend: 0,
  is_night: 0,
  window_features: {
    user_txn_count_5min: 6,
    user_amount_sum_5min: 26800,
    user_txn_count_1h: 11,
    user_amount_sum_1h: 45200,
    user_unique_payee_count_1h: 5,
    device_unique_user_count_10min: 4,
    ip_unique_user_count_10min: 8,
    merchant_txn_count_1h: 38,
    merchant_amount_sum_1h: 188000,
    merchant_unique_user_count_1h: 31,
  },
  user_profile: {
    txn_count: 6,
    total_amount: 30148.94,
    avg_amount: 4541.44,
    common_device_count: 6,
    common_ip_count: 6,
    home_country: "CN",
    account_age_days: 892,
  },
  device_profile: {
    bind_user_count: 12,
    txn_count: 20,
    device_type: "mobile",
    os: "Android",
    is_emulator: 0,
    is_proxy_device: 1,
  },
  ip_profile: {
    bind_user_count: 15,
    country: "US",
    is_proxy: 1,
    is_vpn: 0,
  },
  merchant_profile: {
    merchant_category: "digital_goods",
    merchant_country: "US",
    txn_count: 64,
    unique_user_count: 64,
    total_amount: 54809.54,
    avg_amount: 374.9,
    chargeback_rate: 0.0054,
    complaint_rate: 0.051,
  },
  graph_features: {
    payer_graph_degree: 12,
    payer_graph_fraud_edge_count: 4,
    payer_graph_fraud_edge_ratio: 0.33,
    payer_fraud_group_mining_id: "GM_DEMO_001",
    payer_graph_mining_group_risk_score: 0.82,
    payer_graph_mining_group_user_count: 9,
    payer_graph_mining_shared_device_count: 3,
    payer_graph_mining_shared_ip_count: 2,
    payer_graph_mining_evidence_count: 5,
    payee_graph_degree: 7,
    merchant_graph_degree: 44,
    merchant_graph_mining_group_risk_score: 0.62,
    device_graph_degree: 17,
    device_graph_mining_group_risk_score: 0.76,
    ip_graph_degree: 29,
    ip_graph_mining_group_risk_score: 0.71,
  },
};

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  if (typeof value === "number") return value.toFixed(digits);
  return String(value);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function bind(id, event, handler) {
  const element = document.getElementById(id);
  if (element) element.addEventListener(event, handler);
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function pct(value) {
  return value === undefined || value === null || Number.isNaN(Number(value)) ? "--" : `${(Number(value) * 100).toFixed(2)}%`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function timestampForInput(value) {
  const source = value || new Date().toISOString();
  return String(source).slice(0, 16);
}

function timestampFromInput(value) {
  return value ? new Date(value).toISOString() : new Date().toISOString();
}

function displayModelName(metrics) {
  if (!metrics) return "--";
  return metrics.model_dir || metrics.name || metrics.model_name || "--";
}

function normalizeCountry(value, fieldLabel) {
  const country = String(value || "").trim().toUpperCase();
  if (!/^[A-Z]{2}$/.test(country)) {
    throw new Error(`${fieldLabel}请输入 ISO 两位国家代码，例如 CN、US、SG。`);
  }
  return country;
}

function statusLabel(status) {
  const labels = {
    running: "运行中",
    succeeded: "成功",
    failed: "失败",
  };
  return labels[status] || status || "--";
}

function statusClass(status) {
  if (status === "succeeded") return "status-success";
  if (status === "failed") return "status-failed";
  return "status-running";
}

function reasonExplanation(code) {
  const labels = {
    amount_deviation: "交易金额明显偏离该用户或场景的常见水平。",
    high_frequency_user_window: "用户在短时间窗口内交易频次偏高。",
    shared_device: "设备被多个用户共享，可能存在设备农场或团伙操作。",
    risky_ip: "IP 具备代理、多账号或异常来源风险。",
    merchant_concentration: "商户侧交易集中度较高，可能存在套现或集中攻击。",
    graph_ring_signal: "交易主体命中图挖掘中的疑似团伙链路。",
    cross_border: "收付款国家不同，跨境链路风险更高。",
    night_transaction: "交易发生在夜间等异常时段。",
  };
  return labels[code] || "模型或规则认为该信号对本次风险判断有贡献。";
}

function renderReasonList(reasons) {
  if (!reasons || !reasons.length) {
    return '<span class="evidence-chip">暂无理由码</span>';
  }
  return reasons.map((reason) => `
    <div class="reason-item">
      <span class="reason-chip">${escapeHtml(reason)}</span>
      <small>${escapeHtml(reasonExplanation(reason))}</small>
    </div>
  `).join("");
}

async function api(path, options = {}) {
  const apiKey = window.localStorage.getItem("fraudsim_api_key") || "";
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (apiKey) headers["X-API-Key"] = apiKey;
  const response = await fetch(path, {
    ...options,
    headers,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function loadApiKey() {
  const input = document.getElementById("apiKeyInput");
  if (input) input.value = window.localStorage.getItem("fraudsim_api_key") || "";
}

function saveApiKey() {
  const input = document.getElementById("apiKeyInput");
  if (!input) return;
  const value = input.value.trim();
  if (value) {
    window.localStorage.setItem("fraudsim_api_key", value);
  } else {
    window.localStorage.removeItem("fraudsim_api_key");
  }
  refreshAll();
}

async function refreshAll() {
  const apiDot = document.getElementById("apiDot");
  const apiState = document.getElementById("apiState");
  try {
    const [models, metrics, leaderboard, demoActions, graphSage] = await Promise.all([
      api("/models"),
      api("/metrics"),
      api("/leaderboard"),
      api("/demo/actions"),
      api("/graphsage/metrics"),
    ]);
    state.health = models.active;
    state.models = models.models || [];
    state.adapters = models.adapters || [];
    state.metrics = metrics;
    state.leaderboard = leaderboard.rows || [];
    state.demoActions = demoActions.actions || [];
    state.graphSageMetrics = graphSage.metrics || null;
    await refreshGraphMining(false);
    apiDot.className = "dot ok";
    apiState.textContent = "API 正常";
    render();
  } catch (error) {
    apiDot.className = "dot bad";
    apiState.textContent = "API 异常";
    document.getElementById("predictResult").textContent = error.message;
  }
}

function render() {
  renderOverview();
  renderModels();
  renderMetrics();
  renderGraph(eventValue({ value: state.selectedStreamEvent }) || sampleRecord);
  renderGraphMining();
}

function riskClass(level) {
  if (level === "high") return "risk-high";
  if (level === "medium") return "risk-medium";
  return "risk-low";
}

function eventValue(message) {
  return message && message.value && typeof message.value === "object" ? message.value : {};
}

async function refreshStream() {
  const topic = document.getElementById("topicSelect").value;
  const list = document.getElementById("streamRows");
  list.textContent = "拉取中";
  try {
    const data = await api(`/topics/${topic}/recent?limit=30&timeout_ms=1000`);
    const messages = data.messages || [];
    if (!messages.length) {
      list.textContent = "暂无消息";
      return;
    }
    list.innerHTML = "";
    for (const message of messages.slice().reverse()) {
      const value = eventValue(message);
      const txId = value.transaction_id || value.event?.transaction_id || message.key || "(no transaction_id)";
      const score = value.risk_score !== undefined ? fmt(value.risk_score) : "--";
      const level = value.risk_level || "--";
      const card = document.createElement("div");
      card.className = "stream-card";
      card.innerHTML = `
        <div class="stream-card-head">
          <strong class="labeled-id"><span>交易ID</span><em title="${escapeHtml(txId)}">${escapeHtml(txId)}</em></strong>
          <span class="${riskClass(level)}">${level}</span>
        </div>
        <div class="stream-meta">score ${score} · ${value.decision || "--"} · p${message.partition}/o${message.offset}</div>
        <div class="stream-meta">${escapeHtml((value.reason_codes || []).join(", ") || value.feedback_created_at || "")}</div>
      `;
      card.addEventListener("click", () => selectStreamEvent(message));
      list.appendChild(card);
    }
  } catch (error) {
    list.textContent = error.message;
  }
}

function selectStreamEvent(message) {
  state.selectedStreamEvent = message.value;
  const value = eventValue(message);
  document.getElementById("feedbackTxn").value = value.transaction_id || value.event?.transaction_id || "";
  document.getElementById("feedbackNote").value = JSON.stringify(value, null, 2).slice(0, 1200);
  renderGraph(value);
}

async function submitFeedback() {
  const result = document.getElementById("feedbackResult");
  const transactionId = document.getElementById("feedbackTxn").value.trim();
  if (!transactionId) {
    result.textContent = "请先填写交易 ID";
    return;
  }
  const payload = {
    transaction_id: transactionId,
    reviewed_is_fraud: Number(document.getElementById("feedbackLabel").value),
    reviewer: "dashboard",
    note: document.getElementById("feedbackNote").value.trim() || null,
    event: state.selectedStreamEvent,
  };
  try {
    const response = await api("/feedback", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    result.innerHTML = `
      <div class="demo-status-card">
        <div class="demo-status-head">
          <strong>反馈已写入</strong>
          <span class="run-status status-success">成功</span>
        </div>
        <div class="demo-status-grid">
          <span>交易 ID</span><strong>${escapeHtml(transactionId)}</strong>
          <span>标签</span><strong>${payload.reviewed_is_fraud ? "欺诈" : "正常"}</strong>
          <span>Topic</span><strong>${escapeHtml(response.topic || "feedback_events")}</strong>
          <span>Offset</span><strong>${escapeHtml(response.offset ?? "--")}</strong>
        </div>
      </div>
    `;
  } catch (error) {
    result.textContent = error.message;
  }
}

function renderOverview() {
  const health = state.health || {};
  const metrics = (state.metrics && state.metrics.metrics) || {};
  const highConfidence = (state.graphSummary && state.graphSummary.high_confidence) || {};
  setText("activeModel", health.model_name || "--");
  setText("activeVersion", health.model_version || "--");
  setText("featureCount", fmt(health.feature_count, 0));
  setText("prAuc", fmt(metrics.pr_auc));
  setText("f1Score", fmt(metrics.f1));
  setText("loadedState", health.loaded ? "loaded" : "not loaded");
  setText("modelPath", health.model_path || "--");
  setText("loadedAt", health.loaded_at || "--");
  setText("adapterList", state.adapters.join(", ") || "--");
  setText("overviewModelCount", fmt(state.models.length, 0));
  setText("overviewGraphCount", fmt(Number((state.graphSummary && state.graphSummary.detected_group_count) || state.graphGroups.length || 0), 0));
  setText("overviewGraphPurity", pct(highConfidence.flagged_row_precision_vs_injection ?? state.graphSummary?.flagged_row_precision_vs_injection));
  setText("overviewEvalCount", fmt(state.leaderboard.length, 0));
  setText("overviewGraphSagePr", fmt(state.graphSageMetrics?.pr_auc));

  const selector = document.getElementById("overviewModelSelect");
  if (selector) {
    const current = selector.value;
    selector.innerHTML = "";
    for (const model of state.models) {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = `${model.name} (${model.model_name || "adapter"})`;
      option.selected = model.loaded || model.name === current;
      selector.appendChild(option);
    }
    if (!state.models.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "暂无模型";
      selector.appendChild(option);
    }
  }
}

function renderModels() {
  const body = document.getElementById("modelRows");
  setText("modelCount", `${state.models.length} models`);
  if (!body) return;
  body.innerHTML = "";
  const { key, direction } = state.modelSort;
  const sorted = [...state.models].sort((a, b) => {
    const av = a[key];
    const bv = b[key];
    if (typeof av === "number" || typeof bv === "number" || typeof av === "boolean" || typeof bv === "boolean") {
      return (asNumber(av) - asNumber(bv)) * (direction === "asc" ? 1 : -1);
    }
    return String(av || "").localeCompare(String(bv || ""), "zh-CN") * (direction === "asc" ? 1 : -1);
  });
  document.querySelectorAll(".sort-btn").forEach((button) => {
    const marker = button.querySelector("span");
    const active = button.dataset.sortKey === key;
    button.classList.toggle("active", active);
    if (marker) marker.textContent = active ? (direction === "asc" ? "↑" : "↓") : "↕";
  });
  for (const model of sorted) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${model.name || "--"}</td>
      <td>${model.model_name || "--"}</td>
      <td>${model.dataset || "--"}</td>
      <td>${fmt(model.feature_count, 0)}</td>
      <td>${fmt(model.pr_auc)}</td>
      <td>${fmt(model.roc_auc)}</td>
      <td>${fmt(model.f1)}</td>
      <td>${model.loaded ? '<span class="status-loaded">已加载</span>' : "待加载"}</td>
      <td><button class="secondary-btn" data-model="${model.name}" type="button">${model.loaded ? "重载" : "加载"}</button></td>
    `;
    body.appendChild(row);
  }
  body.querySelectorAll("button[data-model]").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await api("/models/activate", {
          method: "POST",
          body: JSON.stringify({ model_name: button.dataset.model }),
        });
        await refreshAll();
      } catch (error) {
        alert(error.message);
      } finally {
        button.disabled = false;
      }
    });
  });
}

function toggleModelSort(key) {
  if (state.modelSort.key === key) {
    state.modelSort.direction = state.modelSort.direction === "asc" ? "desc" : "asc";
  } else {
    state.modelSort = {
      key,
      direction: ["name", "model_name", "dataset"].includes(key) ? "asc" : "desc",
    };
  }
  renderModels();
}

function renderMetrics() {
  const metricsBox = document.getElementById("metricsReadable");
  const activeMetrics = (state.metrics && state.metrics.metrics) || {};
  const selectedMetrics = state.selectedMetricsModel || activeMetrics;
  const metrics = selectedMetrics || {};
  const isSelectedLeaderboard = Boolean(state.selectedMetricsModel);
  setText("metricsTitle", isSelectedLeaderboard ? `模型指标：${displayModelName(metrics)}` : "当前模型指标");
  if (metricsBox) {
    const thresholds = metrics.thresholds || {};
    const calibration = metrics.threshold_calibration || {};
    const calibrationRows = [
      ["默认高风险", thresholds.high, metrics.precision_at_threshold, metrics.recall_at_threshold, metrics.false_positive_rate_at_threshold],
      ["最佳 F1", calibration.best_f1?.threshold, calibration.best_f1?.precision, calibration.best_f1?.recall, calibration.best_f1?.fpr],
      ["精度 >= 95%", calibration.precision_at_least_0_95?.threshold, calibration.precision_at_least_0_95?.precision, calibration.precision_at_least_0_95?.recall, calibration.precision_at_least_0_95?.fpr],
      ["召回 >= 90%", calibration.recall_at_least_0_90?.threshold, calibration.recall_at_least_0_90?.precision, calibration.recall_at_least_0_90?.recall, calibration.recall_at_least_0_90?.fpr],
    ];
    metricsBox.innerHTML = `
      <div class="metric-card-grid">
        ${renderReadableMetric("模型", displayModelName(metrics), metrics.dataset || "")}
        ${renderReadableMetric("训练/验证/测试", `${fmt(metrics.train_rows, 0)} / ${fmt(metrics.valid_rows, 0)} / ${fmt(metrics.test_rows, 0)}`, "监督样本")}
        ${renderReadableMetric("PR-AUC", fmt(metrics.pr_auc), "欺诈样本不均衡时优先看")}
        ${renderReadableMetric("ROC-AUC", fmt(metrics.roc_auc), "整体排序能力")}
        ${renderReadableMetric("F1", fmt(metrics.f1 ?? metrics.f1_at_0_8), "精度和召回的折中")}
        ${renderReadableMetric("Top 1% 召回", fmt(metrics.recall_at_top_1pct), "最高风险队列覆盖")}
      </div>
      <div class="detail-section">
        <h5>阈值校准</h5>
        <div class="evidence-table-wrap">
          <table class="evidence-table readable-metric-table">
            <thead><tr><th>策略</th><th>阈值</th><th>精度</th><th>召回</th><th>FPR</th></tr></thead>
            <tbody>
              ${calibrationRows.map(([label, threshold, precision, recall, fpr]) => `
                <tr>
                  <td>${escapeHtml(label)}</td>
                  <td>${fmt(threshold)}</td>
                  <td>${fmt(precision)}</td>
                  <td>${fmt(recall)}</td>
                  <td>${fmt(fpr)}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
      <div class="detail-section">
        <h5>评估产物</h5>
        <div class="chip-row">
          <span class="evidence-chip">窗口特征: ${metrics.with_window_features === false ? "未启用" : "已启用"}</span>
          <span class="evidence-chip">图特征: ${metrics.with_graph_features === false ? "未启用" : "已启用"}</span>
          <span class="evidence-chip">团伙挖掘特征: ${metrics.with_graph_mining_features ? "已启用" : "未启用"}</span>
          <span class="evidence-chip">${isSelectedLeaderboard ? "来自 leaderboard 点选" : "当前在线模型"}</span>
        </div>
      </div>
    `;
  }
  const board = document.getElementById("leaderboard");
  if (!board) return;
  board.innerHTML = "";
  for (const item of state.leaderboard) {
    const row = document.createElement("div");
    const selectedName = displayModelName(state.selectedMetricsModel);
    row.className = `leader-row clickable${displayModelName(item) === selectedName ? " selected" : ""}`;
    row.setAttribute("role", "button");
    row.setAttribute("tabindex", "0");
    row.innerHTML = `
      <strong>${displayModelName(item)}<br><span>${item.dataset || ""}</span></strong>
      <span>PR ${fmt(item.pr_auc)}</span>
      <span>ROC ${fmt(item.roc_auc)}</span>
      <span>F1 ${fmt(item.f1 ?? item.f1_at_0_8)}</span>
    `;
    row.addEventListener("click", () => {
      state.selectedMetricsModel = item;
      renderMetrics();
    });
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        state.selectedMetricsModel = item;
        renderMetrics();
      }
    });
    board.appendChild(row);
  }
  if (!state.leaderboard.length) {
    board.textContent = "暂无 leaderboard";
  }
  renderModelChart();
  renderGraphSageMetrics();
}

function renderGraphSageMetrics() {
  const box = document.getElementById("graphSageMetrics");
  if (!box) return;
  const metrics = state.graphSageMetrics;
  if (!metrics) {
    box.textContent = "暂无 GraphSAGE 实验产物";
    return;
  }
  box.innerHTML = `
    <div class="metric-card-grid">
      ${renderReadableMetric("采样节点", fmt(metrics.sampled_nodes, 0), `边 ${fmt(metrics.sampled_edges, 0)}`)}
      ${renderReadableMetric("样本欺诈率", pct(metrics.fraud_node_ratio), "受控图采样")}
      ${renderReadableMetric("PR-AUC", fmt(metrics.pr_auc), "节点风险排序")}
      ${renderReadableMetric("ROC-AUC", fmt(metrics.roc_auc), "节点区分能力")}
      ${renderReadableMetric("F1@0.5", fmt(metrics.f1_at_0_5), "旁路阈值")}
      ${renderReadableMetric("用途", "旁路实验", "不直接改变线上决策")}
    </div>
  `;
}

function renderReadableMetric(label, value, hint) {
  return `
    <div class="readable-metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(hint || "")}</small>
    </div>
  `;
}

function renderModelChart() {
  const chart = document.getElementById("modelChart");
  if (!chart) return;
  chart.innerHTML = "";
  const rows = [...state.leaderboard]
    .filter((item) => item.pr_auc !== undefined || item.f1 !== undefined || item.f1_at_0_8 !== undefined)
    .sort((a, b) => (b.pr_auc || 0) - (a.pr_auc || 0))
    .slice(0, 8);
  if (!rows.length) {
    chart.textContent = "暂无多模型指标";
    return;
  }
  for (const item of rows) {
    const pr = Number(item.pr_auc || 0);
    const f1 = Number(item.f1 ?? item.f1_at_0_8 ?? 0);
    const label = item.model_dir || item.model_name || "--";
    const row = document.createElement("div");
    row.className = "chart-row";
    row.innerHTML = `
      <div class="chart-label">${escapeHtml(label)}</div>
      <div class="bar-track"><span style="width:${Math.max(2, pr * 100)}%"></span></div>
      <strong>PR ${fmt(pr)}</strong>
      <div class="bar-track alt"><span style="width:${Math.max(2, f1 * 100)}%"></span></div>
      <strong>F1 ${fmt(f1)}</strong>
    `;
    chart.appendChild(row);
  }
}

function updateThresholdLabels() {
  const medium = document.getElementById("mediumThreshold");
  const high = document.getElementById("highThreshold");
  setText("mediumThresholdValue", Number(medium?.value || 0.5).toFixed(2));
  setText("highThresholdValue", Number(high?.value || 0.8).toFixed(2));
}

function renderThresholdResult(result) {
  const box = document.getElementById("thresholdResult");
  if (!box) return;
  if (!result) {
    box.textContent = "点击评估阈值查看业务影响";
    return;
  }
  const decisions = result.decisions || {};
  const metrics = result.high_risk_metrics || {};
  const workload = result.workload || {};
  box.innerHTML = `
    <div class="metric-card-grid threshold-metrics">
      ${renderReadableMetric("高风险拦截", fmt(decisions.reject, 0), `拦截率 ${pct(workload.reject_rate)}`)}
      ${renderReadableMetric("人工审核", fmt(decisions.review, 0), `审核率 ${pct(workload.review_rate)}`)}
      ${renderReadableMetric("拦截精度", pct(metrics.precision), `误拦 ${fmt(metrics.false_positive, 0)} 笔`)}
      ${renderReadableMetric("欺诈召回", pct(metrics.recall), `漏过 ${fmt(metrics.missed_fraud, 0)} 笔`)}
      ${renderReadableMetric("F1", fmt(metrics.f1), "精度与召回综合")}
      ${renderReadableMetric("误报率 FPR", pct(metrics.false_positive_rate), `测试样本 ${fmt(result.rows, 0)} 笔`)}
    </div>
    <div class="decision-bar" aria-label="阈值决策分布">
      <span class="pass" style="width:${(decisions.pass / result.rows) * 100}%"></span>
      <span class="review" style="width:${(decisions.review / result.rows) * 100}%"></span>
      <span class="reject" style="width:${(decisions.reject / result.rows) * 100}%"></span>
    </div>
    <div class="decision-legend">
      <span><i class="pass"></i>放行 ${fmt(decisions.pass, 0)}</span>
      <span><i class="review"></i>审核 ${fmt(decisions.review, 0)}</span>
      <span><i class="reject"></i>拦截 ${fmt(decisions.reject, 0)}</span>
    </div>
  `;
}

async function evaluateThresholds() {
  const box = document.getElementById("thresholdResult");
  const medium = Number(document.getElementById("mediumThreshold").value);
  const high = Number(document.getElementById("highThreshold").value);
  if (medium >= high) {
    box.textContent = "人工审核阈值必须低于高风险拦截阈值";
    return;
  }
  box.textContent = "正在计算测试集阈值影响...";
  try {
    state.thresholdResult = await api("/threshold-sandbox", {
      method: "POST",
      body: JSON.stringify({ medium_threshold: medium, high_threshold: high }),
    });
    renderThresholdResult(state.thresholdResult);
  } catch (error) {
    box.textContent = error.message;
  }
}

function graphNodesFromEvent(value) {
  const risk = value.risk_score !== undefined ? `score ${fmt(value.risk_score)}` : "待评分";
  return [
    { id: "payer", label: value.payer_id || "payer", type: "用户", x: 90, y: 150 },
    { id: "payee", label: value.payee_id || "payee", type: "收款方", x: 300, y: 74 },
    { id: "merchant", label: value.merchant_id || "merchant", type: "商户", x: 300, y: 226 },
    { id: "device", label: value.device_id || "device", type: "设备", x: 510, y: 92 },
    { id: "ip", label: value.ip_id || "ip", type: "IP", x: 510, y: 210 },
    { id: "risk", label: value.decision || risk, type: value.risk_level || "risk", x: 650, y: 150 },
  ];
}

function renderGraph(value) {
  const svg = document.getElementById("graphSvg");
  const detail = document.getElementById("graphDetail");
  const hint = document.getElementById("graphHint");
  if (!svg || !detail || !hint) return;
  const event = value && Object.keys(value).length ? value : sampleRecord;
  const nodes = graphNodesFromEvent(event);
  const byId = Object.fromEntries(nodes.map((node) => [node.id, node]));
  const edges = [
    ["payer", "payee", "pay"],
    ["payer", "merchant", "consume"],
    ["payer", "device", "uses"],
    ["payer", "ip", "from"],
    ["merchant", "risk", "score"],
    ["device", "risk", "signal"],
    ["ip", "risk", "signal"],
  ];
  svg.innerHTML = "";
  for (const [source, target, label] of edges) {
    const a = byId[source];
    const b = byId[target];
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y);
    line.setAttribute("class", "graph-edge");
    svg.appendChild(line);
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", (a.x + b.x) / 2);
    text.setAttribute("y", (a.y + b.y) / 2 - 6);
    text.setAttribute("class", "graph-edge-label");
    text.textContent = label;
    svg.appendChild(text);
  }
  for (const node of nodes) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", `graph-node graph-${node.id}`);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", node.x);
    circle.setAttribute("cy", node.y);
    circle.setAttribute("r", "34");
    const type = document.createElementNS("http://www.w3.org/2000/svg", "text");
    type.setAttribute("x", node.x);
    type.setAttribute("y", node.y - 4);
    type.setAttribute("class", "graph-node-type");
    type.textContent = node.type;
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", node.x);
    label.setAttribute("y", node.y + 16);
    label.setAttribute("class", "graph-node-label");
    label.textContent = String(node.label).slice(0, 16);
    group.appendChild(circle);
    group.appendChild(type);
    group.appendChild(label);
    svg.appendChild(group);
  }
  hint.textContent = event.transaction_id || "样例交易";
  detail.innerHTML = renderStreamGraphDetail(event);
}

function renderStreamGraphDetail(event) {
  const reasons = event.reason_codes || [];
  const windowFeatures = event.window_features || {};
  const graphFeatures = event.graph_features || {};
  return `
    <div class="detail-hero">
      <div>
        <span class="detail-eyebrow scroll-value">交易ID: ${escapeHtml(event.transaction_id || "sample")}</span>
        <h4>${escapeHtml(event.risk_level || "待评分")} · ${fmt(event.risk_score)}</h4>
      </div>
      <span class="run-status ${event.decision === "reject" ? "status-failed" : event.decision === "review" ? "status-running" : "status-success"}">${escapeHtml(event.decision || "preview")}</span>
    </div>
    <div class="mini-grid">
      ${renderMetricMini("付款方ID", event.payer_id || "--")}
      ${renderMetricMini("金额", fmt(event.amount, 2))}
      ${renderMetricMini("5分钟笔数", fmt(windowFeatures.user_txn_count_5min, 0))}
      ${renderMetricMini("图风险", fmt(graphFeatures.payer_graph_mining_group_risk_score))}
    </div>
    <div class="detail-section">
      <h5>理由码</h5>
      <div class="reason-list">${renderReasonList(reasons)}</div>
    </div>
  `;
}

async function refreshGraphMining(shouldRender = true) {
  try {
    const [summary, groups] = await Promise.all([
      api("/graph/mining/summary"),
      api("/graph/mining/groups?limit=12"),
    ]);
    state.graphSummary = summary || {};
    state.graphGroups = groups.groups || [];
    if (!state.selectedGraphGroup && state.graphGroups.length) {
      state.selectedGraphGroup = state.graphGroups[0];
      await selectGraphGroup(state.selectedGraphGroup, false);
    }
    if (shouldRender) renderGraphMining();
  } catch (error) {
    state.graphSummary = { error: error.message };
    state.graphGroups = [];
    if (shouldRender) renderGraphMining();
  }
}

async function selectGraphGroup(group, shouldRender = true) {
  state.selectedGraphGroup = group;
  state.selectedGraphGroupDetail = null;
  updateSelectedGroupCards();
  const groupId = group && group.fraud_group_mining_id;
  if (!groupId) {
    if (shouldRender) renderFraudRingGraph(group);
    return;
  }
  state.graphMiningLoading = true;
  showGraphMiningLoading(true);
  try {
    state.selectedGraphGroupDetail = await api(`/graph/mining/groups/${encodeURIComponent(groupId)}`);
  } catch (error) {
    state.selectedGraphGroupDetail = { group, error: error.message };
  } finally {
    state.graphMiningLoading = false;
    showGraphMiningLoading(false);
  }
  if (shouldRender) renderFraudRingGraph(state.selectedGraphGroupDetail || group);
}

function showGraphMiningLoading(isLoading) {
  const overlay = document.getElementById("graphMiningLoading");
  if (overlay) overlay.classList.toggle("hidden", !isLoading);
}

function updateSelectedGroupCards() {
  const selectedGroupId = state.selectedGraphGroup && state.selectedGraphGroup.fraud_group_mining_id;
  document.querySelectorAll(".group-card").forEach((card) => {
    const title = card.querySelector(".group-card-head strong")?.textContent;
    card.classList.toggle("selected", Boolean(selectedGroupId && title === selectedGroupId));
  });
}

function parseMaybeJson(value, fallback) {
  if (Array.isArray(value) || (value && typeof value === "object")) return value;
  if (typeof value !== "string" || !value) return fallback;
  try {
    return JSON.parse(value);
  } catch (_) {
    return fallback;
  }
}

function renderGraphMining() {
  const summary = state.graphSummary || {};
  const highConfidence = summary.high_confidence || {};
  const groups = state.graphGroups || [];
  const groupCount = summary.detected_group_count ?? groups.length;
  const groupRecall = highConfidence.injected_group_recall ?? summary.injected_group_recall;
  const rowRecall = highConfidence.injected_row_recall ?? summary.injected_row_recall;
  const purity = highConfidence.flagged_row_precision_vs_injection ?? summary.flagged_row_precision_vs_injection;
  const groupCountEl = document.getElementById("graphGroupCount");
  if (!groupCountEl) return;
  groupCountEl.textContent = fmt(Number(groupCount || 0), 0);
  document.getElementById("graphGroupRecall").textContent = groupRecall === undefined ? "--" : fmt(groupRecall);
  document.getElementById("graphRowRecall").textContent = rowRecall === undefined ? "--" : fmt(rowRecall);
  document.getElementById("graphPurity").textContent = purity === undefined ? "--" : fmt(purity);

  const list = document.getElementById("graphGroupRows");
  list.innerHTML = "";
  if (!groups.length) {
    list.textContent = summary.error || "暂无图挖掘结果";
    renderFraudRingGraph(null);
    return;
  }
  const selectedGroupId = state.selectedGraphGroup && state.selectedGraphGroup.fraud_group_mining_id;
  for (const group of groups) {
    const row = document.createElement("div");
    row.className = `group-card${group.fraud_group_mining_id === selectedGroupId ? " selected" : ""}`;
    row.innerHTML = `
      <div class="group-card-head">
        <strong>${escapeHtml(group.fraud_group_mining_id || "--")}</strong>
        <span>${fmt(Number(group.graph_mining_group_risk_score || 0))}</span>
      </div>
      <div class="stream-meta">users ${fmt(Number(group.user_count || 0), 0)} · resources ${fmt(Number(group.resource_count || 0), 0)} · seeds ${fmt(Number(group.fraud_seed_count || 0), 0)}</div>
      <div class="stream-meta">device ${fmt(Number(group.shared_device_count || 0), 0)} · ip ${fmt(Number(group.shared_ip_count || 0), 0)} · evidence ${fmt(Number(group.evidence_count || 0), 0)}</div>
    `;
    row.addEventListener("click", async () => {
      row.classList.add("selected-pulse");
      await selectGraphGroup(group);
    });
    list.appendChild(row);
  }
  renderFraudRingGraph(state.selectedGraphGroupDetail || state.selectedGraphGroup || groups[0]);
}

function nodeTypeLabel(type) {
  const labels = {
    fraud_group: "团伙",
    user: "用户",
    device: "设备",
    ip: "IP",
    merchant: "商户",
    payee: "收款方",
    resource: "资源",
    risk: "风险",
  };
  return labels[type] || type || "节点";
}

function layoutSubgraph(subgraph) {
  const nodes = (subgraph && subgraph.nodes ? subgraph.nodes : []).map((node) => ({ ...node }));
  const users = nodes.filter((node) => node.type === "user");
  const resources = nodes.filter((node) => node.role === "shared_resource");
  const center = nodes.find((node) => node.role === "center");
  const risk = nodes.find((node) => node.role === "risk");
  if (center) {
    center.x = FRAUD_RING_VIEWBOX.centerX;
    center.y = FRAUD_RING_VIEWBOX.centerY;
  }
  if (risk) {
    risk.x = FRAUD_RING_VIEWBOX.centerX;
    risk.y = 366;
  }
  users.slice(0, 10).forEach((node, index) => {
    node.x = index % 2 === 0 ? 70 : 150;
    node.y = 50 + index * Math.max(34, Math.min(58, 320 / Math.max(users.length, 1)));
  });
  resources.slice(0, 12).forEach((node, index) => {
    node.x = index % 2 === 0 ? 750 : 670;
    node.y = 48 + index * Math.max(30, Math.min(52, 330 / Math.max(resources.length, 1)));
  });
  return { nodes, edges: (subgraph && subgraph.edges) || [] };
}

function renderFraudRingGraph(payload) {
  const svg = document.getElementById("fraudRingSvg");
  const detail = document.getElementById("graphMiningDetail");
  const hint = document.getElementById("graphMiningHint");
  if (!svg || !detail || !hint) return;
  svg.innerHTML = "";
  if (!payload) {
    state.fraudRingGraphData = null;
    setText("fraudRingZoomLevel", `${Math.round(state.fraudRingZoom * 100)}%`);
    const summary = state.graphSummary || {};
    detail.innerHTML = `
      <div class="readable-empty">
        <strong>${summary.error ? "图挖掘结果不可用" : "暂无选中团伙"}</strong>
        <p>${escapeHtml(summary.error || "点击左侧 Top 风险团伙查看证据链。")}</p>
      </div>
    `;
    hint.textContent = "暂无团伙";
    return;
  }
  const group = payload.group || payload;
  const explanations = payload.explanations || {};
  const evidence = payload.evidence || [];
  let graphData;
  if (payload.subgraph) {
    graphData = layoutSubgraph(payload.subgraph);
  } else {
    const users = parseMaybeJson(group.sample_users, []).slice(0, 5);
    const resources = parseMaybeJson(group.sample_resources, []).slice(0, 6);
    graphData = layoutSubgraph({
      nodes: [
        { id: group.fraud_group_mining_id, label: group.fraud_group_mining_id, type: "fraud_group", role: "center" },
        { id: `${group.fraud_group_mining_id}:risk`, label: `risk ${fmt(Number(group.graph_mining_group_risk_score || 0), 2)}`, type: "risk", role: "risk" },
        ...users.map((label) => ({ id: String(label), label: String(label), type: "user", role: "member" })),
        ...resources.map((label) => ({ id: String(label), label: String(label), type: label.startsWith("IP_") ? "ip" : label.startsWith("D_") ? "device" : "resource", role: "shared_resource" })),
      ],
      edges: [
        ...users.map((label) => ({ source: String(label), target: group.fraud_group_mining_id, label: "member" })),
        ...resources.map((label) => ({ source: group.fraud_group_mining_id, target: String(label), label: "shared" })),
        { source: group.fraud_group_mining_id, target: `${group.fraud_group_mining_id}:risk`, label: "score" },
      ],
    });
  }
  state.fraudRingGraphData = graphData;
  drawFraudRingSvg(svg, graphData);
  hint.textContent = group.fraud_group_mining_id || "团伙";
  const codes = explanations.explanation_codes || parseMaybeJson(group.explanation_codes, []);
  const evidenceSummary = explanations.evidence_summary || parseMaybeJson(group.evidence_summary, {});
  renderGraphMiningDetail(detail, group, explanations, codes, evidenceSummary, evidence);
}

function drawFraudRingSvg(svg, graphData) {
  svg.innerHTML = "";
  setText("fraudRingZoomLevel", `${Math.round(state.fraudRingZoom * 100)}%`);
  const layer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  layer.setAttribute("class", "fraud-ring-layer");
  layer.setAttribute(
    "transform",
    `translate(${FRAUD_RING_VIEWBOX.centerX} ${FRAUD_RING_VIEWBOX.centerY}) scale(${state.fraudRingZoom}) translate(-${FRAUD_RING_VIEWBOX.centerX} -${FRAUD_RING_VIEWBOX.centerY})`,
  );
  svg.appendChild(layer);
  const nodes = graphData.nodes || [];
  const byId = Object.fromEntries(nodes.map((node) => [node.id, node]));
  for (const edge of graphData.edges || []) {
    const a = byId[edge.source];
    const b = byId[edge.target];
    if (!a || !b) continue;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y);
    line.setAttribute("class", "graph-edge");
    layer.appendChild(line);
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", (a.x + b.x) / 2);
    text.setAttribute("y", (a.y + b.y) / 2 - 4);
    text.setAttribute("class", "graph-edge-label");
    text.textContent = edge.label;
    layer.appendChild(text);
  }
  for (const node of nodes) {
    const groupNode = document.createElementNS("http://www.w3.org/2000/svg", "g");
    groupNode.setAttribute("class", `graph-node draggable-node graph-role-${node.role || node.type}`);
    groupNode.setAttribute("data-node-id", node.id);
    groupNode.setAttribute("tabindex", "0");
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", node.x);
    circle.setAttribute("cy", node.y);
    circle.setAttribute("r", node.role === "center" ? "42" : "30");
    const type = document.createElementNS("http://www.w3.org/2000/svg", "text");
    type.setAttribute("x", node.x);
    type.setAttribute("y", node.y - 4);
    type.setAttribute("class", "graph-node-type");
    type.textContent = nodeTypeLabel(node.type);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", node.x);
    label.setAttribute("y", node.y + 15);
    label.setAttribute("class", "graph-node-label");
    label.textContent = String(node.label || "").slice(0, 16);
    groupNode.appendChild(circle);
    groupNode.appendChild(type);
    groupNode.appendChild(label);
    groupNode.addEventListener("pointerdown", (event) => startFraudRingDrag(event, node, svg));
    layer.appendChild(groupNode);
  }
}

function startFraudRingDrag(event, node, svg) {
  event.preventDefault();
  const target = event.currentTarget;
  target.classList.add("dragging");
  const point = svg.createSVGPoint();
  const toSvgPoint = (pointerEvent) => {
    point.x = pointerEvent.clientX;
    point.y = pointerEvent.clientY;
    return point.matrixTransform(svg.getScreenCTM().inverse());
  };
  const start = toSvgPoint(event);
  const origin = { x: node.x, y: node.y };
  const move = (pointerEvent) => {
    const current = toSvgPoint(pointerEvent);
    node.x = clamp(origin.x + (current.x - start.x) / state.fraudRingZoom, FRAUD_RING_VIEWBOX.minX, FRAUD_RING_VIEWBOX.maxX);
    node.y = clamp(origin.y + (current.y - start.y) / state.fraudRingZoom, FRAUD_RING_VIEWBOX.minY, FRAUD_RING_VIEWBOX.maxY);
    drawFraudRingSvg(svg, state.fraudRingGraphData);
  };
  const stop = () => {
    target.classList.remove("dragging");
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", stop);
    window.removeEventListener("pointercancel", stop);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", stop);
  window.addEventListener("pointercancel", stop);
}

function setFraudRingZoom(nextZoom) {
  state.fraudRingZoom = clamp(nextZoom, 0.6, 2.4);
  setText("fraudRingZoomLevel", `${Math.round(state.fraudRingZoom * 100)}%`);
  const svg = document.getElementById("fraudRingSvg");
  if (svg && state.fraudRingGraphData) {
    drawFraudRingSvg(svg, state.fraudRingGraphData);
  }
}

function zoomFraudRing(delta) {
  setFraudRingZoom(state.fraudRingZoom + delta);
}

function handleFraudRingWheel(event) {
  if (!state.fraudRingGraphData) return;
  event.preventDefault();
  const step = event.deltaY > 0 ? -0.1 : 0.1;
  zoomFraudRing(step);
}

function readableEvidenceKind(kind) {
  const labels = {
    shared_device: "共享设备",
    shared_ip: "共享 IP",
    shared_merchant: "共享商户",
    shared_payee: "共享收款方",
  };
  return labels[kind] || kind || "--";
}

function readableCode(code) {
  const labels = {
    high_confidence_ring: "高置信团伙",
    shared_device_ring: "共享设备环",
    shared_ip_cluster: "共享 IP 聚集",
    merchant_cashout_cluster: "商户套现聚集",
    shared_payee_chain: "共享收款链路",
    fraud_seed_propagation: "欺诈种子传播",
    multi_scenario_campaign: "多场景团伙",
    fund_transfer_chain: "资金转移链",
    weak_graph_signal: "弱图信号",
  };
  return labels[code] || code;
}

function renderMetricMini(label, value) {
  return `<div class="mini-metric"><span>${escapeHtml(label)}</span><strong class="scroll-value" title="${escapeHtml(value)}">${escapeHtml(value)}</strong></div>`;
}

function renderGraphMiningDetail(container, group, explanations, codes, evidenceSummary, evidence) {
  const topScenarios = parseMaybeJson(group.top_scenarios, {});
  const byKind = evidenceSummary.by_kind || {};
  const topEvidence = evidence.length ? evidence.slice(0, 8) : (evidenceSummary.top_resources || []);
  const codeHtml = (codes || []).map((code) => `<span class="reason-chip">${escapeHtml(readableCode(code))}</span>`).join("");
  const kindHtml = Object.entries(byKind).length
    ? Object.entries(byKind).map(([kind, count]) => `<span class="evidence-chip">${escapeHtml(readableEvidenceKind(kind))}: ${escapeHtml(count)}</span>`).join("")
    : '<span class="evidence-chip">暂无分类证据</span>';
  const scenarioHtml = Object.entries(topScenarios).length
    ? Object.entries(topScenarios).map(([name, count]) => `<span>${escapeHtml(name)} <strong>${escapeHtml(count)}</strong></span>`).join("")
    : "<span>无注入场景标记</span>";
  const evidenceRows = topEvidence.length
    ? topEvidence.map((item) => `
        <tr>
          <td>${escapeHtml(item.entity_id)}</td>
          <td>${escapeHtml(readableEvidenceKind(item.evidence_kind || item.entity_type))}</td>
          <td>${fmt(Number(item.shared_user_count || 0), 0)}</td>
          <td>${fmt(Number(item.fraud_seed_count || 0), 0)}</td>
          <td>${fmt(Number(item.resource_risk_score || 0))}</td>
        </tr>
      `).join("")
    : '<tr><td colspan="5">暂无资源证据</td></tr>';

  container.innerHTML = `
    <div class="detail-hero">
      <div>
        <span class="detail-eyebrow">${escapeHtml(group.fraud_group_mining_id || "--")}</span>
        <h4>${escapeHtml(explanations.risk_level || group.risk_level || "watch")} · risk ${fmt(Number(group.graph_mining_group_risk_score || 0))}</h4>
      </div>
      <span class="run-status ${statusClass((explanations.risk_level || group.risk_level) === "high" ? "failed" : "running")}">${escapeHtml(explanations.risk_level || group.risk_level || "watch")}</span>
    </div>
    <p class="detail-text">${escapeHtml(explanations.explanation_text || group.explanation_text || "暂无解释摘要")}</p>
    <div class="mini-grid">
      ${renderMetricMini("用户数", fmt(Number(group.user_count || 0), 0))}
      ${renderMetricMini("交易数", fmt(Number(group.txn_count || 0), 0))}
      ${renderMetricMini("欺诈种子", fmt(Number(group.fraud_seed_count || 0), 0))}
      ${renderMetricMini("证据数", fmt(Number(group.evidence_count || topEvidence.length || 0), 0))}
    </div>
    <div class="detail-section">
      <h5>理由码</h5>
      <div class="chip-row">${codeHtml || '<span class="reason-chip">暂无理由码</span>'}</div>
    </div>
    <div class="detail-section">
      <h5>证据类型</h5>
      <div class="chip-row">${kindHtml}</div>
    </div>
    <div class="detail-section">
      <h5>场景分布</h5>
      <div class="scenario-row">${scenarioHtml}</div>
    </div>
    <div class="detail-section">
      <h5>Top 资源证据</h5>
      <div class="evidence-table-wrap">
        <table class="evidence-table">
          <thead><tr><th>实体</th><th>证据</th><th>用户</th><th>欺诈种子</th><th>资源风险</th></tr></thead>
          <tbody>${evidenceRows}</tbody>
        </table>
      </div>
    </div>
  `;
}

function switchView(name) {
  const titles = {
    overview: ["总览", "模型切换、系统能力和核心操作入口"],
    stream: ["实时", "查看 Kafka 风险结果并提交人工审核反馈"],
    graphMining: ["图挖掘", "从共享设备、IP、商户和收款链路中识别疑似团伙"],
    models: ["模型", "发现、加载和重载可插拔模型产物"],
    metrics: ["指标", "当前模型指标、阈值校准和 leaderboard"],
    predict: ["试算", "表单化输入单笔交易并计算风险"],
  };
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === name));
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === name));
  document.getElementById("viewTitle").textContent = (titles[name] || titles.overview)[0];
  document.getElementById("viewSubtitle").textContent = (titles[name] || titles.overview)[1];
}

function renderDemoStatus(run, outputId = "overviewTaskOutput") {
  const output = document.getElementById(outputId);
  if (!output) return;
  const isRunning = run.status === "running";
  const elapsed = run.elapsed_seconds === null || run.elapsed_seconds === undefined ? "--" : `${Math.round(run.elapsed_seconds)}s`;
  output.innerHTML = `
    <div class="demo-status-card">
      <div class="demo-status-head">
        <strong>${escapeHtml(run.label || run.action || "后台任务")}</strong>
        <span class="run-status ${statusClass(run.status)}">${statusLabel(run.status)}</span>
      </div>
      <div class="demo-status-grid">
        <span>任务 ID</span><strong>${escapeHtml(run.run_id || "--")}</strong>
        <span>耗时</span><strong>${escapeHtml(elapsed)}</strong>
        <span>退出码</span><strong>${run.exit_code === null || run.exit_code === undefined ? "--" : escapeHtml(run.exit_code)}</strong>
        <span>日志</span><strong>${escapeHtml(run.log_path || "--")}</strong>
      </div>
      ${isRunning ? '<div class="progress-line"><span></span></div>' : ""}
      <div class="command-line">${escapeHtml(run.command || "")}</div>
      <pre class="log-tail">${escapeHtml(run.log_tail || "等待日志输出")}</pre>
    </div>
  `;
}

async function pollDemoRun(runId, action, outputId = "overviewTaskOutput") {
  if (state.demoPollTimer) {
    window.clearTimeout(state.demoPollTimer);
    state.demoPollTimer = null;
  }
  const run = await api(`/demo/runs/${encodeURIComponent(runId)}`);
  renderDemoStatus(run, outputId);
  if (run.status === "running") {
    state.demoPollTimer = window.setTimeout(() => pollDemoRun(runId, action, outputId), 1500);
    return;
  }
  if (run.status === "succeeded" && action === "graph_mining") {
    await refreshGraphMining(true);
    switchView("graphMining");
  }
  if (run.status === "succeeded" && action === "replay_stream") {
    document.getElementById("topicSelect").value = "risk_results";
    await refreshStream();
  }
  if (run.status === "succeeded" && (action === "retrain_logistic" || action === "adaptive_retrain")) {
    await refreshAll();
    switchView("models");
  }
}

async function runDemoAction(action, outputId = "overviewTaskOutput") {
  const output = document.getElementById(outputId);
  if (!output) return;
  output.innerHTML = `<div class="demo-status-card"><strong>启动 ${escapeHtml(action)} 中...</strong><div class="progress-line"><span></span></div></div>`;
  try {
    const result = await api("/demo/run", {
      method: "POST",
      body: JSON.stringify({ action }),
    });
    renderDemoStatus(result, outputId);
    if (result.run_id) {
      await pollDemoRun(result.run_id, action, outputId);
    }
  } catch (error) {
    output.innerHTML = `<div class="demo-status-card"><span class="run-status status-failed">失败</span><p>${escapeHtml(error.message)}</p></div>`;
  }
}

function formValue(name) {
  const element = document.querySelector(`#predictForm [name="${name}"]`);
  return element ? element.value : "";
}

function numberValue(name, fallback = 0) {
  const value = formValue(name);
  return value === "" ? fallback : asNumber(value, fallback);
}

function setFormValue(name, value) {
  const element = document.querySelector(`#predictForm [name="${name}"]`);
  if (element) element.value = value ?? "";
}

function normalizeCountryInputs() {
  document.querySelectorAll('#predictForm input[name$="_country"]').forEach((input) => {
    input.addEventListener("input", () => {
      input.value = input.value.replace(/[^a-zA-Z]/g, "").slice(0, 2).toUpperCase();
    });
  });
}

function setSampleForm(record = sampleRecord) {
  setFormValue("transaction_id", record.transaction_id);
  setFormValue("timestamp", timestampForInput(record.timestamp));
  setFormValue("amount", record.amount);
  setFormValue("amount_bucket", record.amount_bucket || "large");
  setFormValue("txn_type", record.txn_type || "transfer");
  setFormValue("channel", record.channel || "app");
  setFormValue("payment_method", record.payment_method || "bank_transfer");
  setFormValue("merchant_category", record.merchant_category || "digital_goods");
  setFormValue("payer_id", record.payer_id);
  setFormValue("payee_id", record.payee_id);
  setFormValue("merchant_id", record.merchant_id);
  setFormValue("device_id", record.device_id);
  setFormValue("ip_id", record.ip_id);
  setFormValue("payer_country", record.payer_country || "CN");
  setFormValue("payee_country", record.payee_country || "US");
  setFormValue("is_night", String(record.is_night || 0));
  const wf = record.window_features || {};
  const mp = record.merchant_profile || {};
  setFormValue("user_txn_count_5min", wf.user_txn_count_5min);
  setFormValue("user_amount_sum_5min", wf.user_amount_sum_5min);
  setFormValue("user_txn_count_1h", wf.user_txn_count_1h);
  setFormValue("user_unique_payee_count_1h", wf.user_unique_payee_count_1h);
  setFormValue("device_unique_user_count_10min", wf.device_unique_user_count_10min);
  setFormValue("ip_unique_user_count_10min", wf.ip_unique_user_count_10min);
  setFormValue("merchant_txn_count_1h", wf.merchant_txn_count_1h);
  setFormValue("merchant_complaint_rate", mp.complaint_rate);
}

function recordFromForm() {
  const timestamp = timestampFromInput(formValue("timestamp"));
  const date = new Date(timestamp);
  const merchantCategory = formValue("merchant_category") || "digital_goods";
  const payerCountry = normalizeCountry(formValue("payer_country") || "CN", "付款国家");
  const payeeCountry = normalizeCountry(formValue("payee_country") || "US", "收款国家");
  return {
    transaction_id: formValue("transaction_id") || `dashboard_${Date.now()}`,
    timestamp,
    source: "dashboard_form",
    amount: numberValue("amount", 0),
    amount_bucket: formValue("amount_bucket") || "medium",
    currency: "USD",
    txn_type: formValue("txn_type") || "transfer",
    channel: formValue("channel") || "app",
    payment_method: formValue("payment_method") || "bank_transfer",
    merchant_category: merchantCategory,
    payer_id: formValue("payer_id") || "demo_user",
    payee_id: formValue("payee_id") || "demo_payee",
    merchant_id: formValue("merchant_id") || "demo_merchant",
    device_id: formValue("device_id") || "demo_device",
    ip_id: formValue("ip_id") || "demo_ip",
    payer_country: payerCountry,
    payee_country: payeeCountry,
    hour: date.getHours(),
    day_of_week: date.getDay(),
    is_weekend: [0, 6].includes(date.getDay()) ? 1 : 0,
    is_night: numberValue("is_night", 0),
    window_features: {
      user_txn_count_5min: numberValue("user_txn_count_5min", 0),
      user_amount_sum_5min: numberValue("user_amount_sum_5min", 0),
      user_txn_count_1h: numberValue("user_txn_count_1h", 0),
      user_amount_sum_1h: numberValue("user_amount_sum_5min", 0),
      user_unique_payee_count_1h: numberValue("user_unique_payee_count_1h", 0),
      device_unique_user_count_10min: numberValue("device_unique_user_count_10min", 0),
      ip_unique_user_count_10min: numberValue("ip_unique_user_count_10min", 0),
      merchant_txn_count_1h: numberValue("merchant_txn_count_1h", 0),
      merchant_amount_sum_1h: numberValue("amount", 0) * Math.max(1, numberValue("merchant_txn_count_1h", 1)),
      merchant_unique_user_count_1h: Math.max(1, numberValue("merchant_txn_count_1h", 1)),
    },
    user_profile: {
      txn_count: numberValue("user_txn_count_1h", 0),
      total_amount: numberValue("user_amount_sum_5min", 0),
      avg_amount: numberValue("amount", 0),
      common_device_count: 1,
      common_ip_count: 1,
      home_country: payerCountry,
      account_age_days: 365,
    },
    device_profile: {
      bind_user_count: numberValue("device_unique_user_count_10min", 1),
      txn_count: numberValue("user_txn_count_1h", 1),
      device_type: "mobile",
      os: "Android",
      is_emulator: numberValue("device_unique_user_count_10min", 0) >= 5 ? 1 : 0,
      is_proxy_device: 0,
    },
    ip_profile: {
      bind_user_count: numberValue("ip_unique_user_count_10min", 1),
      country: payeeCountry,
      is_proxy: numberValue("ip_unique_user_count_10min", 0) >= 6 ? 1 : 0,
      is_vpn: 0,
    },
    merchant_profile: {
      merchant_category: merchantCategory,
      merchant_country: payeeCountry,
      txn_count: numberValue("merchant_txn_count_1h", 1),
      unique_user_count: Math.max(1, numberValue("merchant_txn_count_1h", 1)),
      total_amount: numberValue("amount", 0),
      avg_amount: numberValue("amount", 0),
      chargeback_rate: 0.005,
      complaint_rate: numberValue("merchant_complaint_rate", 0),
    },
    graph_features: {
      payer_graph_degree: numberValue("user_unique_payee_count_1h", 0) + numberValue("device_unique_user_count_10min", 0),
      payer_graph_fraud_edge_count: 0,
      payer_graph_fraud_edge_ratio: 0,
      payer_graph_mining_group_risk_score: 0,
      device_graph_degree: numberValue("device_unique_user_count_10min", 0),
      ip_graph_degree: numberValue("ip_unique_user_count_10min", 0),
      merchant_graph_degree: numberValue("merchant_txn_count_1h", 0),
    },
  };
}

function renderPredictResult(result) {
  const payload = (result.results && result.results[0]) || result;
  const reasons = payload.reason_codes || [];
  return `
    <div class="detail-hero">
      <div>
        <span class="detail-eyebrow scroll-value">交易ID: ${escapeHtml(payload.transaction_id || "dashboard")}</span>
        <h4>${escapeHtml(payload.risk_level || "--")} · score ${fmt(payload.risk_score)}</h4>
      </div>
      <span class="run-status ${payload.decision === "reject" ? "status-failed" : payload.decision === "review" ? "status-running" : "status-success"}">${escapeHtml(payload.decision || "--")}</span>
    </div>
    <div class="mini-grid">
      ${renderMetricMini("模型", payload.model_name || "--")}
      ${renderMetricMini("版本", payload.model_version || "--")}
      ${renderMetricMini("风险等级", payload.risk_level || "--")}
      ${renderMetricMini("决策", payload.decision || "--")}
    </div>
    <div class="detail-section">
      <h5>理由码</h5>
      <div class="reason-list">${renderReasonList(reasons)}</div>
    </div>
  `;
}

async function predict() {
  const output = document.getElementById("predictResult");
  try {
    const record = recordFromForm();
    const result = await api("/predict", {
      method: "POST",
      body: JSON.stringify({ record }),
    });
    output.innerHTML = renderPredictResult(result);
  } catch (error) {
    output.textContent = error.message;
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
});

document.querySelectorAll(".sort-btn").forEach((button) => {
  button.addEventListener("click", () => toggleModelSort(button.dataset.sortKey));
});

bind("refreshBtn", "click", refreshAll);
bind("apiKeySaveBtn", "click", saveApiKey);
bind("streamRefreshBtn", "click", refreshStream);
bind("streamReplayBtn", "click", () => runDemoAction("replay_stream", "streamTaskOutput"));
bind("feedbackBtn", "click", submitFeedback);
bind("overviewProfilesBtn", "click", () => runDemoAction("load_profiles", "overviewTaskOutput"));
bind("overviewSimulateBtn", "click", () => runDemoAction("simulate_stream", "overviewTaskOutput"));
bind("overviewRetrainBtn", "click", () => runDemoAction("retrain_logistic", "overviewTaskOutput"));
bind("overviewAdaptiveRetrainBtn", "click", () => runDemoAction("adaptive_retrain", "overviewTaskOutput"));
bind("overviewGraphMiningBtn", "click", () => runDemoAction("graph_mining", "overviewTaskOutput"));
bind("graphRefreshBtn", "click", () => refreshGraphMining(true));
bind("fraudRingZoomOutBtn", "click", () => zoomFraudRing(-0.15));
bind("fraudRingZoomInBtn", "click", () => zoomFraudRing(0.15));
bind("fraudRingZoomResetBtn", "click", () => setFraudRingZoom(1));
bind("fraudRingSvg", "wheel", handleFraudRingWheel);
bind("overviewActivateModelBtn", "click", async () => {
  const selector = document.getElementById("overviewModelSelect");
  const modelName = selector && selector.value;
  if (!modelName) return;
  await api("/models/activate", {
    method: "POST",
    body: JSON.stringify({ model_name: modelName }),
  });
  await refreshAll();
});
bind("reloadBtn", "click", async () => {
  await api("/reload", { method: "POST", body: JSON.stringify({}) });
  await refreshAll();
});
bind("sampleBtn", "click", () => setSampleForm(sampleRecord));
bind("predictBtn", "click", predict);
bind("thresholdEvaluateBtn", "click", evaluateThresholds);
bind("mediumThreshold", "input", updateThresholdLabels);
bind("highThreshold", "input", updateThresholdLabels);

normalizeCountryInputs();
loadApiKey();
setSampleForm(sampleRecord);
updateThresholdLabels();
refreshAll();
