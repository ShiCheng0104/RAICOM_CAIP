const state = {
  health: null,
  models: [],
  adapters: [],
  metrics: null,
  leaderboard: [],
  selectedStreamEvent: null,
};

const sampleRecord = {
  transaction_id: "dashboard_demo_txn",
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
    payee_graph_degree: 7,
    merchant_graph_degree: 44,
    device_graph_degree: 17,
    ip_graph_degree: 29,
  },
};

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  if (typeof value === "number") return value.toFixed(digits);
  return String(value);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

async function refreshAll() {
  const apiDot = document.getElementById("apiDot");
  const apiState = document.getElementById("apiState");
  try {
    const [models, metrics, leaderboard] = await Promise.all([
      api("/models"),
      api("/metrics"),
      api("/leaderboard"),
    ]);
    state.health = models.active;
    state.models = models.models || [];
    state.adapters = models.adapters || [];
    state.metrics = metrics;
    state.leaderboard = leaderboard.rows || [];
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
          <strong>${txId}</strong>
          <span class="${riskClass(level)}">${level}</span>
        </div>
        <div class="stream-meta">score ${score} · ${value.decision || "--"} · p${message.partition}/o${message.offset}</div>
        <div class="stream-meta">${(value.reason_codes || []).join(", ") || value.feedback_created_at || ""}</div>
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
    result.textContent = JSON.stringify(response, null, 2);
  } catch (error) {
    result.textContent = error.message;
  }
}

function renderOverview() {
  const health = state.health || {};
  const metrics = (state.metrics && state.metrics.metrics) || {};
  document.getElementById("activeModel").textContent = health.model_name || "--";
  document.getElementById("activeVersion").textContent = health.model_version || "--";
  document.getElementById("featureCount").textContent = fmt(health.feature_count, 0);
  document.getElementById("prAuc").textContent = fmt(metrics.pr_auc);
  document.getElementById("f1Score").textContent = fmt(metrics.f1);
  document.getElementById("loadedState").textContent = health.loaded ? "loaded" : "not loaded";
  document.getElementById("modelPath").textContent = health.model_path || "--";
  document.getElementById("loadedAt").textContent = health.loaded_at || "--";
  document.getElementById("adapterList").textContent = state.adapters.join(", ") || "--";
}

function renderModels() {
  const body = document.getElementById("modelRows");
  document.getElementById("modelCount").textContent = `${state.models.length} models`;
  body.innerHTML = "";
  for (const model of state.models) {
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

function renderMetrics() {
  document.getElementById("metricsJson").textContent = JSON.stringify(state.metrics || {}, null, 2);
  const board = document.getElementById("leaderboard");
  board.innerHTML = "";
  for (const item of state.leaderboard) {
    const row = document.createElement("div");
    row.className = "leader-row";
    row.innerHTML = `
      <strong>${item.model_name || "--"}<br><span>${item.dataset || ""}</span></strong>
      <span>PR ${fmt(item.pr_auc)}</span>
      <span>ROC ${fmt(item.roc_auc)}</span>
      <span>F1 ${fmt(item.f1)}</span>
    `;
    board.appendChild(row);
  }
  if (!state.leaderboard.length) {
    board.textContent = "暂无 leaderboard";
  }
}

function switchView(name) {
  const titles = {
    overview: ["总览", "实时风控模型状态、数据指标与演示入口"],
    stream: ["实时", "查看 Kafka 风险结果并提交人工审核反馈"],
    models: ["模型", "发现、加载和重载可插拔模型产物"],
    metrics: ["指标", "当前模型指标、评估摘要和 leaderboard"],
    predict: ["试算", "单笔交易风险评分和理由码"],
  };
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === name));
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === name));
  document.getElementById("viewTitle").textContent = titles[name][0];
  document.getElementById("viewSubtitle").textContent = titles[name][1];
}

async function predict() {
  const output = document.getElementById("predictResult");
  try {
    const record = JSON.parse(document.getElementById("predictInput").value);
    const result = await api("/predict", {
      method: "POST",
      body: JSON.stringify({ record }),
    });
    output.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    output.textContent = error.message;
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
});

document.getElementById("refreshBtn").addEventListener("click", refreshAll);
document.getElementById("streamRefreshBtn").addEventListener("click", refreshStream);
document.getElementById("feedbackBtn").addEventListener("click", submitFeedback);
document.getElementById("reloadBtn").addEventListener("click", async () => {
  await api("/reload", { method: "POST", body: JSON.stringify({}) });
  await refreshAll();
});
document.getElementById("sampleBtn").addEventListener("click", () => {
  document.getElementById("predictInput").value = JSON.stringify(sampleRecord, null, 2);
});
document.getElementById("predictBtn").addEventListener("click", predict);

document.getElementById("predictInput").value = JSON.stringify(sampleRecord, null, 2);
refreshAll();
