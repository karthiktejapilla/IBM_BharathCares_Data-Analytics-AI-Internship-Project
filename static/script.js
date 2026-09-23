/* =====================================================
   Olist BI Dashboard — script.js
   ===================================================== */

"use strict";

// ---- STATE ----
const State = {
  activePage: "executive",
  filters: { states: [], categories: [], segments: [] },
  data: {},
};

// ---- COLOUR PALETTES ----
const PALETTE = [
  "#2563eb","#16a34a","#d97706","#7c3aed","#0d9488",
  "#dc2626","#0284c7","#9333ea","#ca8a04","#059669",
  "#be185d","#6366f1","#84cc16","#f97316","#06b6d4",
  "#a855f7","#10b981","#f59e0b","#ef4444","#3b82f6",
];

const SEG_COLORS = {
  "Champions":          "#2563eb",
  "Loyal Customers":    "#16a34a",
  "At Risk":            "#dc2626",
  "New Customers":      "#0d9488",
  "Potential Loyalists":"#d97706",
  "Needs Attention":    "#7c3aed",
  "Lost / Inactive":    "#6b7280",
};

const DELAY_COLORS = {
  "Early >7d":   "#16a34a",
  "Early 3-7d":  "#4ade80",
  "Early 0-3d":  "#86efac",
  "Late 1-3d":   "#fca5a5",
  "Late 4-7d":   "#f87171",
  "Late 8-14d":  "#ef4444",
  "Late >14d":   "#b91c1c",
};

// ---- PLOTLY LAYOUT DEFAULTS ----
function baseLayout(title, xTitle, yTitle) {
  return {
    title:  { text: title || "", font: { size: 13, color: "#1a1d23" } },
    xaxis:  { title: xTitle || "", tickfont: { size: 11 }, gridcolor: "#f0f2f5" },
    yaxis:  { title: yTitle || "", tickfont: { size: 11 }, gridcolor: "#f0f2f5" },
    margin: { t: title ? 30 : 10, r: 16, b: 55, l: 60 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    font: { family: "-apple-system,'Segoe UI',sans-serif", size: 12 },
    legend: { font: { size: 11 }, bgcolor: "rgba(0,0,0,0)" },
    hovermode: "closest",
    autosize: true,
  };
}

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

function fmt_currency(v) {
  if (v == null) return "–";
  if (v >= 1e6) return "R$ " + (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return "R$ " + (v / 1e3).toFixed(1) + "K";
  return "R$ " + v.toFixed(2);
}

function fmt_pct(v, dec=1)  { return v == null ? "–" : v.toFixed(dec) + "%"; }
function fmt_num(v)          { return v == null ? "–" : v.toLocaleString(); }
function fmt_score(v)        { return v == null ? "–" : v.toFixed(2); }

// ---- DISPLAY LABEL FORMATTERS ----
// Converts snake_case category names to Title Case for display.
// The raw value is preserved in tooltips and data; this is purely cosmetic.
const CATEGORY_LABEL_MAP = {
  "health_beauty":             "Health & Beauty",
  "bed_bath_table":            "Bed & Bath Table",
  "sports_leisure":            "Sports & Leisure",
  "computers_accessories":     "Computers & Accessories",
  "furniture_decor":           "Furniture & Decor",
  "housewares":                "Housewares",
  "watches_gifts":             "Watches & Gifts",
  "telephony":                 "Telephony",
  "garden_tools":              "Garden Tools",
  "baby":                      "Baby",
  "toys":                      "Toys",
  "cool_stuff":                "Cool Stuff",
  "perfumery":                 "Perfumery",
  "auto":                      "Auto",
  "electronics":               "Electronics",
  "stationery":                "Stationery",
  "fashion_bags_accessories":  "Fashion Bags & Accessories",
  "office_furniture":          "Office Furniture",
  "luggage_accessories":       "Luggage & Accessories",
  "pet_shop":                  "Pet Shop",
  "food_drink":                "Food & Drink",
  "construction_tools_safety": "Construction Tools Safety",
  "fixed_telephony":           "Fixed Telephony",
  "music":                     "Music",
  "books_general_interest":    "Books General Interest",
  "books_technical":           "Books Technical",
  "small_appliances":          "Small Appliances",
  "kitchen_dining_laundry_garden_furniture": "Kitchen & Dining",
  "market_place":              "Market Place",
  "signaling_and_security":    "Signaling & Security",
  "industry_commerce_and_business": "Industry & Business",
  "art":                       "Art",
  "cds_dvds_musicals":         "CDs & DVDs",
  "fashion_shoes":             "Fashion Shoes",
  "fashion_underwear_beach":   "Fashion Underwear",
  "agro_industry_and_commerce":"Agro Industry",
  "musical_instruments":       "Musical Instruments",
  "home_comfort":              "Home Comfort",
  "home_construction":         "Home Construction",
  "fashion_male_clothing":     "Fashion Male Clothing",
  "la_cuisine":                "La Cuisine",
  "tablets_printing_image":    "Tablets & Printing",
  "christmas_supplies":        "Christmas Supplies",
  "air_conditioning":          "Air Conditioning",
  "audio":                     "Audio",
  "drinks":                    "Drinks",
  "fashio_female_clothing":    "Fashion Female Clothing",
  "flowers":                   "Flowers",
  "food":                      "Food",
  "home_comfort_2":            "Home Comfort 2",
  "diapers_and_hygiene":       "Diapers & Hygiene",
  "party_supplies":            "Party Supplies",
  "furniture_bedroom":         "Furniture Bedroom",
  "furniture_living_room":     "Furniture Living Room",
  "furniture_mattress_and_upholstery": "Furniture Mattress",
  "pc_gamer":                  "PC Gamer",
  "construction_tools_construction": "Construction Tools",
  "home_appliances":           "Home Appliances",
  "home_appliances_2":         "Home Appliances 2",
  "computers":                 "Computers",
  "dvds_blu_ray":              "DVDs & Blu-Ray",
  "security_and_services":     "Security & Services",
  "fashion_sport":             "Fashion Sport",
  "arts_and_craftmanship":     "Arts & Craftmanship",
  "costumes_accessories":      "Costumes & Accessories",
  "insurance_and_services":    "Insurance & Services",
  "consoles_games":            "Consoles & Games",
};

function fmt_category(raw) {
  if (!raw) return raw;
  if (CATEGORY_LABEL_MAP[raw]) return CATEGORY_LABEL_MAP[raw];
  // Fallback: replace underscores with spaces, title-case each word
  return raw.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

// Seller IDs: keep full ID in tooltip, show truncated display label
function fmt_seller_id(id) {
  if (!id) return id;
  return "..." + String(id).slice(-8);
}

// ---- FETCH HELPER ----
async function apiFetch(endpoint, params = {}) {
  const qs = new URLSearchParams(Object.entries(params).filter(([,v]) => v)).toString();
  const url = qs ? `${endpoint}?${qs}` : endpoint;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`API error ${r.status} for ${url}`);
  return r.json();
}

// ---- LOADING OVERLAY ----
const overlay = document.getElementById("loadingOverlay");
function showLoading()  { overlay.classList.add("active"); }
function hideLoading()  { overlay.classList.remove("active"); }

// ---- NAVIGATION ----
function navigateTo(page) {
  State.activePage = page;
  document.querySelectorAll(".page").forEach(el => el.classList.add("hidden"));
  document.getElementById("page-" + page).classList.remove("hidden");
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.page === page);
  });
  const titles = {
    executive: "Executive Overview",
    sales:     "Sales & Product Analysis",
    customers: "Customer Analytics",
    delivery:  "Delivery & Risk Analysis",
  };
  document.getElementById("pageTitle").textContent = titles[page] || page;
  loadPage(page);
}

// ---- INIT ----
async function init() {
  showLoading();
  try {
    // Load filter options
    const filtersData = await apiFetch("/api/filters");
    populateFilters(filtersData);
    State.data.filters = filtersData;

    // Wire nav
    document.querySelectorAll(".nav-item").forEach(el => {
      el.addEventListener("click", () => navigateTo(el.dataset.page));
    });

    // Wire filter buttons
    document.getElementById("btnApplyFilters").addEventListener("click", applyFilters);
    document.getElementById("btnResetFilters").addEventListener("click", resetFilters);

    // Wire insight tabs
    document.querySelectorAll(".insight-tab").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".insight-tab").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        filterInsightCards(btn.dataset.filter);
      });
    });

    // Initial page — use server-sent active_page if available
    const initialPage = document.body.dataset.activePage || "executive";
    navigateTo(initialPage);
  } finally {
    hideLoading();
  }
}

function populateFilters(filtersData) {
  const stateSelect = document.getElementById("filterState");
  const catSelect   = document.getElementById("filterCategory");

  filtersData.states.forEach(s => {
    const o = document.createElement("option");
    o.value = s; o.textContent = s;
    stateSelect.appendChild(o);
  });

  filtersData.categories.forEach(c => {
    const o = document.createElement("option");
    o.value = c; o.textContent = c;
    catSelect.appendChild(o);
  });
}

function applyFilters() {
  const stateSelect = document.getElementById("filterState");
  const catSelect   = document.getElementById("filterCategory");
  State.filters.states     = Array.from(stateSelect.selectedOptions).map(o => o.value).filter(v => v);
  State.filters.categories = catSelect.value ? [catSelect.value] : [];
  loadPage(State.activePage);
}

function resetFilters() {
  document.getElementById("filterState").selectedIndex = -1;
  document.getElementById("filterCategory").selectedIndex = 0;
  State.filters = { states: [], categories: [], segments: [] };
  loadPage(State.activePage);
}

// ---- PAGE LOADER DISPATCHER ----
async function loadPage(page) {
  showLoading();
  try {
    if (page === "executive") await loadExecutive();
    else if (page === "sales")     await loadSales();
    else if (page === "customers") await loadCustomers();
    else if (page === "delivery")  await loadDelivery();
  } catch(e) {
    console.error("Page load error:", e);
  } finally {
    hideLoading();
  }
}

// ===================================================================
// PAGE: EXECUTIVE
// ===================================================================
async function loadExecutive() {
  const [kpiData, monthlyData, catData, stateData, insightData] = await Promise.all([
    apiFetch("/api/kpis"),
    apiFetch("/api/monthly-trends"),
    apiFetch("/api/categories", { top: 15 }),
    apiFetch("/api/states"),
    apiFetch("/api/insights"),
  ]);

  // KPI Cards
  const k = kpiData;
  setText("v-product-sales", fmt_currency(k.total_product_sales));
  setText("v-order-value",   fmt_currency(k.total_order_value));
  setText("v-orders",        fmt_num(k.total_orders));
  setText("v-delivered",     fmt_num(k.total_delivered_orders));
  setText("v-customers",     fmt_num(k.total_customers));
  setText("v-aov",           fmt_currency(k.avg_order_value));
  setText("v-review",        fmt_score(k.avg_review_score));
  setText("v-ontime",        fmt_pct(k.on_time_delivery_pct));
  document.querySelectorAll(".kpi-card").forEach(el => el.classList.remove("loading"));

  // Monthly Sales
  const months = monthlyData.map(r => r.ym);
  const sales  = monthlyData.map(r => r.product_sales_value);

  Plotly.newPlot("chartMonthlySales", [{
    type: "scatter", mode: "lines+markers",
    x: months, y: sales,
    line: { color: "#2563eb", width: 2.5 },
    marker: { size: 5 },
    fill: "tozeroy", fillcolor: "rgba(37,99,235,.08)",
    hovertemplate: "<b>%{x}</b><br>Sales: R$ %{y:,.0f}<extra></extra>",
  }], { ...baseLayout("", "Month", "R$"), ...{ margin: { t:10, r:16, b:55, l:70 } } }, PLOTLY_CONFIG);

  // Monthly Orders
  const orders = monthlyData.map(r => r.order_count);
  Plotly.newPlot("chartMonthlyOrders", [{
    type: "bar",
    x: months, y: orders,
    marker: { color: "#0d9488" },
    hovertemplate: "<b>%{x}</b><br>Orders: %{y:,}<extra></extra>",
  }], { ...baseLayout("", "Month", "Orders"), ...{ margin: { t:10, r:16, b:55, l:70 } } }, PLOTLY_CONFIG);

  // Category Sales (horizontal bar, top 15)
  const catPerf = catData.performance.slice(0,15);
  const catNames = catPerf.map(r => fmt_category(r.product_category_name_english)).reverse();
  const catSales = catPerf.map(r => r.product_sales_value).reverse();

  Plotly.newPlot("chartCategorySales", [{
    type: "bar", orientation: "h",
    x: catSales, y: catNames,
    marker: { color: PALETTE.slice(0,15).reverse() },
    hovertemplate: "<b>%{y}</b><br>Sales: R$ %{x:,.0f}<extra></extra>",
  }], { ...baseLayout("","R$",""), margin: { t:10, r:20, b:45, l:180 } }, PLOTLY_CONFIG);

  // State Sales
  const stPerf = stateData.performance.slice(0,15);
  const stNames = stPerf.map(r => r.customer_state).reverse();
  const stSales = stPerf.map(r => r.product_sales_value).reverse();

  Plotly.newPlot("chartStateSales", [{
    type: "bar", orientation: "h",
    x: stSales, y: stNames,
    marker: { color: "#7c3aed" },
    hovertemplate: "<b>%{y}</b><br>Sales: R$ %{x:,.0f}<extra></extra>",
  }], { ...baseLayout("","R$","State"), margin: { t:10, r:20, b:45, l:55 } }, PLOTLY_CONFIG);

  // Insights
  renderInsightGrid("execInsightGrid", insightData.insights);
}

// ===================================================================
// PAGE: SALES
// ===================================================================
async function loadSales() {
  const params = buildFilterParams();
  const [catData, stateData, monthlyData, sellerData] = await Promise.all([
    apiFetch("/api/categories", { ...params, top: 20 }),
    apiFetch("/api/states", params),
    apiFetch("/api/monthly-trends", params),
    apiFetch("/api/sellers", { top: 20, sort: "product_sales_value" }),
  ]);

  const cats    = catData.performance;
  const catRevs = catData.reviews;
  const cNames  = cats.map(r => fmt_category(r.product_category_name_english)).reverse();

  // Sales by category
  Plotly.newPlot("chartSalesByCat", [{
    type: "bar", orientation: "h",
    x: cats.map(r => r.product_sales_value).reverse(), y: cNames,
    marker: { color: "#2563eb" },
    hovertemplate: "<b>%{y}</b><br>Sales: R$ %{x:,.0f}<extra></extra>",
  }], { ...baseLayout("","R$",""), margin: { t:10,r:20,b:45,l:200 } }, PLOTLY_CONFIG);

  // Orders by category
  Plotly.newPlot("chartOrdersByCat", [{
    type: "bar", orientation: "h",
    x: cats.map(r => r.order_count).reverse(), y: cNames,
    marker: { color: "#0d9488" },
    hovertemplate: "<b>%{y}</b><br>Orders: %{x:,}<extra></extra>",
  }], { ...baseLayout("","Orders",""), margin: { t:10,r:20,b:45,l:200 } }, PLOTLY_CONFIG);

  // AOV by category
  Plotly.newPlot("chartAovByCat", [{
    type: "bar", orientation: "h",
    x: cats.map(r => r.avg_order_value).reverse(), y: cNames,
    marker: { color: "#7c3aed" },
    hovertemplate: "<b>%{y}</b><br>AOV: R$ %{x:,.2f}<extra></extra>",
  }], { ...baseLayout("","R$",""), margin: { t:10,r:20,b:45,l:200 } }, PLOTLY_CONFIG);

  // Review by category — join catRevs
  const revMap = {};
  catRevs.forEach(r => { revMap[r.product_category_name_english] = r.avg_review_score; });
  const revScores = cats.map(r => revMap[r.product_category_name_english] ?? null);

  Plotly.newPlot("chartReviewByCat", [{
    type: "bar", orientation: "h",
    x: revScores.reverse(), y: cNames,
    marker: { color: revScores.map(v => v == null ? "#e5e7eb" : v >= 4.2 ? "#16a34a" : v >= 3.8 ? "#d97706" : "#dc2626").reverse() },
    hovertemplate: "<b>%{y}</b><br>Avg Review: %{x:.3f}<extra></extra>",
  }], { ...baseLayout("","Score",""), xaxis: { range:[0,5] }, margin: { t:10,r:20,b:45,l:200 } }, PLOTLY_CONFIG);

  // Monthly sales trend
  Plotly.newPlot("chartSalesTrend", [
    { type:"scatter", mode:"lines+markers", name:"Product Sales",
      x: monthlyData.map(r=>r.ym), y: monthlyData.map(r=>r.product_sales_value),
      line:{color:"#2563eb",width:2.5}, fill:"tozeroy", fillcolor:"rgba(37,99,235,.08)",
      hovertemplate:"<b>%{x}</b><br>Sales: R$ %{y:,.0f}<extra></extra>" },
    { type:"scatter", mode:"lines+markers", name:"Order Value",
      x: monthlyData.map(r=>r.ym), y: monthlyData.map(r=>r.order_value),
      line:{color:"#7c3aed",width:2, dash:"dot"},
      hovertemplate:"<b>%{x}</b><br>Order Value: R$ %{y:,.0f}<extra></extra>" },
  ], { ...baseLayout("","Month","R$"), margin:{t:10,r:16,b:55,l:70} }, PLOTLY_CONFIG);

  // Sales by state (top 15)
  const stPerf = stateData.performance.slice(0,15).reverse();
  Plotly.newPlot("chartSalesByState", [{
    type:"bar", orientation:"h",
    x: stPerf.map(r=>r.product_sales_value), y: stPerf.map(r=>r.customer_state),
    marker: { color: "#d97706" },
    hovertemplate:"<b>%{y}</b><br>Sales: R$ %{x:,.0f}<extra></extra>",
  }], { ...baseLayout("","R$",""), margin:{t:10,r:20,b:45,l:55} }, PLOTLY_CONFIG);

  // Top sellers
  const sellers = sellerData.sellers;
  Plotly.newPlot("chartTopSellers", [
    { type:"bar", name:"Product Sales (R$)",
      x: sellers.map(r=>fmt_seller_id(r.seller_id)),
      y: sellers.map(r=>r.product_sales_value),
      marker:{color:"#2563eb"},
      customdata: sellers.map(r=>r.seller_id),
      hovertemplate:"<b>Seller: %{customdata}</b><br>Sales: R$ %{y:,.0f}<extra></extra>" },
    { type:"scatter", mode:"markers", name:"Late % (right axis)", yaxis:"y2",
      x: sellers.map(r=>fmt_seller_id(r.seller_id)),
      y: sellers.map(r=>r.late_pct),
      marker:{color:"#dc2626",size:8,symbol:"circle"},
      hovertemplate:"<b>Seller: ...%{x}</b><br>Late%%: %{y:.1f}%<extra></extra>" },
  ], {
    ...baseLayout("","Seller","R$"),
    yaxis2: { title:"Late %", overlaying:"y", side:"right", tickfont:{size:11}, range:[0,35] },
    margin:{t:10,r:60,b:60,l:70}, barmode:"group",
  }, PLOTLY_CONFIG);

  // Sales insights
  const insD = await apiFetch("/api/insights", { category: "sales" });
  renderInsightGrid("salesInsightGrid", insD.insights);
}

// ===================================================================
// PAGE: CUSTOMERS
// ===================================================================
async function loadCustomers() {
  const [custData, insightData] = await Promise.all([
    apiFetch("/api/customers"),
    apiFetch("/api/insights", { category: "customer" }),
  ]);

  const rfmSegs = custData.rfm_segments;
  const repeat  = custData.repeat_onetime;
  const geo     = custData.geography;
  const spDist  = custData.spending_dist;
  const segSp   = custData.segment_spend;

  // RFM Pie
  Plotly.newPlot("chartRfmPie", [{
    type: "pie",
    labels: rfmSegs.map(r=>r.segment),
    values: rfmSegs.map(r=>r.customer_count),
    marker: { colors: rfmSegs.map(r=>SEG_COLORS[r.segment]||"#9ca3af") },
    hovertemplate:"<b>%{label}</b><br>Customers: %{value:,}<br>%{percent}<extra></extra>",
    textinfo: "label+percent", textfont:{size:11},
  }], { ...baseLayout(),...{ margin:{t:10,r:10,b:10,l:10}, showlegend:false } }, PLOTLY_CONFIG);

  // Revenue by RFM segment
  const rfmSorted = [...rfmSegs].sort((a,b)=>b.total_revenue-a.total_revenue);
  Plotly.newPlot("chartRfmRevenue", [{
    type:"bar",
    x: rfmSorted.map(r=>r.segment),
    y: rfmSorted.map(r=>r.total_revenue),
    marker:{ color: rfmSorted.map(r=>SEG_COLORS[r.segment]||"#9ca3af") },
    hovertemplate:"<b>%{x}</b><br>Revenue: R$ %{y:,.0f}<extra></extra>",
  }], { ...baseLayout("","Segment","R$"), margin:{t:10,r:16,b:90,l:70},
    xaxis:{tickangle:-30} }, PLOTLY_CONFIG);

  // Avg & Median spend by segment
  Plotly.newPlot("chartSegmentSpend", [
    { type:"bar", name:"Avg Spend",
      x: segSp.map(r=>r.segment), y: segSp.map(r=>r.avg_spend),
      marker:{color:"#2563eb"}, hovertemplate:"<b>%{x}</b><br>Avg: R$ %{y:,.2f}<extra></extra>" },
    { type:"bar", name:"Median Spend",
      x: segSp.map(r=>r.segment), y: segSp.map(r=>r.median_spend),
      marker:{color:"#0d9488"}, hovertemplate:"<b>%{x}</b><br>Median: R$ %{y:,.2f}<extra></extra>" },
  ], { ...baseLayout("","Segment","R$"), barmode:"group",
    margin:{t:10,r:16,b:90,l:70}, xaxis:{tickangle:-30} }, PLOTLY_CONFIG);

  // One-time vs Repeat (grouped bar)
  Plotly.newPlot("chartRepeat", [
    { type:"bar", name:"Customers", x: repeat.map(r=>r.customer_type),
      y: repeat.map(r=>r.customer_count), marker:{color:["#2563eb","#16a34a"]},
      hovertemplate:"<b>%{x}</b><br>%{y:,} customers<extra></extra>" },
  ], { ...baseLayout("","Type","Customers"), margin:{t:10,r:16,b:50,l:70} }, PLOTLY_CONFIG);

  // Customer count by state (top 20)
  const geoTop = [...geo].sort((a,b)=>b.customer_uid_count-a.customer_uid_count).slice(0,20).reverse();
  Plotly.newPlot("chartCustByState", [{
    type:"bar", orientation:"h",
    x: geoTop.map(r=>r.customer_uid_count), y: geoTop.map(r=>r.customer_state),
    marker:{color:"#2563eb"},
    hovertemplate:"<b>%{y}</b><br>Customers: %{x:,}<extra></extra>",
  }], { ...baseLayout("","Customers",""), margin:{t:10,r:20,b:45,l:55} }, PLOTLY_CONFIG);

  // Revenue by state (top 20)
  const geoRevTop = [...geo].sort((a,b)=>b.total_revenue-a.total_revenue).slice(0,20).reverse();
  Plotly.newPlot("chartRevByState", [{
    type:"bar", orientation:"h",
    x: geoRevTop.map(r=>r.total_revenue), y: geoRevTop.map(r=>r.customer_state),
    marker:{color:"#7c3aed"},
    hovertemplate:"<b>%{y}</b><br>Revenue: R$ %{x:,.0f}<extra></extra>",
  }], { ...baseLayout("","R$",""), margin:{t:10,r:20,b:45,l:55} }, PLOTLY_CONFIG);

  // Spending distribution
  Plotly.newPlot("chartSpendDist", [
    { type:"bar", name:"Customers",
      x: spDist.map(r=>r.spend_bucket), y: spDist.map(r=>r.customer_count),
      marker:{color:"#2563eb"},
      hovertemplate:"<b>%{x}</b><br>Customers: %{y:,}<extra></extra>" },
    { type:"scatter", mode:"lines+markers", name:"Revenue % (right axis)", yaxis:"y2",
      x: spDist.map(r=>r.spend_bucket), y: spDist.map(r=>r.revenue_pct),
      line:{color:"#d97706",width:2.5}, marker:{size:6},
      hovertemplate:"<b>%{x}</b><br>Revenue %%: %{y:.1f}%<extra></extra>" },
  ], {
    ...baseLayout("","Spending Bucket","Customers"),
    yaxis2:{ title:"Revenue %", overlaying:"y", side:"right", tickfont:{size:11} },
    margin:{t:10,r:60,b:55,l:70},
  }, PLOTLY_CONFIG);

  // RFM Table
  buildRfmTable(rfmSegs);

  // Insights
  renderInsightGrid("custInsightGrid", insightData.insights);
}

function buildRfmTable(segs) {
  const tbody = document.querySelector("#rfmTable tbody");
  tbody.innerHTML = "";
  const total_cust = segs.reduce((s,r)=>s+(r.customer_count||0),0);
  const total_rev  = segs.reduce((s,r)=>s+(r.total_revenue||0),0);

  segs.forEach(r => {
    const pct_c = total_cust ? (r.customer_count/total_cust*100).toFixed(1) : "–";
    const pct_r = total_rev  ? (r.total_revenue/total_rev*100).toFixed(1) : "–";
    const color = SEG_COLORS[r.segment] || "#9ca3af";
    tbody.innerHTML += `<tr>
      <td><span class="seg-dot" style="background:${color}"></span>${r.segment}</td>
      <td class="numeric">${fmt_num(r.customer_count)}</td>
      <td class="numeric">${pct_c}%</td>
      <td class="numeric">${fmt_currency(r.total_revenue)}</td>
      <td class="numeric">${pct_r}%</td>
      <td class="numeric">${fmt_currency(r.avg_spend)}</td>
      <td class="numeric">${fmt_currency(r.median_spend)}</td>
      <td class="numeric">${r.avg_frequency ? r.avg_frequency.toFixed(3) : "–"}</td>
      <td class="numeric">${r.avg_recency_days ? Math.round(r.avg_recency_days)+"d" : "–"}</td>
    </tr>`;
  });
}

// ===================================================================
// PAGE: DELIVERY
// ===================================================================
async function loadDelivery() {
  const params = buildFilterParams();
  const [delData, revData, sellerData, mlData, insData] = await Promise.all([
    apiFetch("/api/delivery", params),
    apiFetch("/api/reviews"),
    apiFetch("/api/sellers", { top: 30, sort: "late_pct", asc: "false" }),
    apiFetch("/api/ml"),
    apiFetch("/api/insights", { category: "delivery" }),
  ]);

  const ov = delData.overview;
  setText("d-ontime",   fmt_pct(ov.on_time_pct));
  setText("d-late",     fmt_pct(ov.late_pct));
  setText("d-avgdays",  ov.avg_delivery_days ? ov.avg_delivery_days.toFixed(1)+"d" : "–");
  setText("d-meddays",  ov.median_delivery_days ? ov.median_delivery_days.toFixed(1)+"d" : "–");
  setText("d-early",    ov.avg_delay_days ? ov.avg_delay_days.toFixed(1)+"d" : "–");
  setText("d-review",   fmt_score(mlData.metrics.logistic_regression ? null : null));

  // Get avg review from reviews endpoint
  const allRevDelay = revData.by_delay;
  const natRevScore = allRevDelay.reduce((sum,r)=>sum+(r.avg_review_score*r.order_count),0) /
                      allRevDelay.reduce((sum,r)=>sum+r.order_count,0);
  setText("d-review", fmt_score(natRevScore));

  // Delivery over time (dual axis)
  const dTime = delData.over_time;
  Plotly.newPlot("chartDelTime", [
    { type:"scatter", mode:"lines+markers", name:"On-Time %",
      x: dTime.map(r=>r.ym), y: dTime.map(r=>r.on_time_pct),
      line:{color:"#16a34a",width:2.5}, marker:{size:5},
      hovertemplate:"<b>%{x}</b><br>On-Time: %{y:.1f}%<extra></extra>" },
    { type:"scatter", mode:"lines+markers", name:"Late %",
      x: dTime.map(r=>r.ym), y: dTime.map(r=>r.late_pct),
      line:{color:"#dc2626",width:2.5}, marker:{size:5},
      hovertemplate:"<b>%{x}</b><br>Late: %{y:.1f}%<extra></extra>" },
    { type:"scatter", mode:"lines+markers", name:"Avg Review (right)", yaxis:"y2",
      x: dTime.map(r=>r.ym), y: dTime.map(r=>r.avg_review_score),
      line:{color:"#d97706",width:2,dash:"dot"}, marker:{size:5},
      hovertemplate:"<b>%{x}</b><br>Avg Review: %{y:.3f}<extra></extra>" },
  ], {
    ...baseLayout("","Month","%"),
    yaxis2:{ title:"Avg Review Score", overlaying:"y", side:"right", range:[3.5,4.6], tickfont:{size:11} },
    margin:{t:10,r:60,b:55,l:60},
  }, PLOTLY_CONFIG);

  // Delay distribution
  const dDist = delData.delay_dist;
  Plotly.newPlot("chartDelayDist", [{
    type:"bar",
    x: dDist.map(r=>r.delay_bucket),
    y: dDist.map(r=>r.pct),
    marker:{ color: dDist.map(r=>DELAY_COLORS[r.delay_bucket]||"#9ca3af") },
    hovertemplate:"<b>%{x}</b><br>%{y:.1f}%<br>Orders: %{customdata:,}<extra></extra>",
    customdata: dDist.map(r=>r.order_count),
  }], { ...baseLayout("","Bucket","%"), margin:{t:10,r:16,b:80,l:55}, xaxis:{tickangle:-30} }, PLOTLY_CONFIG);

  // Late % by state
  const dState = [...delData.by_state].sort((a,b)=>b.late_pct-a.late_pct).slice(0,27).reverse();
  Plotly.newPlot("chartDelByState", [{
    type:"bar", orientation:"h",
    x: dState.map(r=>r.late_pct), y: dState.map(r=>r.customer_state),
    marker:{ color: dState.map(r => r.late_pct > 15 ? "#dc2626" : r.late_pct > 10 ? "#f97316" : r.late_pct > 6.77 ? "#d97706" : "#16a34a") },
    hovertemplate:"<b>%{y}</b><br>Late: %{x:.1f}%<br>Orders: %{customdata:,}<extra></extra>",
    customdata: dState.map(r=>r.order_count),
  }], { ...baseLayout("","%",""), margin:{t:10,r:30,b:45,l:55},
    shapes:[{ type:"line", x0:6.77, x1:6.77, y0:-0.5, y1:dState.length-0.5,
              line:{color:"#2563eb",width:1.5,dash:"dot"} }],
    annotations:[{ x:6.77, y:dState.length-1, xanchor:"left", text:"National avg", font:{size:10,color:"#2563eb"}, showarrow:false }],
  }, PLOTLY_CONFIG);

  // Late % by category (top 20, >=100 orders)
  const dCat = [...delData.by_category]
    .filter(r => r.order_count >= 100)
    .sort((a,b)=>b.late_pct-a.late_pct).slice(0,20).reverse();
  Plotly.newPlot("chartDelByCat", [{
    type:"bar", orientation:"h",
    x: dCat.map(r=>r.late_pct), y: dCat.map(r=>fmt_category(r.product_category_name_english)),
    marker:{ color: dCat.map(r => r.late_pct > 10 ? "#dc2626" : r.late_pct > 7 ? "#f97316" : "#d97706") },
    hovertemplate:"<b>%{y}</b><br>Late: %{x:.1f}%<extra></extra>",
  }], { ...baseLayout("","%",""), margin:{t:10,r:30,b:45,l:200} }, PLOTLY_CONFIG);

  // Review score by delay bucket
  const rDel = revData.by_delay;
  Plotly.newPlot("chartRevDelay", [{
    type:"bar",
    x: rDel.map(r=>r.delay_bucket),
    y: rDel.map(r=>r.avg_review_score),
    marker:{ color: rDel.map(r=>DELAY_COLORS[r.delay_bucket]||"#9ca3af") },
    hovertemplate:"<b>%{x}</b><br>Avg Score: %{y:.3f}<br>n=%{customdata:,}<br>1★: %{text}<extra></extra>",
    customdata: rDel.map(r=>r.order_count),
    text: rDel.map(r=>r.pct_1star ? r.pct_1star.toFixed(1)+"%" : "–"),
  }], { ...baseLayout("","Delay Bucket","Avg Review Score"),
    yaxis:{range:[0,5.2]}, margin:{t:10,r:16,b:80,l:60}, xaxis:{tickangle:-30} }, PLOTLY_CONFIG);

  // Seller delivery (top 30 by late%, qualified)
  const sellers = sellerData.sellers.slice(0,30);
  Plotly.newPlot("chartSellerDel", [
    { type:"bar", name:"Late %",
      x: sellers.map(r=>fmt_seller_id(r.seller_id)), y: sellers.map(r=>r.late_pct),
      marker:{color:"#dc2626"},
      customdata: sellers.map(r=>r.seller_id),
      hovertemplate:"<b>%{customdata}</b><br>Late: %{y:.1f}%<br>Orders: %{meta:,}<extra></extra>",
      meta: sellers.map(r=>r.order_count) },
    { type:"scatter", mode:"markers", name:"Avg Review (right)", yaxis:"y2",
      x: sellers.map(r=>fmt_seller_id(r.seller_id)), y: sellers.map(r=>r.avg_review_score),
      marker:{color:"#d97706",size:7},
      hovertemplate:"<b>%{x}</b><br>Review: %{y:.3f}<extra></extra>" },
  ], {
    ...baseLayout("","Seller","%"),
    yaxis2:{ title:"Avg Review", overlaying:"y", side:"right", range:[1,5.5], tickfont:{size:11} },
    margin:{t:10,r:60,b:70,l:55}, xaxis:{tickangle:-45},
  }, PLOTLY_CONFIG);

  // ML metrics
  const lr = mlData.metrics.logistic_regression;
  setText("ml-lr-auc",      fmt_score(lr.roc_auc));
  setText("ml-lr-recall",   fmt_pct(lr.recall*100));
  setText("ml-lr-precision",fmt_pct(lr.precision*100));
  setText("ml-lr-f1",       fmt_score(lr.f1));

  // RF feature importance
  const fi = mlData.feature_importance.slice(0,15);
  Plotly.newPlot("chartMlFI", [{
    type:"bar", orientation:"h",
    x: fi.map(r=>r.importance).reverse(),
    y: fi.map(r=>r.feature).reverse(),
    marker:{color:"#7c3aed"},
    hovertemplate:"<b>%{y}</b><br>Importance: %{x:.5f}<extra></extra>",
  }], { ...baseLayout("","Importance",""), margin:{t:10,r:20,b:45,l:180} }, PLOTLY_CONFIG);

  // LR coefficients chart
  const lrc = mlData.logistic_regression_coefficients || [];
  if (lrc.length && document.getElementById("chartMlLRCoef")) {
    const lrcSorted = [...lrc].sort((a,b) => a.coefficient - b.coefficient);
    Plotly.newPlot("chartMlLRCoef", [{
      type: "bar", orientation: "h",
      x: lrcSorted.map(r => r.coefficient),
      y: lrcSorted.map(r => r.feature),
      marker: { color: lrcSorted.map(r => r.coefficient > 0 ? "#dc2626" : "#16a34a") },
      hovertemplate: "<b>%{y}</b><br>Coefficient: %{x:.4f}<br>Direction: %{customdata}<extra></extra>",
      customdata: lrcSorted.map(r => r.direction),
    }], {
      ...baseLayout("","Coefficient",""),
      shapes:[{ type:"line", x0:0, x1:0, y0:-0.5, y1:lrcSorted.length-0.5, line:{color:"#374151",width:1,dash:"dot"} }],
      margin:{t:10,r:20,b:45,l:190},
    }, PLOTLY_CONFIG);
  }

  // LR confusion matrix heatmap
  const cm = mlData.confusion_matrices.find(r=>r.model==="logistic_regression");
  if (cm) {
    const cmVals = [[cm.TN, cm.FP],[cm.FN, cm.TP]];
    Plotly.newPlot("chartMlCM", [{
      type:"heatmap",
      z: cmVals,
      x:["Predicted On-Time","Predicted Late"],
      y:["Actual On-Time","Actual Late"],
      colorscale:[[0,"#f0f9ff"],[0.5,"#bfdbfe"],[1,"#1d4ed8"]],
      showscale: false,
      hovertemplate:"<b>%{y} → %{x}</b><br>Count: %{z:,}<extra></extra>",
      text: cmVals.map(row=>row.map(v=>v.toLocaleString())),
      texttemplate:"%{text}",
    }], { ...baseLayout("","",""), margin:{t:10,r:10,b:60,l:110} }, PLOTLY_CONFIG);
  }

  // Delivery insights
  const satIns = await apiFetch("/api/insights", { category: "satisfaction" });
  renderInsightGrid("delivInsightGrid", [...insData.insights, ...satIns.insights]);
}

// ===================================================================
// INSIGHT RENDERING
// ===================================================================
function renderInsightGrid(containerId, insights) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";
  if (!insights || !insights.length) {
    container.innerHTML = '<div style="color:#6b7280;font-size:13px;">No insights available.</div>';
    return;
  }

  insights.forEach(ins => {
    const typeClass = ins.type === "risk" ? "risk" : "opportunity";
    const badgeClass = ins.type === "risk" ? "badge-risk" : "badge-opportunity";
    const badgeLabel = ins.type === "risk" ? "⚠ Risk" : "✓ Opportunity";
    const catLabel = ins.category ? ins.category.charAt(0).toUpperCase()+ins.category.slice(1) : "";

    const card = document.createElement("div");
    card.className = `insight-card ${typeClass}`;
    card.dataset.type = ins.type;
    card.dataset.id   = ins.id;

    card.innerHTML = `
      <div class="insight-header">
        <span class="insight-id">${ins.id || ""}</span>
        <div style="display:flex;gap:5px;align-items:center;">
          <span class="badge-cat">${catLabel}</span>
          <span class="insight-type-badge ${badgeClass}">${badgeLabel}</span>
        </div>
      </div>
      <div class="insight-fact">${escHtml(ins.fact)}</div>
      <div class="insight-body">${escHtml(ins.insight)}</div>
      ${ins.metric ? `<div class="insight-metric">${escHtml(ins.metric)}</div>` : ""}
      <div class="insight-rec"><strong>Recommended Action:</strong> ${escHtml(ins.recommendation)}</div>
      ${ins.qualification ? `<div class="insight-qual">${escHtml(ins.qualification)}</div>` : ""}
    `;
    container.appendChild(card);
  });
}

function filterInsightCards(filter) {
  const cards = document.querySelectorAll("#execInsightGrid .insight-card");
  cards.forEach(card => {
    const show = filter === "all" || card.dataset.type === filter;
    card.style.display = show ? "" : "none";
  });
  // Update tab counts
  const risks = document.querySelectorAll("#execInsightGrid .insight-card.risk").length;
  const opps  = document.querySelectorAll("#execInsightGrid .insight-card.opportunity").length;
  document.querySelectorAll(".insight-tab").forEach(btn => {
    if (btn.dataset.filter === "risk")        btn.textContent = `Risks (${risks})`;
    if (btn.dataset.filter === "opportunity") btn.textContent = `Opportunities (${opps})`;
    if (btn.dataset.filter === "all")         btn.textContent = `All (${risks+opps})`;
  });
}

// ===================================================================
// UTILITIES
// ===================================================================
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function escHtml(str) {
  if (!str) return "";
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function buildFilterParams() {
  const p = {};
  if (State.filters.states.length)     p.states     = State.filters.states.join(",");
  if (State.filters.categories.length) p.categories = State.filters.categories.join(",");
  return p;
}

// ---- BOOT ----
document.addEventListener("DOMContentLoaded", init);
