/**
 * geomap.js
 * Interactive Moroccan Geographic Intelligence Engine
 * Handles vector rendering, event hooks, and region-specific statistical updates.
 */

// Comprehensive statistical database for the 12 regions of Morocco (ENCDM/RGPH targets)
const REGIONAL_DATABASE = {
    "Guelmim-Oued Noun": {
      frenchName: "Guelmim-Oued Noun",
      povertyRate: "14.2%",
      vulnerabilityRate: "19.8%",
      householdCount: "114,350",
      avgHouseholdSize: "5.1",
      cleanWaterAccess: "74.5%",
      electricityAccess: "92.1%",
      primaryModel: "CatBoostClassifier",
      activeSurveyWeight: "420.15",
      povertyIndex: 0.142,
      vulnerabilityIndex: 0.198,
      capital: "Guelmim"
    },
    "Souss-Massa": {
      frenchName: "Souss-Massa",
      povertyRate: "6.2%",
      vulnerabilityRate: "12.4%",
      householdCount: "612,400",
      avgHouseholdSize: "4.8",
      cleanWaterAccess: "81.2%",
      electricityAccess: "95.6%",
      primaryModel: "LightGBMRegressor",
      activeSurveyWeight: "610.40",
      povertyIndex: 0.062,
      vulnerabilityIndex: 0.124,
      capital: "Agadir"
    },
    "Casablanca-Settat": {
      frenchName: "Casablanca-Settat",
      povertyRate: "2.1%",
      vulnerabilityRate: "6.5%",
      householdCount: "1,684,200",
      avgHouseholdSize: "4.2",
      cleanWaterAccess: "97.8%",
      electricityAccess: "99.2%",
      primaryModel: "LightGBMRegressor",
      activeSurveyWeight: "1142.50",
      povertyIndex: 0.021,
      vulnerabilityIndex: 0.065,
      capital: "Casablanca"
    },
    "Rabat-Salé-Kénitra": {
      frenchName: "Rabat-Salé-Kénitra",
      povertyRate: "3.4%",
      vulnerabilityRate: "8.1%",
      householdCount: "1,120,500",
      avgHouseholdSize: "4.4",
      cleanWaterAccess: "94.5%",
      electricityAccess: "98.1%",
      primaryModel: "ExtraTreesRegressor",
      activeSurveyWeight: "890.60",
      povertyIndex: 0.034,
      vulnerabilityIndex: 0.081,
      capital: "Rabat"
    },
    "Tanger-Tétouan-Al Hoceïma": {
      frenchName: "Tanger-Tétouan-Al Hoceïma",
      povertyRate: "5.8%",
      vulnerabilityRate: "11.2%",
      householdCount: "845,100",
      avgHouseholdSize: "4.7",
      cleanWaterAccess: "86.4%",
      electricityAccess: "96.8%",
      primaryModel: "LightGBMClassifier",
      activeSurveyWeight: "740.20",
      povertyIndex: 0.058,
      vulnerabilityIndex: 0.112,
      capital: "Tanger"
    },
    "L'Oriental": {
      frenchName: "L'Oriental",
      povertyRate: "8.4%",
      vulnerabilityRate: "14.6%",
      householdCount: "512,300",
      avgHouseholdSize: "4.9",
      cleanWaterAccess: "79.1%",
      electricityAccess: "93.4%",
      primaryModel: "BayesianRidge",
      activeSurveyWeight: "495.80",
      povertyIndex: 0.084,
      vulnerabilityIndex: 0.146,
      capital: "Oujda"
    },
    "Fès-Meknès": {
      frenchName: "Fès-Meknès",
      povertyRate: "7.1%",
      vulnerabilityRate: "13.2%",
      householdCount: "982,400",
      avgHouseholdSize: "4.8",
      cleanWaterAccess: "84.3%",
      electricityAccess: "95.1%",
      primaryModel: "LightGBMRegressor",
      activeSurveyWeight: "812.30",
      povertyIndex: 0.071,
      vulnerabilityIndex: 0.132,
      capital: "Fès"
    },
    "Béni Mellal-Khénifra": {
      frenchName: "Béni Mellal-Khénifra",
      povertyRate: "9.3%",
      vulnerabilityRate: "16.1%",
      householdCount: "524,100",
      avgHouseholdSize: "5.2",
      cleanWaterAccess: "72.4%",
      electricityAccess: "91.8%",
      primaryModel: "KNNRegressor",
      activeSurveyWeight: "480.90",
      povertyIndex: 0.093,
      vulnerabilityIndex: 0.161,
      capital: "Béni Mellal"
    },
    "Marrakech-Safi": {
      frenchName: "Marrakech-Safi",
      povertyRate: "6.8%",
      vulnerabilityRate: "12.9%",
      householdCount: "1,045,200",
      avgHouseholdSize: "5.0",
      cleanWaterAccess: "78.6%",
      electricityAccess: "94.3%",
      primaryModel: "LightGBMRegressor",
      activeSurveyWeight: "910.15",
      povertyIndex: 0.068,
      vulnerabilityIndex: 0.129,
      capital: "Marrakech"
    },
    "Drâa-Tafilalet": {
      frenchName: "Drâa-Tafilalet",
      povertyRate: "12.1%",
      vulnerabilityRate: "18.4%",
      householdCount: "312,800",
      avgHouseholdSize: "5.5",
      cleanWaterAccess: "68.2%",
      electricityAccess: "89.5%",
      primaryModel: "CatBoostClassifier",
      activeSurveyWeight: "312.40",
      povertyIndex: 0.121,
      vulnerabilityIndex: 0.184,
      capital: "Errachidia"
    },
    "Laâyoune-Sakia El Hamra": {
      frenchName: "Laâyoune-Sakia El Hamra",
      povertyRate: "2.4%",
      vulnerabilityRate: "7.1%",
      householdCount: "94,200",
      avgHouseholdSize: "4.5",
      cleanWaterAccess: "92.1%",
      electricityAccess: "97.4%",
      primaryModel: "ExtraTreesRegressor",
      activeSurveyWeight: "145.20",
      povertyIndex: 0.024,
      vulnerabilityIndex: 0.071,
      capital: "Laâyoune"
    },
    "Dakhla-Oued Ed-Dahab": {
      frenchName: "Dakhla-Oued Ed-Dahab",
      povertyRate: "1.8%",
      vulnerabilityRate: "5.9%",
      householdCount: "38,500",
      avgHouseholdSize: "4.1",
      cleanWaterAccess: "95.4%",
      electricityAccess: "98.2%",
      primaryModel: "BayesianRidge",
      activeSurveyWeight: "78.40",
      povertyIndex: 0.018,
      vulnerabilityIndex: 0.059,
      capital: "Dakhla"
    }
  };
  
  // Simplified coordinate geometry map to render a scalable Moroccan administrative map block
  const MAP_PATHS = [
    { id: "Tanger-Tétouan-Al Hoceïma", label: "Tanger-Tétouan-Al Hoceïma", d: "M 120 40 L 150 35 L 160 50 L 140 65 L 115 55 Z" },
    { id: "L'Oriental", label: "L'Oriental", d: "M 160 50 L 210 65 L 200 120 L 175 140 L 155 100 L 140 65 Z" },
    { id: "Fès-Meknès", label: "Fès-Meknès", d: "M 115 55 L 140 65 L 155 100 L 135 120 L 110 95 Z" },
    { id: "Rabat-Salé-Kénitra", label: "Rabat-Salé-Kénitra", d: "M 95 60 L 115 55 L 110 95 L 90 90 Z" },
    { id: "Béni Mellal-Khénifra", label: "Béni Mellal-Khénifra", d: "M 90 90 L 110 95 L 135 120 L 120 145 L 85 125 Z" },
    { id: "Casablanca-Settat", label: "Casablanca-Settat", d: "M 65 95 L 90 90 L 85 125 L 60 115 Z" },
    { id: "Marrakech-Safi", label: "Marrakech-Safi", d: "M 40 125 L 60 115 L 85 125 L 75 170 L 30 150 Z" },
    { id: "Drâa-Tafilalet", label: "Drâa-Tafilalet", d: "M 120 145 L 175 140 L 160 190 L 110 200 L 95 165 Z" },
    { id: "Souss-Massa", label: "Souss-Massa", d: "M 30 150 L 75 170 L 95 165 L 80 215 L 25 195 Z" },
    { id: "Guelmim-Oued Noun", label: "Guelmim-Oued Noun", d: "M 25 195 L 80 215 L 70 260 L 15 235 Z" },
    { id: "Laâyoune-Sakia El Hamra", label: "Laâyoune-Sakia El Hamra", d: "M 15 235 L 70 260 L 55 315 L 5 285 Z" },
    { id: "Dakhla-Oued Ed-Dahab", label: "Dakhla-Oued Ed-Dahab", d: "M 5 285 L 55 315 L 35 390 L 1 370 Z" }
  ];
  
  /**
   * Initializes and draws the vector interactive map inside the DOM target.
   * Automatically locks the active state on 'Guelmim-Oued Noun' by default.
   */
  function initMoroccoMap(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
  
    // Clear container
    container.innerHTML = "";
  
    // Create SVG Canvas
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 220 400");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.style.cursor = "pointer";
  
    // Append Interactive Path Nodes
    MAP_PATHS.forEach(region => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", region.d);
      path.setAttribute("id", `map-region-${region.id.replace(/\s+/g, '-')}`);
      path.setAttribute("class", "map-region-path");
      
      // Set standard presentation parameters (neutral white borders, standard transitions)
      path.style.fill = "#E5E7EB";
      path.style.stroke = "#FFFFFF";
      path.style.strokeWidth = "1.5px";
      path.style.transition = "fill 0.2s ease, stroke 0.2s ease";
  
      // Set Hover Event Listeners
      path.addEventListener("mouseenter", () => {
        if (path.getAttribute("data-active") !== "true") {
          path.style.fill = "#FEE2E2"; // Subtle coral tint
        }
      });
  
      path.addEventListener("mouseleave", () => {
        if (path.getAttribute("data-active") !== "true") {
          path.style.fill = "#E5E7EB";
        }
      });
  
      // Set Selection Event Listener
      path.addEventListener("click", () => {
        selectRegion(region.id);
      });
  
      svg.appendChild(path);
    });
  
    container.appendChild(svg);
  
    // Force Guelmim-Oued Noun selection state on load
    selectRegion("Guelmim-Oued Noun");
  }
  
  /**
   * Handles highlighting the vector path and updating statistical containers.
   */
  function selectRegion(regionId) {
    const targetData = REGIONAL_DATABASE[regionId];
    if (!targetData) return;
  
    // Reset all map paths
    document.querySelectorAll(".map-region-path").forEach(path => {
      path.setAttribute("data-active", "false");
      path.style.fill = "#E5E7EB";
      path.style.stroke = "#FFFFFF";
    });
  
    // Highlight selected path element
    const targetElementId = `map-region-${regionId.replace(/\s+/g, '-')}`;
    const targetPath = document.getElementById(targetElementId);
    if (targetPath) {
      targetPath.setAttribute("data-active", "true");
      targetPath.style.fill = "#FF6B4A"; // Accent Coral Color
      targetPath.style.stroke = "#FF6B4A";
    }
  
    // Update Statistics Panels in UI
    updateRegionalStatsUI(targetData);
  }
  
  /**
   * Updates UI textual readouts and structural micro-progress bars dynamically
   */
  function updateRegionalStatsUI(data) {
    // Update core text indicators
    const titleElem = document.getElementById("selected-region-title");
    const capitalElem = document.getElementById("selected-region-capital");
    const pdsElem = document.getElementById("selected-region-weight");
    const modelElem = document.getElementById("selected-region-model");
    const householdElem = document.getElementById("selected-region-households");
  
    if (titleElem) titleElem.innerText = data.frenchName;
    if (capitalElem) capitalElem.innerText = `Regional Capital: ${data.capital}`;
    if (pdsElem) pdsElem.innerText = data.activeSurveyWeight;
    if (modelElem) modelElem.innerText = data.primaryModel;
    if (householdElem) householdElem.innerText = data.householdCount;
  
    // Update dynamic metric metrics block
    const povertyValue = document.getElementById("stat-poverty-value");
    const povertyBar = document.getElementById("stat-poverty-bar");
    const vulnValue = document.getElementById("stat-vuln-value");
    const vulnBar = document.getElementById("stat-vuln-bar");
    const waterValue = document.getElementById("stat-water-value");
    const waterBar = document.getElementById("stat-water-bar");
    const elecValue = document.getElementById("stat-elec-value");
    const elecBar = document.getElementById("stat-elec-bar");
  
    if (povertyValue) povertyValue.innerText = data.povertyRate;
    if (povertyBar) povertyBar.style.width = `${data.povertyIndex * 100}%`;
  
    if (vulnValue) vulnValue.innerText = data.vulnerabilityRate;
    if (vulnBar) vulnBar.style.width = `${data.vulnerabilityIndex * 100}%`;
  
    if (waterValue) waterValue.innerText = data.cleanWaterAccess;
    if (waterBar) waterBar.style.width = data.cleanWaterAccess;
  
    if (elecValue) elecValue.innerText = data.electricityAccess;
    if (elecBar) elecBar.style.width = data.electricityAccess;
  }