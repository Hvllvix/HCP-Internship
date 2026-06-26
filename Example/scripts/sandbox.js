/**
 * sandbox.js
 * Comprehensive numerical scaler and imputation model routing simulator.
 * Fits within the 'Simpute' data architecture parameters.
 */

// Statistical distribution baselines for scaling simulations
const STAT_PROFILES = {
    S04Q03_AGE: { mean: 41.2, std: 18.5, min: 0, max: 100 },
    REVENU_MOL: { mean: 4250.0, std: 2450.0, min: 500, max: 25000 },
    Household_Size: { mean: 4.8, std: 2.1, min: 1, max: 18 }
  };
  
  /**
   * Sync default numerical values when dropdown variable switches
   */
  function updateDefaultValue() {
    const selectedVar = document.getElementById("var-select").value;
    const rawInput = document.getElementById("raw-value");
    
    if (selectedVar === "S04Q03_AGE") {
      rawInput.value = 42;
    } else if (selectedVar === "REVENU_MOL") {
      rawInput.value = 3500;
    } else if (selectedVar === "Household_Size") {
      rawInput.value = 5;
    }
  }
  
  /**
   * Resets sandbox controls to baseline states
   */
  function resetSandbox() {
    document.getElementById("var-select").selectedIndex = 0;
    document.getElementById("scaler-select").selectedIndex = 0;
    document.getElementById("mask-select").selectedIndex = 0;
    document.getElementById("engine-select").selectedIndex = 0;
    updateDefaultValue();
    
    const terminal = document.getElementById("console-terminal");
    terminal.innerHTML = `[Ready] System pipeline configurations reset to defaults.\n[Ready] Awaiting telemetry parameters to execute...`;
  }
  
  /**
   * Executes statistical scaling, random masking checks, and model imputation simulator
   */
  function runSandboxEngine() {
    const terminal = document.getElementById("console-terminal");
    
    const selectedVar = document.getElementById("var-select").value;
    const rawVal = parseFloat(document.getElementById("raw-value").value);
    const scalerType = document.getElementById("scaler-select").value;
    const maskRatio = parseFloat(document.getElementById("mask-select").value);
    const modelEngine = document.getElementById("engine-select").value;
  
    // Basic validate constraints
    if (isNaN(rawVal)) {
      terminal.innerHTML = `<span style="color: var(--text-error)">[ERROR] Execution halted: Input observation is not a valid number.</span>`;
      return;
    }
  
    const profile = STAT_PROFILES[selectedVar];
    
    // Boundary constraints enforcement
    if (rawVal < profile.min || rawVal > profile.max) {
      terminal.innerHTML = `<span style="color: var(--text-warning)">[WARNING] Out of bounds: ${selectedVar} value ${rawVal} violates logical bounds [${profile.min}, ${profile.max}].\nContinuing with pipeline execution constraints...</span>\n\n`;
    } else {
      terminal.innerHTML = ``;
    }
  
    terminal.innerHTML += `[SYSTEM] Processing matrix payload for variable: ${selectedVar}\n`;
    terminal.innerHTML += `[SYSTEM] Raw Observation input value detected: ${rawVal}\n`;
  
    // Scale computation
    let scaledValue = rawVal;
    if (scalerType === "StandardScaler") {
      scaledValue = (rawVal - profile.mean) / profile.std;
      terminal.innerHTML += `[SCALER] StandardScaler fitted. Transform result: ${scaledValue.toFixed(4)} (z-score)\n`;
    } else if (scalerType === "MinMaxScaler") {
      scaledValue = (rawVal - profile.min) / (profile.max - profile.min);
      terminal.innerHTML += `[SCALER] MinMaxScaler fitted. Transform result: ${scaledValue.toFixed(4)} (bound [0,1])\n`;
    } else {
      terminal.innerHTML += `[SCALER] Scaling transformations disabled. Continuous metric unmodified.\n`;
    }
  
    // Masking determination (simulates if this specific cell gets corrupted)
    const isMasked = Math.random() < maskRatio;
    terminal.innerHTML += `[MASKER] Assessing missingness risk. Threshold = ${maskRatio}\n`;
    
    if (maskRatio >= 0.70) {
      terminal.innerHTML += `<span style="color: var(--text-warning)">[WARNING] Missingness threshold exceeded 70% threshold (Current: ${maskRatio * 100}%).\nSimpute core triggers data structural degradation flag.</span>\n`;
    }
  
    if (isMasked) {
      terminal.innerHTML += `[MASKER] Cell index matches mask criteria. State set to NaN (Missing).\n`;
      terminal.innerHTML += `[ENGINE] Triggering dynamic routing engine: ${modelEngine}\n`;
      
      // Imputation heuristics
      let imputedVal = 0;
      let deviationOffset = (Math.random() - 0.5) * (profile.std * 0.1); // Small variance offset to simulate model predictions
      
      if (modelEngine === "KNN") {
        imputedVal = profile.mean + deviationOffset;
        terminal.innerHTML += `[ENGINE] KNNRegressor selected target neighborhood points. Imputing value...\n`;
      } else if (modelEngine === "LGBM") {
        imputedVal = (profile.mean * 1.02) + deviationOffset;
        terminal.innerHTML += `[ENGINE] LightGBMRegressor evaluated gradient boosted leaf metrics. Imputing value...\n`;
      } else if (modelEngine === "BayesianRidge") {
        imputedVal = (profile.mean * 0.98) + deviationOffset;
        terminal.innerHTML += `[ENGINE] BayesianRidge computed posterior Gaussian estimates. Imputing value...\n`;
      }
  
      // Clip output value to legitimate bounds
      imputedVal = Math.max(profile.min, Math.min(profile.max, imputedVal));
      
      terminal.innerHTML += `\n<span style="color: var(--accent-mint)">[SUCCESS] Imputation complete.\n[SUCCESS] Reconstructed Value: ${imputedVal.toFixed(2)}\n[SUCCESS] Variance Retention: OK</span>`;
    } else {
      terminal.innerHTML += `[MASKER] Cell index remains active (Intact). Imputation engine bypassed.\n`;
      terminal.innerHTML += `\n<span style="color: var(--accent-mint)">[SUCCESS] Observation preserved. Output value: ${rawVal}</span>`;
    }
  }