/**
 * charts.js
 * Programmatic vector SVG rendering engine compiling 15 distinct high-density
 * statistical visualizations. Conforms to system color tokens & spacing grids.
 */

function initVectorChartsSuite() {
    // Define global style constants inside rendering scope
    const coralColor = "#FF6B4A";
    const mintColor = "#10B981";
    const grayMuted = "#9CA3AF";
    const textDark = "#1F2937";
    const gridLineColor = "#E5E7EB";
  
    // --------------------------------------------------------------------------
    // PLOT 1: Comparative KDE Distributions (Continuous Metrics)
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-kde-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 800 240" width="100%" height="100%">
          <!-- X/Y Axes Grid lines -->
          <line x1="50" y1="200" x2="750" y2="200" stroke="${gridLineColor}" stroke-width="1" />
          <line x1="50" y1="20" x2="50" y2="200" stroke="${gridLineColor}" stroke-width="1" />
          <line x1="50" y1="140" x2="750" y2="140" stroke="${gridLineColor}" stroke-dasharray="4" />
          <line x1="50" y1="80" x2="750" y2="80" stroke="${gridLineColor}" stroke-dasharray="4" />
          
          <!-- KDE Curve 1: Raw Observation (Muted Gray Line) -->
          <path d="M 50 190 Q 200 180 300 120 T 450 70 T 550 150 T 750 198" fill="none" stroke="${grayMuted}" stroke-width="2.5" />
          
          <!-- KDE Curve 2: Reconstructed Simpute Output (Coral Highlight Curve) -->
          <path d="M 50 195 Q 190 185 290 115 T 445 65 T 545 145 T 750 199" fill="none" stroke="${coralColor}" stroke-width="2.5" />
          <path d="M 50 195 Q 190 185 290 115 T 445 65 T 545 145 T 750 199 L 750 200 L 50 200 Z" fill="${coralColor}" fill-opacity="0.08" />
  
          <!-- Reference Labels -->
          <text x="50" y="215" fill="${textDark}" font-size="10" text-anchor="middle">0.0 (Age Lower Bound)</text>
          <text x="400" y="215" fill="${textDark}" font-size="10" text-anchor="middle">50.0 (Mean Value Target)</text>
          <text x="750" y="215" fill="${textDark}" font-size="10" text-anchor="middle">100.0 (Upper Bound)</text>
  
          <!-- Legends -->
          <circle cx="580" cy="30" r="5" fill="${grayMuted}" />
          <text x="592" y="33" fill="${textDark}" font-size="11">Raw Demographics</text>
          <circle cx="695" cy="30" r="5" fill="${coralColor}" />
          <text x="707" y="33" fill="${textDark}" font-size="11">Simpute Preserved</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 2: Double Heatmap completeness layout
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-completeness-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 800 200" width="100%" height="100%">
          <!-- Left Panel: Raw Missingness Holes -->
          <text x="180" y="20" fill="${textDark}" font-size="11" font-weight="600" text-anchor="middle">Raw Data (Observed Missingness)</text>
          <rect x="20" y="35" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="100" y="35" width="70" height="30" fill="${coralColor}" rx="3" />
          <rect x="180" y="35" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="260" y="35" width="70" height="30" fill="${mintColor}" rx="3" />
  
          <rect x="20" y="75" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="100" y="75" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="180" y="75" width="70" height="30" fill="${coralColor}" rx="3" />
          <rect x="260" y="75" width="70" height="30" fill="${mintColor}" rx="3" />
  
          <rect x="20" y="115" width="70" height="30" fill="${coralColor}" rx="3" />
          <rect x="100" y="115" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="180" y="115" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="260" y="115" width="70" height="30" fill="${mintColor}" rx="3" />
          
          <!-- Right Panel: Completely Solved Array -->
          <text x="620" y="20" fill="${textDark}" font-size="11" font-weight="600" text-anchor="middle">Post-Imputation (Reconstructed Matrix)</text>
          <rect x="460" y="35" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="540" y="35" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="620" y="35" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="700" y="35" width="70" height="30" fill="${mintColor}" rx="3" />
  
          <rect x="460" y="75" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="540" y="75" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="620" y="75" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="700" y="75" width="70" height="30" fill="${mintColor}" rx="3" />
  
          <rect x="460" y="115" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="540" y="115" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="620" y="115" width="70" height="30" fill="${mintColor}" rx="3" />
          <rect x="700" y="115" width="70" height="30" fill="${mintColor}" rx="3" />
  
          <!-- Vertical Divider -->
          <line x1="400" y1="10" x2="400" y2="160" stroke="${gridLineColor}" stroke-dasharray="3" />
  
          <!-- Key Legends -->
          <rect x="280" y="165" width="12" height="12" fill="${mintColor}" rx="2" />
          <text x="298" y="175" fill="${textDark}" font-size="10">Active Clean Cells</text>
          <rect x="430" y="165" width="12" height="12" fill="${coralColor}" rx="2" />
          <text x="448" y="175" fill="${textDark}" font-size="10">Missing Cells Masked</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 3: Linear Correlation Matrix (4x4 Grid Heatmap)
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-correlation-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <!-- Cell Row 1 -->
          <rect x="60" y="20" width="50" height="40" fill="${coralColor}" fill-opacity="0.8" rx="2" />
          <text x="85" y="44" fill="#FFF" font-size="10" text-anchor="middle">1.00</text>
          <rect x="120" y="20" width="50" height="40" fill="${coralColor}" fill-opacity="0.4" rx="2" />
          <text x="145" y="44" fill="#FFF" font-size="10" text-anchor="middle">0.48</text>
          <rect x="180" y="20" width="50" height="40" fill="${mintColor}" fill-opacity="0.1" rx="2" />
          <text x="205" y="44" fill="${textDark}" font-size="10" text-anchor="middle">-0.08</text>
          <rect x="240" y="20" width="50" height="40" fill="${coralColor}" fill-opacity="0.5" rx="2" />
          <text x="265" y="44" fill="#FFF" font-size="10" text-anchor="middle">0.62</text>
  
          <!-- Cell Row 2 -->
          <rect x="60" y="70" width="50" height="40" fill="${coralColor}" fill-opacity="0.4" rx="2" />
          <text x="85" y="94" fill="#FFF" font-size="10" text-anchor="middle">0.48</text>
          <rect x="120" y="70" width="50" height="40" fill="${coralColor}" fill-opacity="0.8" rx="2" />
          <text x="145" y="94" fill="#FFF" font-size="10" text-anchor="middle">1.00</text>
          <rect x="180" y="70" width="50" height="40" fill="${mintColor}" fill-opacity="0.2" rx="2" />
          <text x="205" y="94" fill="${textDark}" font-size="10" text-anchor="middle">-0.15</text>
          <rect x="240" y="70" width="50" height="40" fill="${coralColor}" fill-opacity="0.2" rx="2" />
          <text x="265" y="94" fill="${textDark}" font-size="10" text-anchor="middle">0.24</text>
  
          <!-- Cell Row 3 -->
          <rect x="60" y="120" width="50" height="40" fill="${mintColor}" fill-opacity="0.1" rx="2" />
          <text x="85" y="144" fill="${textDark}" font-size="10" text-anchor="middle">-0.08</text>
          <rect x="120" y="120" width="50" height="40" fill="${mintColor}" fill-opacity="0.2" rx="2" />
          <text x="145" y="144" fill="${textDark}" font-size="10" text-anchor="middle">-0.15</text>
          <rect x="180" y="120" width="50" height="40" fill="${coralColor}" fill-opacity="0.8" rx="2" />
          <text x="205" y="144" fill="#FFF" font-size="10" text-anchor="middle">1.00</text>
          <rect x="240" y="120" width="50" height="40" fill="${mintColor}" fill-opacity="0.3" rx="2" />
          <text x="265" y="144" fill="${textDark}" font-size="10" text-anchor="middle">-0.35</text>
  
          <!-- Cell Row 4 -->
          <rect x="60" y="170" width="50" height="40" fill="${coralColor}" fill-opacity="0.5" rx="2" />
          <text x="85" y="194" fill="#FFF" font-size="10" text-anchor="middle">0.62</text>
          <rect x="120" y="170" width="50" height="40" fill="${coralColor}" fill-opacity="0.2" rx="2" />
          <text x="145" y="194" fill="${textDark}" font-size="10" text-anchor="middle">0.24</text>
          <rect x="180" y="170" width="50" height="40" fill="${mintColor}" fill-opacity="0.3" rx="2" />
          <text x="205" y="194" fill="${textDark}" font-size="10" text-anchor="middle">-0.35</text>
          <rect x="240" y="170" width="50" height="40" fill="${coralColor}" fill-opacity="0.8" rx="2" />
          <text x="265" y="194" fill="#FFF" font-size="10" text-anchor="middle">1.00</text>
  
          <!-- Axis Labels -->
          <text x="50" y="44" fill="${textDark}" font-size="9" text-anchor="end">Age</text>
          <text x="50" y="94" fill="${textDark}" font-size="9" text-anchor="end">Income</text>
          <text x="50" y="144" fill="${textDark}" font-size="9" text-anchor="end">Size</text>
          <text x="50" y="194" fill="${textDark}" font-size="9" text-anchor="end">Poverty</text>
  
          <text x="85" y="222" fill="${textDark}" font-size="9" text-anchor="middle">Age</text>
          <text x="145" y="222" fill="${textDark}" font-size="9" text-anchor="middle">Income</text>
          <text x="205" y="222" fill="${textDark}" font-size="9" text-anchor="middle">Size</text>
          <text x="265" y="222" fill="${textDark}" font-size="9" text-anchor="middle">Poverty</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 4: Dynamic Model Allocation (Horizontal Bar representation)
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-allocation-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <!-- Axis Boundary -->
          <line x1="80" y1="200" x2="300" y2="200" stroke="${gridLineColor}" stroke-width="1" />
          <line x1="80" y1="20" x2="80" y2="200" stroke="${gridLineColor}" stroke-width="1" />
  
          <!-- Bar Row 1: LightGBM -->
          <text x="70" y="55" fill="${textDark}" font-size="9" text-anchor="end">LightGBM</text>
          <rect x="80" y="40" width="160" height="22" fill="${mintColor}" rx="2" />
          <text x="248" y="54" fill="${textDark}" font-size="9" font-weight="600">51.2%</text>
  
          <!-- Bar Row 2: CatBoost -->
          <text x="70" y="95" fill="${textDark}" font-size="9" text-anchor="end">CatBoost</text>
          <rect x="80" y="80" width="85" height="22" fill="${coralColor}" rx="2" />
          <text x="173" y="94" fill="${textDark}" font-size="9" font-weight="600">27.3%</text>
  
          <!-- Bar Row 3: ExtraTrees -->
          <text x="70" y="135" fill="${textDark}" font-size="9" text-anchor="end">ExtraTrees</text>
          <rect x="80" y="120" width="45" height="22" fill="${grayMuted}" rx="2" />
          <text x="133" y="134" fill="${textDark}" font-size="9" font-weight="600">14.1%</text>
  
          <!-- Bar Row 4: KNN/Ridge -->
          <text x="70" y="175" fill="${textDark}" font-size="9" text-anchor="end">Bayesian/KNN</text>
          <rect x="80" y="160" width="24" height="22" fill="#E5E7EB" rx="2" />
          <text x="112" y="174" fill="${textDark}" font-size="9" font-weight="600">7.4%</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 5: Multiclass Poverty distributions by Household size
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-poverty-size-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <line x1="40" y1="180" x2="300" y2="180" stroke="${gridLineColor}" stroke-width="1" />
          <line x1="40" y1="20" x2="40" y2="180" stroke="${gridLineColor}" stroke-width="1" />
  
          <!-- Grid Lines -->
          <line x1="40" y1="130" x2="300" y2="130" stroke="${gridLineColor}" stroke-width="0.5" stroke-dasharray="2" />
          <line x1="40" y1="80" x2="300" y2="80" stroke="${gridLineColor}" stroke-width="0.5" stroke-dasharray="2" />
  
          <!-- Double Column 1 (Size: 1-3) -->
          <rect x="60" y="90" width="18" height="90" fill="${grayMuted}" rx="1" />
          <rect x="80" y="145" width="18" height="35" fill="${coralColor}" rx="1" />
  
          <!-- Double Column 2 (Size: 4-6) -->
          <rect x="130" y="45" width="18" height="135" fill="${grayMuted}" rx="1" />
          <rect x="150" y="110" width="18" height="70" fill="${coralColor}" rx="1" />
  
          <!-- Double Column 3 (Size: 7+) -->
          <rect x="200" y="70" width="18" height="110" fill="${grayMuted}" rx="1" />
          <rect x="220" y="55" width="18" height="125" fill="${coralColor}" rx="1" />
  
          <!-- Tick Labels -->
          <text x="79" y="195" fill="${textDark}" font-size="9" text-anchor="middle">Small (1-3)</text>
          <text x="149" y="195" fill="${textDark}" font-size="9" text-anchor="middle">Medium (4-6)</text>
          <text x="219" y="195" fill="${textDark}" font-size="9" text-anchor="middle">Large (7+)</text>
  
          <!-- Legends -->
          <rect x="60" y="210" width="10" height="10" fill="${grayMuted}" rx="1" />
          <text x="75" y="219" fill="${textDark}" font-size="9">Non-Poor</text>
          <rect x="160" y="210" width="10" height="10" fill="${coralColor}" rx="1" />
          <text x="175" y="219" fill="${textDark}" font-size="9">Poor (FGT0 Mapped)</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 6: Respondent Age Boxplot (Continuous Spread)
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-age-box-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <!-- Center Axis lines -->
          <line x1="50" y1="180" x2="280" y2="180" stroke="${gridLineColor}" />
          <line x1="50" y1="20" x2="50" y2="180" stroke="${gridLineColor}" />
  
          <!-- Boxplot 1: Urban Area (Horizontal alignment representation) -->
          <text x="45" y="65" fill="${textDark}" font-size="9" text-anchor="end">Urbain</text>
          <line x1="70" y1="65" x2="240" y2="65" stroke="${textDark}" stroke-width="1.5" />
          <rect x="110" y="50" width="80" height="30" fill="${mintColor}" fill-opacity="0.8" stroke="${textDark}" stroke-width="1.5" rx="1" />
          <line x1="150" y1="50" x2="150" y2="80" stroke="${textDark}" stroke-width="2.5" />
  
          <!-- Boxplot 2: Rural Area -->
          <text x="45" y="135" fill="${textDark}" font-size="9" text-anchor="end">Rural</text>
          <line x1="60" y1="135" x2="260" y2="135" stroke="${textDark}" stroke-width="1.5" />
          <rect x="95" y="120" width="105" height="30" fill="${coralColor}" fill-opacity="0.8" stroke="${textDark}" stroke-width="1.5" rx="1" />
          <line x1="140" y1="120" x2="140" y2="150" stroke="${textDark}" stroke-width="2.5" />
  
          <!-- Tick markings -->
          <text x="70" y="195" fill="${textDark}" font-size="9" text-anchor="middle">5</text>
          <text x="140" y="195" fill="${textDark}" font-size="9" text-anchor="middle">45 (Median)</text>
          <text x="240" y="195" fill="${textDark}" font-size="9" text-anchor="middle">85 (Max)</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 7: Survey Weights Distribution Density
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-weight-density-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <line x1="40" y1="180" x2="300" y2="180" stroke="${gridLineColor}" />
          <line x1="40" y1="20" x2="40" y2="180" stroke="${gridLineColor}" />
  
          <!-- Weighted density curve path -->
          <path d="M 40 178 Q 80 140 120 40 T 180 120 T 240 170 T 300 179" fill="none" stroke="${mintColor}" stroke-width="2" />
          <path d="M 40 178 Q 80 140 120 40 T 180 120 T 240 170 T 300 179 L 300 180 L 40 180 Z" fill="${mintColor}" fill-opacity="0.1" />
  
          <text x="120" y="25" fill="${textDark}" font-size="9" text-anchor="middle">Core Sample Concentration (1142.50)</text>
          <text x="120" y="195" fill="${textDark}" font-size="9" text-anchor="middle">Weight Values (coef_indiv)</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 8: MNAR Cluster Map (Dimensional Scatter representation)
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-mnar-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <!-- Box Boundary -->
          <rect x="30" y="20" width="260" height="160" fill="none" stroke="${gridLineColor}" stroke-width="1" />
  
          <!-- Cluster 1 (Present values) -->
          <circle cx="80" cy="60" r="4" fill="${mintColor}" />
          <circle cx="100" cy="70" r="4" fill="${mintColor}" />
          <circle cx="70" cy="85" r="4" fill="${mintColor}" />
          <circle cx="95" cy="50" r="4" fill="${mintColor}" />
          <circle cx="110" cy="80" r="4" fill="${mintColor}" />
  
          <!-- Cluster 2 (Missing values MNAR alignment) -->
          <circle cx="210" cy="130" r="4" fill="${coralColor}" />
          <circle cx="230" cy="140" r="4" fill="${coralColor}" />
          <circle cx="200" cy="115" r="4" fill="${coralColor}" />
          <circle cx="240" cy="120" r="4" fill="${coralColor}" />
          <circle cx="225" cy="105" r="4" fill="${coralColor}" />
  
          <text x="90" y="105" fill="${textDark}" font-size="8" text-anchor="middle">Cluster Alpha (Clean)</text>
          <text x="225" y="95" fill="${textDark}" font-size="8" text-anchor="middle">Cluster Beta (MNAR)</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 9: Column Completeness Ratios
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-completeness-bars-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <line x1="80" y1="180" x2="300" y2="180" stroke="${gridLineColor}" />
          <line x1="80" y1="20" x2="80" y2="180" stroke="${gridLineColor}" />
  
          <!-- Variable Row 1 -->
          <text x="70" y="55" fill="${textDark}" font-size="9" text-anchor="end">S04Q03_AGE</text>
          <rect x="80" y="42" width="200" height="16" fill="${mintColor}" rx="1" />
  
          <!-- Variable Row 2 -->
          <text x="70" y="95" fill="${textDark}" font-size="9" text-anchor="end">REVENU_MOL</text>
          <rect x="80" y="82" width="155" height="16" fill="${coralColor}" rx="1" />
  
          <!-- Variable Row 3 -->
          <text x="70" y="135" fill="${textDark}" font-size="9" text-anchor="end">EAU_MODE</text>
          <rect x="80" y="122" width="170" height="16" fill="${mintColor}" rx="1" />
  
          <!-- Scale markings -->
          <line x1="180" y1="180" x2="180" y2="185" stroke="${gridLineColor}" />
          <text x="180" y="198" fill="${textDark}" font-size="8" text-anchor="middle">50% Complete</text>
          <line x1="280" y1="180" x2="280" y2="185" stroke="${gridLineColor}" />
          <text x="280" y="198" fill="${textDark}" font-size="8" text-anchor="middle">100%</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 10: Cumulative PCA Variance
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-pca-variance-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <line x1="40" y1="180" x2="280" y2="180" stroke="${gridLineColor}" />
          <line x1="40" y1="20" x2="40" y2="180" stroke="${gridLineColor}" />
  
          <!-- Cumulative variance curve path -->
          <path d="M 40 180 L 80 130 L 120 90 L 160 65 L 200 48 L 240 38 L 280 32" fill="none" stroke="${coralColor}" stroke-width="2" />
          <circle cx="80" cy="130" r="3" fill="${coralColor}" />
          <circle cx="120" cy="90" r="3" fill="${coralColor}" />
          <circle cx="160" cy="65" r="3" fill="${coralColor}" />
          <circle cx="200" cy="48" r="3" fill="${coralColor}" />
          <circle cx="240" cy="38" r="3" fill="${coralColor}" />
          <circle cx="280" cy="32" r="3" fill="${coralColor}" />
  
          <text x="230" y="75" fill="${textDark}" font-size="9">Total Components</text>
          <text x="120" y="195" fill="${textDark}" font-size="9" text-anchor="middle">Number of Eigenvalues</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 11: Mutual Information Rank
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-mutual-info-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <line x1="90" y1="180" x2="300" y2="180" stroke="${gridLineColor}" />
          <line x1="90" y1="20" x2="90" y2="180" stroke="${gridLineColor}" />
  
          <!-- Metric 1 -->
          <text x="80" y="55" fill="${textDark}" font-size="9" text-anchor="end">Household Size</text>
          <rect x="90" y="42" width="180" height="16" fill="${coralColor}" rx="1" />
  
          <!-- Metric 2 -->
          <text x="80" y="95" fill="${textDark}" font-size="9" text-anchor="end">Education Level</text>
          <rect x="90" y="82" width="135" height="16" fill="${mintColor}" rx="1" />
  
          <!-- Metric 3 -->
          <text x="80" y="135" fill="${textDark}" font-size="9" text-anchor="end">Geographic Reg</text>
          <rect x="90" y="122" width="95" height="16" fill="${grayMuted}" rx="1" />
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 12: LGBM Probability Margins (Distribution)
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-probability-margins-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <line x1="40" y1="180" x2="280" y2="180" stroke="${gridLineColor}" />
          <line x1="40" y1="20" x2="40" y2="180" stroke="${gridLineColor}" />
  
          <!-- Double peaks histogram path -->
          <path d="M 40 180 Q 70 110 100 80 T 140 170 Q 180 140 210 50 T 280 180" fill="none" stroke="${mintColor}" stroke-width="2" />
          <path d="M 40 180 Q 70 110 100 80 T 140 170 Q 180 140 210 50 T 280 180 Z" fill="${mintColor}" fill-opacity="0.1" />
  
          <!-- Decision threshold limit -->
          <line x1="150" y1="20" x2="150" y2="180" stroke="${coralColor}" stroke-dasharray="3" stroke-width="1.5" />
          <text x="156" y="35" fill="${coralColor}" font-size="8">Classification Threshold (0.50)</text>
  
          <text x="150" y="195" fill="${textDark}" font-size="9" text-anchor="middle">Calculated Probability Score</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 13: Household Modernity Index
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-modernity-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <line x1="40" y1="180" x2="280" y2="180" stroke="${gridLineColor}" />
          <line x1="40" y1="20" x2="40" y2="180" stroke="${gridLineColor}" />
  
          <!-- Bar Series 1 -->
          <rect x="70" y="60" width="35" height="120" fill="${mintColor}" rx="1" />
          <text x="87" y="52" fill="${textDark}" font-size="9" font-weight="600" text-anchor="middle">84.5%</text>
  
          <!-- Bar Series 2 -->
          <rect x="170" y="115" width="35" height="65" fill="${coralColor}" rx="1" />
          <text x="187" y="107" fill="${textDark}" font-size="9" font-weight="600" text-anchor="middle">38.2%</text>
  
          <text x="87" y="195" fill="${textDark}" font-size="9" text-anchor="middle">Urban Zones</text>
          <text x="187" y="195" fill="${textDark}" font-size="9" text-anchor="middle">Rural Zones</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 14: Pairwise Domain Drift (Scatter representation)
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-drift-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 320 240" width="100%" height="100%">
          <rect x="30" y="20" width="260" height="160" fill="none" stroke="${gridLineColor}" />
  
          <!-- Dots group A (ENCDM Survey) -->
          <circle cx="60" cy="120" r="3" fill="${mintColor}" fill-opacity="0.7" />
          <circle cx="80" cy="110" r="3" fill="${mintColor}" fill-opacity="0.7" />
          <circle cx="110" cy="90" r="3" fill="${mintColor}" fill-opacity="0.7" />
          <circle cx="140" cy="70" r="3" fill="${mintColor}" fill-opacity="0.7" />
  
          <!-- Dots group B (Census predictions showing identical coverage) -->
          <circle cx="63" cy="123" r="3" fill="${coralColor}" fill-opacity="0.7" />
          <circle cx="83" cy="112" r="3" fill="${coralColor}" fill-opacity="0.7" />
          <circle cx="114" cy="94" r="3" fill="${coralColor}" fill-opacity="0.7" />
          <circle cx="145" cy="72" r="3" fill="${coralColor}" fill-opacity="0.7" />
  
          <text x="160" y="195" fill="${textDark}" font-size="9" text-anchor="middle">Joint Distribution (No Covariate Shift)</text>
        </svg>
      `;
    })();
  
    // --------------------------------------------------------------------------
    // PLOT 15: Imputation Confidence Intervals
    // --------------------------------------------------------------------------
    (function() {
      const container = document.getElementById("plot-confidence-container");
      if (!container) return;
      container.innerHTML = `
        <svg viewBox="0 0 800 240" width="100%" height="100%">
          <line x1="50" y1="200" x2="750" y2="200" stroke="${gridLineColor}" />
          <line x1="50" y1="20" x2="50" y2="200" stroke="${gridLineColor}" />
  
          <!-- Confidence Area polygon path -->
          <polygon points="50,140 150,110 300,90 450,115 600,65 750,55 750,105 600,115 450,155 300,150 150,170 50,190" fill="${coralColor}" fill-opacity="0.1" />
  
          <!-- Core target value path -->
          <path d="M 50 165 L 150 140 L 300 120 L 450 135 L 600 90 L 750 80" fill="none" stroke="${coralColor}" stroke-width="2.5" />
  
          <circle cx="150" cy="140" r="4" fill="${coralColor}" />
          <circle cx="300" cy="120" r="4" fill="${coralColor}" />
          <circle cx="450" cy="135" r="4" fill="${coralColor}" />
          <circle cx="600" cy="90" r="4" fill="${coralColor}" />
  
          <text x="400" y="220" fill="${textDark}" font-size="10" text-anchor="middle">Data Ingestion Index Steps</text>
          <text x="610" y="75" fill="${textDark}" font-size="9">Upper Confidence Bound (95% CI)</text>
        </svg>
      `;
    })();
  }