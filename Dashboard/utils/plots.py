import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from .palette import PALETTE
from .theme import mpl_style

FIGSIZE = (10, 4.2)

EXPLORER_PLOTS = [
    {
        'id': 'poverty_region',
        'title': 'Poverty Rate by Region',
        'desc': 'Survey-weighted share of individuals below the national poverty line across Morocco\'s 12 regions. Highlights spatial inequality in living standards.',
    },
    {
        'id': 'vulnerability_region',
        'title': 'Vulnerability Rate by Region',
        'desc': 'Individuals at risk of falling into poverty, weighted by coef_indiv. Complements the poverty map with near-poor populations.',
    },
    {
        'id': 'poverty_milieu',
        'title': 'Poverty Split: Urban vs Rural',
        'desc': 'National composition of poor individuals by environment. Rural areas typically concentrate a disproportionate share of poverty.',
    },
    {
        'id': 'education_poverty',
        'title': 'Education & Poverty',
        'desc': 'Weighted counts of poor vs non-poor individuals by education level. Strong gradient between schooling and poverty status.',
    },
    {
        'id': 'age_profile',
        'title': 'Age Profile: Poor vs Non-Poor',
        'desc': 'Weighted age distributions for poor and non-poor individuals. Reveals which life stages are most exposed to poverty.',
    },
    {
        'id': 'household_size',
        'title': 'Household Size Distribution',
        'desc': 'How household size differs between poor and non-poor individuals. Larger households often face tighter per-capita resources.',
    },
    {
        'id': 'gender_gap',
        'title': 'Poverty by Gender',
        'desc': 'Weighted poverty rates for men and women nationally. A compact view of gender disparities in economic vulnerability.',
    },
    {
        'id': 'quintile_milieu',
        'title': 'Income Quintile by Environment',
        'desc': 'Distribution of individuals across income quintiles in urban and rural areas. Shows how welfare varies by milieu.',
    },
    {
        'id': 'rgph_electricity',
        'title': 'Electricity Access (RGPH)',
        'desc': 'Share of households without electricity by region, weighted by PDS. A basic modernity indicator at census scale.',
    },
    {
        'id': 'rgph_internet',
        'title': 'Internet Access by Milieu',
        'desc': 'Household internet ownership in urban vs rural areas from RGPH. Digital divide visible at national level.',
    },
    {
        'id': 'rgph_housing',
        'title': 'Housing Type Distribution',
        'desc': 'Top housing categories in the census, weighted by PDS. Structural housing conditions underpin amenity deprivation.',
    },
    {
        'id': 'amenity_heatmap',
        'title': 'Amenity Deprivation Heatmap',
        'desc': 'Regional rates of missing key amenities (electricity, water, internet). Rows are regions, columns are deprivation flags.',
    },
    {
        'id': 'profession_poverty',
        'title': 'Poverty by Profession',
        'desc': 'Weighted poverty rates across professional situations. Shows which labour market positions correlate with deprivation.',
    },
    {
        'id': 'decile_gradient',
        'title': 'Poverty Along the Income Ladder',
        'desc': 'Poverty rate by income decile. A steep gradient confirms that welfare labels align with the consumption distribution.',
    },
    {
        'id': 'rgph_water',
        'title': 'Water Access Gaps (RGPH)',
        'desc': 'Share of households without modern water access by region. Complements electricity and internet deprivation maps.',
    },
    {
        'id': 'guelmim_spotlight',
        'title': 'Guelmim-Oued Noun Spotlight',
        'desc': 'Side-by-side poverty and vulnerability rates for the internship region against the national average.',
    },
]

def _style(theme):
    plt.rcParams.update(mpl_style(theme))

def _title(ax, text, theme):
    ax.set_title(text, loc = 'left', fontsize = 13, fontweight = 600, color = PALETTE[theme]['text'], pad = 12)

def _render_plot(plotid, theme):
    from .data import load_data
    encdm, rgph, mapencdm, maprgph, _ = load_data()
    p = PALETTE[theme]
    _style(theme)
    fig, ax = plt.subplots(figsize = FIGSIZE)

    if plotid == 'poverty_region' :
        rates = encdm.groupby('Région_12').apply(lambda g : g.loc[g['Pauvre'] == 1, 'coef_indiv'].sum() / g['coef_indiv'].sum() * 100, include_groups = False)
        labels = [mapencdm['Région_12'].get(i, str(i))[:18] for i in rates.index]
        order = rates.sort_values().index
        vals = rates.loc[order]
        labs = [mapencdm['Région_12'].get(i, str(i))[:18] for i in order]
        colors = [p['poverty'] if v > rates.median() else p['accent'] for v in vals]
        ax.barh(labs, vals, color = colors, height = 0.65)
        ax.set_xlabel('Poverty rate (%)', color = p['muted'])
        _title(ax, 'Poverty Rate by Region', theme)

    elif plotid == 'vulnerability_region' :
        rates = encdm.groupby('Région_12').apply(lambda g : g.loc[g['Vulnérable'] == 1, 'coef_indiv'].sum() / g['coef_indiv'].sum() * 100, include_groups = False)
        order = rates.sort_values().index
        vals = rates.loc[order]
        labs = [mapencdm['Région_12'].get(i, str(i))[:18] for i in order]
        ax.barh(labs, vals, color = p['vulnerable'], height = 0.65, alpha = 0.85)
        ax.set_xlabel('Vulnerability rate (%)', color = p['muted'])
        _title(ax, 'Vulnerability Rate by Region', theme)

    elif plotid == 'poverty_milieu' :
        counts = encdm.groupby(['Milieu', 'Pauvre'])['coef_indiv'].sum().unstack(fill_value = 0)
        labels = [mapencdm['Milieu'].get(i, str(i)) for i in counts.index]
        poor = counts.get(1, pd.Series(0, index = counts.index))
        nonpoor = counts.get(0, pd.Series(0, index = counts.index))
        x = np.arange(len(labels))
        ax.bar(x, nonpoor, label = 'Non-poor', color = p['accent'], width = 0.55)
        ax.bar(x, poor, bottom = nonpoor, label = 'Poor', color = p['poverty'], width = 0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(frameon = False, labelcolor = p['text'])
        ax.set_ylabel('Weighted individuals', color = p['muted'])
        _title(ax, 'Poverty Split: Urban vs Rural', theme)

    elif plotid == 'education_poverty' :
        pivot = encdm.groupby(['Niveau_scolaire_agreg_CM', 'Pauvre'])['coef_indiv'].sum().unstack(fill_value = 0)
        labels = [mapencdm['Niveau_scolaire_agreg_CM'].get(i, str(i))[:14] for i in pivot.index]
        x = np.arange(len(labels))
        w = 0.38
        ax.bar(x - w/2, pivot.get(0, 0), w, label = 'Non-poor', color = p['accent'])
        ax.bar(x + w/2, pivot.get(1, 0), w, label = 'Poor', color = p['poverty'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation = 35, ha = 'right', fontsize = 8)
        ax.legend(frameon = False, labelcolor = p['text'])
        _title(ax, 'Education & Poverty', theme)

    elif plotid == 'age_profile' :
        for label, mask, color in [('Non-poor', encdm['Pauvre'] == 0, p['accent']), ('Poor', encdm['Pauvre'] == 1, p['poverty'])] :
            ages, weights = encdm.loc[mask, 'Age_CM'], encdm.loc[mask, 'coef_indiv']
            ax.hist(ages, bins = 25, weights = weights, alpha = 0.55, label = label, color = color, density = True)
        ax.legend(frameon = False, labelcolor = p['text'])
        ax.set_xlabel('Age', color = p['muted'])
        _title(ax, 'Age Profile: Poor vs Non-Poor', theme)

    elif plotid == 'household_size' :
        for label, mask, color in [('Non-poor', encdm['Pauvre'] == 0, p['accent']), ('Poor', encdm['Pauvre'] == 1, p['poverty'])] :
            vals, weights = encdm.loc[mask, 'Taille_ménage'], encdm.loc[mask, 'coef_indiv']
            ax.hist(vals, bins = range(1, 14), weights = weights, alpha = 0.55, label = label, color = color, density = True)
        ax.legend(frameon = False, labelcolor = p['text'])
        ax.set_xlabel('Household size', color = p['muted'])
        _title(ax, 'Household Size Distribution', theme)

    elif plotid == 'gender_gap' :
        rates = encdm.groupby('Sexe_CM').apply(lambda g : g.loc[g['Pauvre'] == 1, 'coef_indiv'].sum() / g['coef_indiv'].sum() * 100, include_groups = False)
        labels = [mapencdm['Sexe_CM'].get(i, str(i)) for i in rates.index]
        ax.bar(labels, rates.values, color = [p['accent'], p['poverty']][:len(rates)], width = 0.5)
        ax.set_ylabel('Poverty rate (%)', color = p['muted'])
        for i, v in enumerate(rates.values) : ax.text(i, v + 0.15, f'{v:.1f}%', ha = 'center', color = p['text'], fontsize = 10)
        _title(ax, 'Poverty by Gender', theme)

    elif plotid == 'quintile_milieu' :
        pivot = encdm.groupby(['Milieu', 'Quintiles'])['coef_indiv'].sum().unstack(fill_value = 0)
        pivot = pivot.div(pivot.sum(axis = 1), axis = 0) * 100
        x = np.arange(pivot.shape[1])
        for i, (milieu, row) in enumerate(pivot.iterrows()) :
            ax.plot(x, row.values, marker = 'o', linewidth = 2, label = mapencdm['Milieu'].get(milieu, str(milieu)), color = p['chart'][i % len(p['chart'])])
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        ax.set_ylabel('Share (%)', color = p['muted'])
        ax.set_xlabel('Income quintile', color = p['muted'])
        ax.legend(frameon = False, labelcolor = p['text'])
        _title(ax, 'Income Quintile by Environment', theme)

    elif plotid == 'rgph_electricity' :
        rates = rgph.groupby('REG').apply(lambda g : g.loc[g['ELEC'] == 0, 'PDS'].sum() / g['PDS'].sum() * 100, include_groups = False)
        order = rates.sort_values(ascending = False).head(12).index
        labs = [maprgph['REG'].get(i, str(i))[:16] for i in order]
        ax.barh(labs, rates.loc[order], color = p['accent2'], height = 0.65)
        ax.set_xlabel('Without electricity (%)', color = p['muted'])
        ax.invert_yaxis()
        _title(ax, 'Electricity Access (RGPH)', theme)

    elif plotid == 'rgph_internet' :
        rates = rgph.groupby('MIL').apply(lambda g : g.loc[g['NET'] == 1, 'PDS'].sum() / g['PDS'].sum() * 100, include_groups = False)
        labels = [maprgph['MIL'].get(i, str(i)) for i in rates.index]
        ax.bar(labels, rates.values, color = [p['accent'], p['accent2']][:len(rates)], width = 0.45)
        ax.set_ylabel('With internet (%)', color = p['muted'])
        for i, v in enumerate(rates.values) : ax.text(i, v + 1, f'{v:.0f}%', ha = 'center', color = p['text'])
        _title(ax, 'Internet Access by Milieu', theme)

    elif plotid == 'rgph_housing' :
        top = rgph.groupby('TYPE.LOG')['PDS'].sum().sort_values(ascending = False).head(8)
        labs = [maprgph['TYPE.LOG'].get(i, str(i))[:16] for i in top.index]
        ax.barh(labs[::-1], top.values[::-1], color = p['chart'][2], height = 0.65)
        ax.set_xlabel('Weighted households', color = p['muted'])
        _title(ax, 'Housing Type Distribution', theme)

    elif plotid == 'amenity_heatmap' :
        cols = ['ELEC', 'NET', 'EAU.MODE']
        regions = sorted(rgph['REG'].unique())[:12]
        matrix = np.array([
            [rgph.loc[(rgph['REG'] == r) & (rgph[c] == 0), 'PDS'].sum() / rgph.loc[rgph['REG'] == r, 'PDS'].sum() * 100 for c in cols]
            for r in regions
        ])
        im = ax.imshow(matrix, aspect = 'auto', cmap = 'YlOrRd')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(['No electricity', 'No internet', 'No water'], fontsize = 8)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels([maprgph['REG'].get(r, str(r))[:14] for r in regions], fontsize = 8)
        _title(ax, 'Amenity Deprivation Heatmap', theme)
        fig.colorbar(im, ax = ax, fraction = 0.03, pad = 0.02, label = '%')

    elif plotid == 'profession_poverty' :
        rates = encdm.groupby('Situation_profession_agreg_CM').apply(
            lambda g : g.loc[g['Pauvre'] == 1, 'coef_indiv'].sum() / g['coef_indiv'].sum() * 100, include_groups = False,
        ).sort_values(ascending = True)
        labs = [mapencdm['Situation_profession_agreg_CM'].get(i, str(i))[:16] for i in rates.index]
        ax.barh(labs, rates.values, color = p['chart'][3], height = 0.65)
        ax.set_xlabel('Poverty rate (%)', color = p['muted'])
        _title(ax, 'Poverty by Profession', theme)

    elif plotid == 'decile_gradient' :
        rates = encdm.groupby('Deciles').apply(
            lambda g : g.loc[g['Pauvre'] == 1, 'coef_indiv'].sum() / g['coef_indiv'].sum() * 100, include_groups = False,
        )
        ax.plot(rates.index, rates.values, marker = 'o', linewidth = 2.2, color = p['poverty'])
        ax.fill_between(rates.index, rates.values, alpha = 0.15, color = p['poverty'])
        ax.set_xlabel('Income decile', color = p['muted'])
        ax.set_ylabel('Poverty rate (%)', color = p['muted'])
        _title(ax, 'Poverty Along the Income Ladder', theme)

    elif plotid == 'rgph_water' :
        col = 'EAU.MODE' if 'EAU.MODE' in rgph.columns else 'ELEC'
        rates = rgph.groupby('REG').apply(
            lambda g : g.loc[g[col] == 0, 'PDS'].sum() / g['PDS'].sum() * 100, include_groups = False,
        )
        order = rates.sort_values(ascending = False).head(10).index
        labs = [maprgph['REG'].get(i, str(i))[:16] for i in order]
        ax.barh(labs, rates.loc[order], color = p['chart'][5], height = 0.65)
        ax.set_xlabel('Without modern water (%)', color = p['muted'])
        ax.invert_yaxis()
        _title(ax, 'Water Access Gaps (RGPH)', theme)

    elif plotid == 'guelmim_spotlight' :
        guelmim = 2
        natpoor = encdm.loc[encdm['Pauvre'] == 1, 'coef_indiv'].sum() / encdm['coef_indiv'].sum() * 100
        natvuln = encdm.loc[encdm['Vulnérable'] == 1, 'coef_indiv'].sum() / encdm['coef_indiv'].sum() * 100
        g = encdm.loc[encdm['Région_12'] == guelmim]
        gpoor = g.loc[g['Pauvre'] == 1, 'coef_indiv'].sum() / g['coef_indiv'].sum() * 100
        gvuln = g.loc[g['Vulnérable'] == 1, 'coef_indiv'].sum() / g['coef_indiv'].sum() * 100
        labels = ['Poverty', 'Vulnerability']
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, [natpoor, natvuln], w, label = 'National', color = p['accent'], alpha = 0.85)
        ax.bar(x + w/2, [gpoor, gvuln], w, label = 'Guelmim-Oued Noun', color = p['accent2'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Rate (%)', color = p['muted'])
        ax.legend(frameon = False, labelcolor = p['text'])
        _title(ax, 'Guelmim-Oued Noun Spotlight', theme)

    ax.grid(True, axis = 'y' if plotid not in {'amenity_heatmap', 'rgph_housing', 'poverty_region', 'vulnerability_region', 'rgph_water', 'profession_poverty'} else 'x', alpha = 0.25, linestyle = '--')
    plt.tight_layout()
    return fig

def _fig_to_bytes(fig, theme):
    buf = io.BytesIO()
    fig.savefig(buf, format = 'png', dpi = 130, bbox_inches = 'tight', facecolor = PALETTE[theme]['plot_bg'])
    plt.close(fig)
    return buf.getvalue()

@st.cache_data(show_spinner = False)
def build_plot(plotid, theme):
    return _fig_to_bytes(_render_plot(plotid, theme), theme)

@st.cache_data(show_spinner = False)
def overview_raw_nulls_plot(theme):
    from .data import raw_null_stats
    stats = raw_null_stats()
    p = PALETTE[theme]
    _style(theme)
    fig, axes = plt.subplots(1, 2, figsize = (10, 4.2))

    datasets = ['ENCDM', 'RGPH']
    totals = [stats['encdm_nulls'], stats['rgph_nulls']]
    axes[0].bar(datasets, totals, color = [p['accent'], p['accent2']], width = 0.45)
    axes[0].set_ylabel('Total null cells', color = p['muted'])
    _title(axes[0], 'Missing Values in Raw .sav Files', theme)

    encdmtop = stats['encdm_top']
    if len(encdmtop) :
        axes[1].barh(encdmtop.index[::-1], encdmtop.values[::-1], color = p['poverty'], height = 0.65)
    axes[1].set_xlabel('Null count', color = p['muted'])
    _title(axes[1], 'Top ENCDM Columns with Nulls', theme)

    plt.tight_layout()
    return _fig_to_bytes(fig, theme)

@st.cache_data(show_spinner = False)
def hypernetwork_loss_plot(theme):
    from .data import load_hypernetwork
    checkpoint = load_hypernetwork()
    p = PALETTE[theme]
    _style(theme)
    fig, ax = plt.subplots(figsize = FIGSIZE)
    if checkpoint and checkpoint.get('history') :
        history = checkpoint['history']
        ax.plot(range(1, len(history) + 1), history, marker = 'o', linewidth = 2, color = p['vulnerable'])
        ax.set_xlabel('Epoch', color = p['muted'])
        ax.set_ylabel('Training loss', color = p['muted'])
        _title(ax, 'Hypernetwork Training Loss (from Hypernet.pt)', theme)
    else :
        ax.text(0.5, 0.5, 'Hypernet.pt not found', ha = 'center', va = 'center', color = p['muted'], transform = ax.transAxes)
    ax.grid(True, alpha = 0.25, linestyle = '--')
    plt.tight_layout()
    return _fig_to_bytes(fig, theme)

@st.cache_data(show_spinner = False)
def feature_importance_plot(theme):
    import joblib
    from .data import PROJECTROOT, load_data
    encdm, _, _, _, _ = load_data()
    bundle = joblib.load(PROJECTROOT / 'Models' / 'Classifier' / 'ENCDM_LGBM_Pauvre.joblib')
    model, features = bundle['model'], bundle['features']
    imp = pd.Series(model.feature_importances_, index = features).sort_values()
    p = PALETTE[theme]
    _style(theme)
    fig, ax = plt.subplots(figsize = FIGSIZE)
    ax.barh(imp.index, imp.values, color = p['accent'], height = 0.6)
    _title(ax, 'LightGBM Feature Importance (Pauvre)', theme)
    ax.set_xlabel('Importance', color = p['muted'])
    ax.grid(True, axis = 'x', alpha = 0.25, linestyle = '--')
    plt.tight_layout()
    return _fig_to_bytes(fig, theme)
