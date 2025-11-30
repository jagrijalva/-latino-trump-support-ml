#!/usr/bin/env python3
"""
Generate PDF Summary Report for Latino Trump Support ML Analysis
Using matplotlib for PDF generation (more compatible)
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def create_text_page(pdf, title, content_lines, fontsize=10):
    """Create a page with text content"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, title, fontsize=14, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # Content
    y_pos = 0.88
    for line in content_lines:
        if line.startswith('##'):  # Section header
            ax.text(0.05, y_pos, line[2:].strip(), fontsize=11, fontweight='bold',
                   ha='left', va='top', transform=ax.transAxes)
            y_pos -= 0.035
        elif line.startswith('**'):  # Bold line
            ax.text(0.05, y_pos, line.replace('**',''), fontsize=fontsize, fontweight='bold',
                   ha='left', va='top', transform=ax.transAxes, family='monospace')
            y_pos -= 0.025
        elif line == '':  # Empty line
            y_pos -= 0.015
        else:
            ax.text(0.05, y_pos, line, fontsize=fontsize,
                   ha='left', va='top', transform=ax.transAxes, family='monospace')
            y_pos -= 0.025

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_table_page(pdf, title, df, start_rank=1):
    """Create a page with a table showing all rows in df"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, title, fontsize=12, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # Prepare display dataframe
    df_display = df.copy()
    if 'Feature' in df_display.columns:
        df_display['Feature'] = df_display['Feature'].str[:45]

    # Add rank column
    df_display.insert(0, 'Rank', range(start_rank, start_rank + len(df_display)))

    # Calculate table height based on number of rows
    n_rows = len(df_display)
    table_height = min(0.78, 0.025 * (n_rows + 1))

    # Create table
    table = ax.table(
        cellText=df_display.round(4).values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='upper center',
        bbox=[0.02, 0.92 - table_height - 0.05, 0.96, table_height]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)

    # Style header
    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor('#404040')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def main():
    output_path = 'analysis_summary_report.pdf'

    with PdfPages(output_path) as pdf:

        # =================================================================
        # PAGE 1: Overview and Performance
        # =================================================================
        page1_content = [
            '## Study Overview',
            '',
            'DV: Trump vote (1) vs Non-Trump vote (0) among Latino voters',
            'Data: CMPS 2016, Latino respondents who voted',
            'Method: Random Forest (500 trees, balanced class weights)',
            '',
            '## Sample Characteristics',
            '',
            'Total Latino Voters:          3,001',
            'Trump Voters (DV=1):          539 (18.0%)',
            'Non-Trump Voters (DV=0):      2,462 (82.0%)',
            'Features (one-hot encoded):   1,517',
            'Train/Test Split:             80/20 stratified',
            '',
            '## Model Performance',
            '',
            '**Full Model (all 1,517 features)',
            '  Test ROC-AUC:               0.9383',
            '',
            '**Non-Partisan Model (1,471 features)',
            '  Test ROC-AUC:               0.8736',
            '  AUC Drop:                   0.0647 (6.9% relative)',
            '',
            'Note: Non-partisan model excludes 46 variables measuring',
            'party ID, ideology, and candidate favorability ratings.',
            '',
            '## Key Findings',
            '',
            '1. Trump favorability is strongest single predictor',
            '2. Party ID and partisan leanings dominate full model',
            '3. Removing partisan vars reduces AUC by only 6.9%',
            '4. Non-partisan model reveals immigration attitudes:',
            '   - Views on immigration levels (increase/decrease)',
            '   - Citizenship pathway for undocumented',
            '   - Attitudes toward hearing Spanish spoken',
            '   - Support for deportation policies',
            '5. Strong predictive power (AUC 0.87) without partisan vars',
        ]

        create_text_page(pdf,
            'Latino Trump Support: ML Analysis Summary\nCMPS 2016 | Random Forest Classification',
            page1_content)

        # =================================================================
        # PAGE 2-3: Full Model Top 30 Predictors (split across 2 pages)
        # =================================================================
        try:
            df_full = pd.read_csv('top30_full_model.csv')
            df_full_display = df_full[['feature', 'importance_mean', 'importance_std']].head(30)
            df_full_display.columns = ['Feature', 'Importance', 'Std']

            # Page 2: Ranks 1-15
            create_table_page(pdf,
                'Top 30 Predictors - Full Model (AUC: 0.9383)\nRanks 1-15',
                df_full_display.iloc[0:15].copy(),
                start_rank=1)

            # Page 3: Ranks 16-30
            create_table_page(pdf,
                'Top 30 Predictors - Full Model (AUC: 0.9383)\nRanks 16-30',
                df_full_display.iloc[15:30].copy(),
                start_rank=16)

        except Exception as e:
            print(f'Could not load full model results: {e}')

        # =================================================================
        # PAGE 4-5: Non-Partisan Model Top 30 Predictors (split across 2 pages)
        # =================================================================
        try:
            df_np = pd.read_csv('top30_nonpartisan_model.csv')
            df_np_display = df_np[['feature', 'importance_mean', 'importance_std']].head(30)
            df_np_display.columns = ['Feature', 'Importance', 'Std']

            # Page 4: Ranks 1-15
            create_table_page(pdf,
                'Top 30 Predictors - Non-Partisan Model (AUC: 0.8736)\nRanks 1-15',
                df_np_display.iloc[0:15].copy(),
                start_rank=1)

            # Page 5: Ranks 16-30
            create_table_page(pdf,
                'Top 30 Predictors - Non-Partisan Model (AUC: 0.8736)\nRanks 16-30',
                df_np_display.iloc[15:30].copy(),
                start_rank=16)

        except Exception as e:
            print(f'Could not load non-partisan results: {e}')

        # =================================================================
        # PAGE 6: Partisan Variables Removed
        # =================================================================
        page6_content = [
            '## Partisan Variables Removed (46 columns total)',
            '',
            'Variable    Description',
            '----------- ------------------------------------------',
            'C25         Party registration',
            'C26         Strong partisan identification',
            'C31         Ideology (liberal-conservative scale)',
            'L46         Which party better on immigration',
            'L266        Which party better for Latinos',
            'L267        Which party better on values',
            'C2          Hillary Clinton favorability',
            'C3          Bernie Sanders favorability',
            'C4          Donald Trump favorability',
            'C5          Ted Cruz favorability',
            'C8          Bill Clinton favorability',
            'C9          Barack Obama favorability',
            'C242_HID    Party identification (derived)',
            '',
            '',
            '## Interpretation Notes',
            '',
            'Permutation Importance: Measures feature importance by',
            'shuffling each feature and measuring AUC decrease.',
            '(50 repeats for stability)',
            '',
            'SHAP Values: Measures average contribution of each',
            'feature to individual predictions.',
            '(Computed on 500 test observations)',
            '',
            '',
            f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        ]

        create_text_page(pdf, 'Appendix: Variables & Methods', page6_content)

    print(f'Report saved: {output_path}')

if __name__ == '__main__':
    main()
