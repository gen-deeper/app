# -*- coding: utf-8 -*-

# --- Install necessary packages ---
!pip install numpy pandas scipy scikit-learn statsmodels patsy matplotlib seaborn google-colab pyreadr rpy2 semopy
# pyreadr and rpy2 are for calling R functions from Python
# semopy for SEM analysis

# --- Imports ---
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import os
import random
from collections import defaultdict
import plotly.graph_objects as go  # For interactive plots
import plotly.express as px  # For easier interactive plots
import ipywidgets as widgets  # For interactive widgets
from IPython.display import display, HTML, clear_output # For displaying in Jupyter/Colab

# --- R and IPMA Setup (using rpy2) ---
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Activate pandas conversion for rpy2
pandas2ri.activate()

# Install the 'IPMA' R package (if not already installed)
utils = importr('utils')
try:
    IPMA_r = importr('IPMA')  # Try importing; will raise an error if not installed
except Exception as e:
    print(f"Error importing IPMA R package: {e}. Attempting to install...")
    utils.install_packages('IPMA', repos='https://cloud.r-project.org')
    try:
        IPMA_r = importr('IPMA')  # Import after installation
        print("IPMA R package successfully installed and imported.")
    except Exception as e:
        print(f"Failed to install and import IPMA R package: {e}. IPMA functionality will be limited.")
        IPMA_r = None  # Set to None to avoid further errors

def run_IPMA_in_r(data, x_columns, y_column):
    """
    Runs IPMA analysis using the R 'IPMA' package via rpy2.

    Args:
        data (pd.DataFrame): The input data.
        x_columns (list): List of X variable column names.
        y_column (str): The Y variable column name.

    Returns:
        dict: A dictionary containing the IPMA results (effect sizes, bottleneck table, etc.).
    """
    if IPMA_r is None:
        print("IPMA R package is not available. Skipping IPMA analysis.")
        return {}

    # Convert the pandas DataFrame to an R DataFrame
    r_data = pandas2ri.py2rpy(data)

    # Construct the IPMA formula string
    formula_str = f"{y_column} ~ { '+'.join(x_columns) }"

    # Run the IPMA analysis in R
    try:
        IPMA_results_r = IPMA_r.IPMA(r_data, formula_str)
    except Exception as e:
        print(f"Error running IPMA in R: {e}")
        return {}

    # Extract relevant results (effect sizes, bottleneck table, etc.)
    effect_sizes = {}
    for i, x_col in enumerate(x_columns):
        # Access effect sizes (example - adjust indices as needed based on IPMA_results_r structure)
        try:
            effect_sizes[x_col] = IPMA_results_r[0][i]  # Accessing effect sizes.  Index might need adjustment.
        except IndexError:
            effect_sizes[x_col] = None # Handle cases where effect size isn't available
            print(f"Warning: Could not retrieve effect size for {x_col} using direct indexing.")

    # Extract and convert the bottleneck table to a pandas DataFrame
    try:
        bottleneck_table_r = IPMA_results_r[6]  # Bottleneck table (adjust index if needed)
        bottleneck_table_pd = pandas2ri.rpy2py(bottleneck_table_r)
    except:
        bottleneck_table_pd = None
        print("Warning: Could not retrieve bottleneck table using direct indexing.")

    # Extract ceiling coordinates for plotting
    try:
        x_ceiling = list(IPMA_results_r[4][0])  # Example: Accessing x_ceiling (adjust as needed)
        y_ceiling = list(IPMA_results_r[4][1])  # Example: Accessing y_ceiling (adjust as needed)
    except:
        x_ceiling = None
        y_ceiling = None
        print("Warning: Could not retrieve ceiling coordinates using direct indexing.")

    return {
        "effect_sizes": effect_sizes,
        "bottleneck_table": bottleneck_table_pd,
        "x_ceiling": x_ceiling,
        "y_ceiling": y_ceiling,
        "raw_r_results": IPMA_results_r  # Include the raw R results for debugging
    }


# --- SEMopy Setup ---
import semopy

def perform_sem_ipma(data, model_specification):
    """
    Performs SEM analysis and IPMA using semopy.

    Args:
        data (pd.DataFrame): The input data.
        model_specification (str): The SEM model specification in semopy syntax.

    Returns:
        tuple: (semopy.Model, dict) The fitted model and IPMA results.
    """
    model = semopy.Model(model_specification)
    try:
        model.fit(data)
    except Exception as e:
        print(f"SEM Fit Error: {e}")
        return None, None

    # IPMA using semopy
    try:
        ipma_results = model.inspect(what='ipma', target='Performance') # Adjust target as needed
    except Exception as e:
        print(f"IPMA Error: {e}")
        return model, None

    return model, ipma_results


# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Define Output Directory ---
output_dir = '/content/drive/MyDrive/data'  # Path to the 'data' folder
os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

# --- Explicitly set Matplotlib backend to 'Agg' for Colab ---
plt.switch_backend('Agg')

# --- Neon Theme Function (for Matplotlib/Seaborn) ---
# Modified to avoid white
def apply_minimalist_neon_theme():
    """Applies a minimalist neon theme with no filled areas and no white elements."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#000000',  # Black background
        'figure.facecolor': '#000000',
        'text.color': '#00FF00',  # Bright green text
        'axes.labelcolor': '#00FFFF',  # Cyan axis labels
        'xtick.color': '#00FFFF',
        'ytick.color': '#00FFFF',
        'grid.color': '#444444',  # Darker grid lines
        'lines.color': '#00FF00',
        'patch.facecolor': '#000000',  # Black patch fill
        'patch.edgecolor': '#00FFFF',  # Cyan patch outline
        'axes.edgecolor': '#00FFFF', # Cyan axes
        'boxplot.boxprops.color': '#00FFFF',
        'boxplot.whiskerprops.color': '#00FFFF',
        'boxplot.capprops.color': '#FF00FF', # Magenta median
        'boxplot.flierprops.markeredgecolor': '#00FF00',
        'figure.edgecolor': '#000000', # Black figure edge
        'savefig.facecolor': '#000000', # Black background for saved figures
        'savefig.edgecolor': '#000000', # Black edge for saved figures
    })
    sns.set_style("darkgrid", {"axes.facecolor": "#000000", "grid.color": "#444444"})
    sns.set_palette(["#00FF00", "#00FFFF", "#FF00FF", "#FFFF00"])  # Neon palette


# --- Data Simulation ---
def simulate_data(n_participants=40, seed=42):
    """Simulates data, including demographics, interventions, psychological
    measures, performance, and neurophysiological data.  Effects of LLM and
    herbal blend are simulated.

    Args:
        n_participants (int): Number of participants.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: The simulated dataset.
    """

    np.random.seed(seed)

    # Demographics
    age = np.random.randint(18, 30, size=n_participants)
    gender = np.random.choice(['Male', 'Female', 'Other'], size=n_participants)
    programming_experience = np.random.choice(
        ['Beginner', 'Intermediate', 'Advanced'], size=n_participants
    )

    # Group Assignment (balanced)
    llm_usage = np.array([1, 1, 0, 0] * (n_participants // 4))
    herbal_blend = np.array([1, 0, 1, 0] * (n_participants // 4))

    # Psychological Measures (initial and final)
    initial_self_efficacy = np.random.normal(3.5, 0.5, size=n_participants)
    initial_anxiety = np.random.normal(2.5, 0.6, size=n_participants)
    final_self_efficacy = initial_self_efficacy.copy()
    final_anxiety = initial_anxiety.copy()

    # Performance Measures
    errors_identified = np.random.randint(5, 20, size=n_participants)
    completion_time = np.random.uniform(180, 400, size=n_participants)

    # Adjust based on group (simulated effects)
    for i in range(n_participants):
        if llm_usage[i] == 1:
            final_self_efficacy[i] += 0.5
            final_anxiety[i] -= 0.4
            errors_identified[i] += 3
            completion_time[i] -= 15
        if herbal_blend[i] == 1:
            final_anxiety[i] -= 0.3
            errors_identified[i] += 1

    # Ensure reasonable bounds
    final_self_efficacy = np.clip(final_self_efficacy, 1, 5)
    final_anxiety = np.clip(final_anxiety, 1, 4)
    errors_identified = np.maximum(0, errors_identified)
    completion_time = np.maximum(60, completion_time)

    # Combine ErrorsIdentified and CompletionTime into a single 'Performance' variable
    #  This is a simplification; a more sophisticated approach might involve standardization
    #  or a weighted average.
    performance = (errors_identified + (500 - completion_time)/10) / 2 # Scale completion time to be positive and comparable

    # Neurophysiological Data (simplified)
    eeg_alpha = np.random.normal(10, 2, size=n_participants)
    eeg_beta = np.random.normal(18, 3, size=n_participants)
    ecg_hr = np.random.normal(75, 10, size=n_participants)
    eda_scr = np.random.normal(0.5, 0.2, size=n_participants)
    pog_fixations = np.random.randint(20, 100, size=n_participants)
    pog_fixation_duration = np.random.uniform(200, 500, size=n_participants)
    pog_pupil_diameter = np.random.normal(3.5, 0.5, size=n_participants)
    pog_blink_rate = np.random.uniform(10, 30, size=n_participants)

    # Adjust based on group (simulated effects)
    for i in range(n_participants):
        if llm_usage[i] == 1:
            eeg_beta[i] += 2
            pog_fixations[i] -= 5
            pog_fixation_duration[i] += 50
        if herbal_blend[i] == 1:
            ecg_hr[i] -= 5
            eda_scr[i] -= 0.1

    # Create DataFrame
    data = pd.DataFrame({
        'ParticipantID': range(1, n_participants + 1),
        'Age': age,
        'Gender': gender,
        'ProgrammingExperience': programming_experience,
        'LLMUsage': llm_usage,
        'HerbalBlend': herbal_blend,
        'InitialSelfEfficacy': initial_self_efficacy,
        'FinalSelfEfficacy': final_self_efficacy,
        'InitialAnxiety': initial_anxiety,
        'FinalAnxiety': final_anxiety,
        'ErrorsIdentified': errors_identified,
        'CompletionTime': completion_time,
        'Performance': performance, # Combined performance measure
        'EEGAlpha': eeg_alpha,
        'EEGBeta': eeg_beta,
        'ECG_HR': ecg_hr,
        'EDA_SCR': eda_scr,
        'POGFixations': pog_fixations,
        'POGFixationDuration': pog_fixation_duration,
        'POGPupilDiameter': pog_pupil_diameter,
        'POGBlinkRate': pog_blink_rate
    })

    return data

# --- Data Preprocessing ---
def preprocess_data(data):
    """Preprocesses data: one-hot encodes categoricals and scales numericals.
    Splitting into training and testing sets is now done within the SEM analysis.

    Args:
        data (pd.DataFrame): The raw data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """

    features = data.drop(columns=['ParticipantID', 'ErrorsIdentified', 'CompletionTime', 'Performance'])
    performance = data[['Performance']] # Use the combined performance measure
    features = pd.get_dummies(features, columns=['Gender', 'ProgrammingExperience'])
    numerical_features = features.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    features[numerical_features] = scaler.fit_transform(features[numerical_features])
    processed_data = pd.concat([features, performance], axis=1) # Combine features and performance
    return processed_data

# --- Statistical Analyses ---
def perform_statistical_analysis(data):
    """Performs descriptive stats, correlations, and group comparisons (t-tests).

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        tuple: (descriptive_stats, correlation_matrix, group_comparison_results)
    """

    descriptive_stats = data.describe()
    correlation_matrix = data[[
        'FinalSelfEfficacy', 'FinalAnxiety', 'Performance' # Use combined performance
    ]].corr()
    group_comparison_results = {}
    for variable in ['FinalSelfEfficacy', 'FinalAnxiety', 'Performance']: # Use combined performance
        llm_group = data[data['LLMUsage'] == 1][variable]
        no_llm_group = data[data['LLMUsage'] == 0][variable]
        t_stat, p_val = stats.ttest_ind(llm_group, no_llm_group)
        group_comparison_results[variable] = {'t-statistic': t_stat, 'p-value': p_val}
    return descriptive_stats, correlation_matrix, group_comparison_results


def perform_regression_analysis(data, dependent_variable='Performance'): # Use combined performance
    """Performs regression analysis using statsmodels on the entire dataset.

    Args:
        data (pd.DataFrame): The dataset.
        dependent_variable (str): Dependent variable to predict.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: Results.
    """

    formula = f"{dependent_variable} ~ LLMUsage + HerbalBlend + InitialSelfEfficacy + InitialAnxiety"
    y, X = dmatrices(formula, data=data, return_type='dataframe')
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

# --- Qualitative Analysis ---
def analyze_prompts(data):
    """Simulates prompt analysis, generating more realistic prompt data
    based on LLM usage and then analyzing it.  Handles potential
    ZeroDivisionError.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        dict: Analysis results, including generated prompts.
    """
    prompts = []
    for i in range(len(data)):
        if data['LLMUsage'][i] == 1:
            # Simulate more specific prompts for LLM users
            prompt_type = random.choice(["debug", "explain", "optimize"])
            if prompt_type == "debug":
                prompts.append(f"P{i+1}: Find the error in this code: `x = 10; y = 0; z = x / y`")
            elif prompt_type == "explain":
                prompts.append(f"P{i+1}: Explain what this function does: `def add(a, b): return a + b`")
            else:  # optimize
                prompts.append(f"P{i+1}: How can I make this code faster: `for i in range(1000000): pass`")
        else:
            # Simulate more general questions for non-LLM users
            prompts.append(f"P{i+1}: I'm stuck on this task, can you give me a hint?")

    # Analyze the generated prompts
    prompt_lengths = [len(p.split()) for p in prompts]
    # Handle potential ZeroDivisionError if prompt_lengths is empty
    average_prompt_length = np.mean(prompt_lengths) if prompt_lengths else 0

    # Count keywords (more robustly)
    keyword_counts = defaultdict(int)
    for p in prompts:
        for word in p.lower().split():
            if word not in ["i", "this", "the", "a", "in", "on", "can", "you", "me", "what", "how", "is", "do", "does", "an", "here", "fix"]: # Common words
                keyword_counts[word] += 1
    most_common_keywords = sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)[:5]


    prompt_analysis = {
        "prompts": prompts,  # Include the generated prompts
        "average_prompt_length": average_prompt_length,
        "most_common_keywords": most_common_keywords,
        "question_types": ["debug", "explain", "optimize", "general help"],  # Based on simulation
    }
    return prompt_analysis


def analyze_interviews(data):
    """Simulates interview analysis, generating more realistic feedback.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        dict: Analysis results, including generated feedback.
    """
    qualitative_feedback = []
    for i in range(len(data)):
        if data['LLMUsage'][i] == 1:
            feedback = random.choice([
                "The LLM helped me find the bug quickly.",
                "I understood the code better with the LLM's explanation.",
                "The LLM gave me suggestions I wouldn't have thought of."
            ])
        else:
            feedback = random.choice([
                "I wish I had a tool to help me understand the code.",
                "I spent a lot of time trying to find the error myself.",
                "It was difficult to debug without assistance."
            ])
        qualitative_feedback.append(f"P{i+1}: {feedback}")

    interview_analysis = {
        "perceived_usefulness_llm": np.random.uniform(3, 5) if data['LLMUsage'].any() else np.random.uniform(1, 3),
        "anxiety_reduction_llm": np.random.uniform(1, 3) if data['LLMUsage'].any() else np.random.uniform(0, 1),
        "anxiety_reduction_herbal": np.random.uniform(1, 3) if data['HerbalBlend'].any() else np.random.uniform(0, 1),
        "qualitative_feedback": qualitative_feedback,  # Include generated feedback
    }
    return interview_analysis

# --- SEM Diagram Generation (using Matplotlib) ---
def create_sem_diagram_mpl(model_name, nodes, edges, filename):
    """Creates a conceptual SEM diagram using Matplotlib, with arrowheads.

    Args:
        model_name (str): Name of the model.
        nodes (dict): Node labels and positions: {'node_label': (x, y)}.
        edges (list of tuples): Edges: [(source, target), ...].
        filename (str): Output filename.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # Removed facecolor settings, handled by theme now
    #fig.set_facecolor('#262626')
    #ax.set_facecolor('#262626')


    # Draw nodes (perfect circles)
    for label, pos in nodes.items():
        ax.add_patch(plt.Circle(pos, 0.3, edgecolor='#00FFFF', facecolor='none', linewidth=2, zorder=2))  # Cyan outline, no fill
        ax.text(pos[0], pos[1], label, color='#00FF00', ha='center', va='center', fontsize=10, zorder=3) # Green text

    # Draw edges with arrowheads
    for source, target in edges:
        x1, y1 = nodes[source]
        x2, y2 = nodes[target]
        # Use arrow instead of line for arrowheads
        ax.arrow(x1, y1, x2 - x1, y2 - y1,
                 head_width=0.15,  # Adjust arrowhead size
                 head_length=0.2,  # Adjust arrowhead length
                 fc='#00FF00',  # Arrowhead color (green)
                 ec='#00FF00',  # Edge color (green)
                 length_includes_head=True,
                 linewidth=2, # Thicker lines
                 zorder=1)

    ax.set_title(f"SEM Model: {model_name}", color='#00FFFF')
    ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"SEM diagram saved to: {filename}")


# --- Statistical Plotting Functions (Matplotlib/Seaborn) ---
def create_histogram_mpl(data, column, filename):
    """Creates a histogram with the neon theme (Matplotlib)."""
    plt.figure(figsize=(8, 6))
    # Use plt.hist for more control, no fill
    n, bins, patches = plt.hist(data[column], edgecolor='#00FFFF', facecolor='none', linewidth=2)
    plt.title(f"Histogram of {column}", color='#00FFFF') # Explicitly set title color
    plt.xlabel(column, color='#00FFFF') # Explicitly set label colors
    plt.ylabel("Frequency", color='#00FFFF')
    plt.savefig(filename)
    plt.close()
    print(f"Histogram saved to: {filename}")

def create_violin_plot_mpl(data, x_column, y_column, filename):
    """Creates a violin plot with the neon theme (Matplotlib/Seaborn)."""
    plt.figure(figsize=(8, 6))
    # Use Seaborn, but customize to remove fill
    sns.violinplot(x=data[x_column], y=data[y_column], color='#00FFFF', linewidth=2, inner=None) # No fill, cyan outline
    plt.title(f"Violin Plot of {y_column} by {x_column}", color='#00FFFF')
    plt.xlabel(x_column, color='#00FFFF')
    plt.ylabel(y_column)
    plt.savefig(filename)
    plt.close()
    print(f"Violin plot saved to: {filename}")

def create_kde_plot_mpl(data, column1, column2, filename):
    """Creates a 2D KDE plot with the neon theme (Matplotlib/Seaborn)."""
    plt.figure(figsize=(8, 6))
    # Use Seaborn, but plot only the contours, no fill
    sns.kdeplot(x=data[column1], y=data[column2], color='#00FFFF', linewidths=2, thresh=0, levels=10) # Only contour lines
    plt.title(f"KDE Plot of {column1} vs. {column2}", color='#00FFFF')
    plt.xlabel(column1, color='#00FFFF')
    plt.ylabel(column2)
    plt.savefig(filename)
    plt.close()
    print(f"KDE plot saved to: {filename}")

def create_stacked_bar_plot_mpl(data, x_column, y_column, color_column, filename):
    """Creates a stacked bar plot (Matplotlib/Seaborn)."""
    plt.figure(figsize=(8, 6))
    pivot_data = data.groupby([x_column, color_column])[y_column].mean().unstack()
    # Plot with no fill, only outlines
    pivot_data.plot(kind='bar', stacked=True, edgecolor='#00FFFF', linewidth=2, ax=plt.gca(), legend=False)
    plt.title(f"Stacked Bar Plot of {y_column} by {x_column} and {color_column}", color='#00FFFF')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    #plt.legend(title=color_column) # Removed legend to simplify
    plt.savefig(filename)
    plt.close()
    print(f"Stacked bar plot saved to: {filename}")

def create_heatmap(data, filename):
    """Creates a correlation heatmap and saves it as a PNG file."""
    data_numeric = data.select_dtypes(include=np.number)
    if data_numeric.shape[1] <= 1:
        print("Not enough numeric columns to create a heatmap.")
        return

    plt.figure(figsize=(10, 8))
    # Use Seaborn, customize colormap and lines
    sns.heatmap(data_numeric.corr(), annot=True, cmap=sns.color_palette("coolwarm", as_cmap=True), fmt=".2f",
                linewidths=.5, linecolor='#00FFFF', cbar=False, annot_kws={"color": "#00FF00"}) # No colorbar
    plt.title("Correlation Heatmap")
    plt.savefig(filename, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Heatmap saved to: {filename}")

def generate_summary_html(data, filename="summary_statistics.html"):
    """Generates descriptive statistics and saves them as an HTML file."""
    if data.empty:
        print("No data available to generate summary.")
        return

    # Select numeric columns for analysis
    data_numeric = data.select_dtypes(include=np.number)

    # Generate descriptive statistics
    summary_stats = data_numeric.describe().transpose()

    # Save to HTML
    summary_html = summary_stats.to_html(classes='table table-dark', border=0)
    with open(filename, "w") as f:
        f.write("<html><head><title>Summary Statistics</title></head><body>")
        f.write("<h1>Descriptive Statistics</h1>")
        f.write(summary_html)
        f.write("</body></html>")

    print(f"Summary statistics saved to {filename}")

# --- Interactive Visualization with Plotly and ipywidgets ---

def interactive_IPMA_visualization(data):
    """
    Creates an interactive IPMA visualization using Plotly and ipywidgets.
    Allows users to select X and Y variables and view the IPMA results.
    This version uses the R-based IPMA.
    """

    # --- Widgets ---
    x_variable_dropdown = widgets.SelectMultiple(
        options=[col for col in data.columns if data[col].dtype in ['int64', 'float64']],
        description='X Variables:',
        style={'description_width': 'initial'}
    )

    y_variable_dropdown = widgets.Dropdown(
        options=[col for col in data.columns if data[col].dtype in ['int64', 'float64']],
        description='Y Variable:',
        style={'description_width': 'initial'}
    )


    run_button = widgets.Button(description="Run IPMA (R)")
    output_area = widgets.Output()  # For displaying results and plots
    bottleneck_table_output = widgets.Output() # For displaying bottleneck table

    # --- Layout ---
    input_widgets = widgets.HBox([x_variable_dropdown, y_variable_dropdown, run_button])
    display(input_widgets)
    display(output_area)
    display(bottleneck_table_output)

    # --- Event Handler ---
    def run_IPMA_on_click(button):
        """Runs IPMA and updates the output area with results and plots."""
        with output_area:
            clear_output(wait=True)  # Clear previous output
            x_vars = list(x_variable_dropdown.value)
            y_var = y_variable_dropdown.value


            if not x_vars or not y_var:
                print("Please select both X and Y variables.")
                return

            # Run IPMA using the R function
            IPMA_results = run_IPMA_in_r(data, x_vars, y_var)

            # Display Summary
            print(f"IPMA Results (R): {', '.join(x_vars)} vs. {y_var}")
            for x_var, effect_size in IPMA_results['effect_sizes'].items():
                if effect_size is not None:
                    print(f"Effect Size ({x_var}): {effect_size:.2f}")
                else:
                    print(f"Effect Size ({x_var}): Not available")

            # --- Interactive Scatter Plot with Ceiling Line ---
            if len(x_vars) == 1:  # Only create the plot if there's a single X variable
                x_var = x_vars[0]
                fig = go.Figure()

                # Scatter plot of data points
                fig.add_trace(go.Scatter(x=data[x_var], y=data[y_var], mode='markers',
                                         marker=dict(color='#00FF00', line=dict(color='#00FFFF', width=0.5)),
                                         name='Data Points'))

                # Ceiling line (if available)
                if IPMA_results['x_ceiling'] is not None and IPMA_results['y_ceiling'] is not None:
                    fig.add_trace(go.Scatter(x=IPMA_results['x_ceiling'], y=IPMA_results['y_ceiling'], mode='lines',
                                             line=dict(color='#FF00FF', width=2),
                                             name='Ceiling Line'))

                fig.update_layout(
                    title=f"IPMA (R): {x_var} vs. {y_var}",
                    xaxis_title=x_var,
                    yaxis_title=y_var,
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='#00FFFF'),
                    xaxis=dict(gridcolor='#444444', zerolinecolor='#444444'),
                    yaxis=dict(gridcolor='#444444')
                )
                fig.show()
            else:
                print("Scatter plot with ceiling line is only displayed for a single X variable.")


        with bottleneck_table_output:
            clear_output(wait=True)
            # Display Bottleneck Table (as HTML for better formatting)
            if IPMA_results['bottleneck_table'] is not None:
                display(HTML(IPMA_results['bottleneck_table'].to_html(classes='bottleneck-table')))
            else:
                print("Bottleneck table not available.")


    # --- Attach Event Handler ---
    run_button.on_click(run_IPMA_on_click)


def interactive_sem_visualization(data):
    """
    Creates an interactive SEM visualization using Plotly and ipywidgets.
    Allows users to select variables and visualize relationships.
    This version integrates SEMopy and IPMA-SEM.
    """

    # --- Widgets ---
    variable_selector = widgets.SelectMultiple(
        options=data.columns.tolist(),
        description='Select Variables:',
        style={'description_width': 'initial'}
    )

    # SEM Model Specification Widget
    model_specification_text = widgets.Textarea(
        value='',
        placeholder='Enter SEM model specification (semopy syntax)',
        description='SEM Model Specification:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(height='150px', width='50%')
    )

    run_sem_button = widgets.Button(description="Run SEM & IPMA")
    output_area = widgets.Output()

    # --- Layout ---
    display(widgets.VBox([variable_selector, model_specification_text, run_sem_button, output_area]))

    # --- Event Handlers ---
    def run_sem_ipma(button):
        """Runs SEM and IPMA based on user selections."""
        with output_area:
            clear_output(wait=True)
            selected_vars = list(variable_selector.value)
            model_spec = model_specification_text.value

if not selected_vars or not model_spec.strip():  # Check for empty string after removing whitespace
                print("Please select variables and provide a valid model specification.")
                return

            # --- Run SEM and IPMA ---
            sem_model, sem_ipma_results = perform_sem_ipma(data, model_spec)

            if sem_model is None:
                print("SEM model failed to fit. Check the model specification.")
                return

            # --- Display SEM Results ---
            print("SEM Model Summary:")
            print(sem_model.summary())

            # --- Display IPMA Results ---
            if sem_ipma_results is not None:
                print("\nIPMA Results:")
                print(sem_ipma_results)

                # --- Visualize IPMA Results (Example) ---
                #  This is a basic example; adapt it to your specific needs.
                if isinstance(sem_ipma_results, pd.DataFrame): # Check if results are a DataFrame
                    try:
                        fig = px.scatter(sem_ipma_results, x="total", y="importance",
                                        color_discrete_sequence=['#00FFFF'],
                                        title="IPMA: Total Effect vs. Importance")
                        fig.update_traces(marker=dict(line=dict(color='#00FFFF', width=0.5)))
                        fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#00FFFF'))
                        fig.show()
                    except Exception as e:
                        print(f"Error creating IPMA plot: {e}")
                else:
                    print("IPMA results are not in a DataFrame format and cannot be plotted directly.")


            else:
                print("IPMA analysis failed.")


    # --- Attach Event Handler ---
    run_sem_button.on_click(run_sem_ipma)


def interactive_data_exploration(data):
    """
    Provides interactive widgets for general data exploration (histograms, violin plots, etc.).
    """
    # --- Widgets ---
    plot_type = widgets.Dropdown(
        options=['Histogram', 'Violin Plot', 'Scatter Plot', 'Box Plot', 'KDE Plot (2D)'],
        value='Histogram',
        description='Plot Type:',
        style={'description_width': 'initial'}
    )

    x_variable = widgets.Dropdown(
        options=data.columns.tolist(),
        description='X Variable:',
        style={'description_width': 'initial'}
    )

    y_variable = widgets.Dropdown(
        options=['None'] + data.columns.tolist(),  # Allow "None" for univariate plots
        value='None',
        description='Y Variable:',
        style={'description_width': 'initial'}
    )

    group_variable = widgets.Dropdown( # for stacked bar plots and grouped plots
        options = ['None'] + data.columns.tolist(),
        value = 'None',
        description = 'Group by (for stacked/grouped plots):',
        style={'description_width': 'initial'}
    )

    plot_button = widgets.Button(description="Generate Plot")
    output_area = widgets.Output()

    # --- Layout ---
    display(widgets.VBox([plot_type, x_variable, y_variable, group_variable, plot_button, output_area]))

    # --- Event Handlers ---
    def generate_plot(button):
        with output_area:
            clear_output(wait=True)
            plot_choice = plot_type.value
            x_var = x_variable.value
            y_var = y_variable.value if y_variable.value != 'None' else None
            group_var = group_variable.value if group_variable.value != 'None' else None

            if plot_choice == 'Histogram':
                fig = px.histogram(data, x=x_var, color_discrete_sequence=['#00FFFF'],
                                   title=f"Histogram of {x_var}")
                fig.update_traces(marker=dict(line=dict(color='#00FFFF', width=0.5)))
                fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#00FFFF'))
                fig.show()

            elif plot_choice == 'Violin Plot':
                if y_var:
                    fig = px.violin(data, x=x_var, y=y_var, color_discrete_sequence=['#00FFFF'],
                                    title=f"Violin Plot of {y_var} by {x_var}")
                    fig.update_traces(line=dict(color='#00FFFF')) # No fill, just outline
                    fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#00FFFF'))
                    fig.show()
                else:
                    print("Please select a Y variable for the Violin Plot.")

            elif plot_choice == 'Scatter Plot':
                if y_var:
                    fig = px.scatter(data, x=x_var, y=y_var, color_discrete_sequence=['#00FFFF'],
                                     title=f"Scatter Plot of {x_var} vs. {y_var}")
                    fig.update_traces(marker=dict(line=dict(color='#00FFFF', width=0.5)))
                    fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#00FFFF'))
                    fig.show()
                else:
                    print("Please select a Y variable for the Scatter Plot.")

            elif plot_choice == 'Box Plot':
                if y_var:
                    fig = px.box(data, x=x_var, y=y_var, color_discrete_sequence=['#00FFFF'],
                                 title=f"Box Plot of {y_var} by {x_var}")
                    fig.update_traces(line=dict(color='#00FFFF'))
                    fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#00FFFF'))
                    fig.show()
                else:
                    print("Please select a Y variable for the Box Plot")

            elif plot_choice == 'KDE Plot (2D)':
                if y_var:
                    try:
                        fig = px.density_contour(data, x=x_var, y=y_var,
                                                 color_discrete_sequence=['#00FFFF'],
                                                 title=f"KDE Plot of {x_var} vs. {y_var}")
                        fig.update_traces(contours_coloring="lines", line_width=2) # Contour lines only
                        fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#00FFFF'))
                        fig.show()
                    except ValueError as e:
                        print(f"Error creating KDE Plot: {e}.  Check if the selected variables are suitable for a 2D KDE.")
                else:
                    print("Please select a Y variable for the KDE Plot.")

    # --- Attach Event Handler ---
    plot_button.on_click(generate_plot)


# --- Main Execution Block ---

if __name__ == '__main__':
    apply_minimalist_neon_theme()  # Apply the theme
    data = simulate_data()  # Generate the data
    processed_data = preprocess_data(data) #preprocess

    # --- Basic Statistical Analysis and Plotting ---
    descriptive_stats, correlation_matrix, group_comparison_results = perform_statistical_analysis(data)

    # Save descriptive stats to HTML
    generate_summary_html(data, filename=os.path.join(output_dir, "summary_statistics.html"))

    # --- Create and save plots (REVISED PLOT LIST) ---

    # Violin Plots
    create_violin_plot_mpl(data, 'LLMUsage', 'CompletionTime', os.path.join(output_dir, 'llm_completion_violin.png'))
    create_violin_plot_mpl(data, 'LLMUsage', 'ErrorsIdentified', os.path.join(output_dir, 'llm_errors_violin.png'))
    create_violin_plot_mpl(data, 'HerbalBlend', 'CompletionTime', os.path.join(output_dir, 'herbal_completion_violin.png'))
    create_violin_plot_mpl(data, 'HerbalBlend', 'ErrorsIdentified', os.path.join(output_dir, 'herbal_errors_violin.png'))
    create_violin_plot_mpl(data, 'LLMUsage', 'FinalSelfEfficacy', os.path.join(output_dir, 'llm_selfefficacy_violin.png'))
    create_violin_plot_mpl(data, 'HerbalBlend', 'FinalAnxiety', os.path.join(output_dir, 'herbal_anxiety_violin.png'))

    # Stacked Bar Plots
    create_stacked_bar_plot_mpl(data, 'ProgrammingExperience', 'CompletionTime', 'LLMUsage', os.path.join(output_dir, 'experience_completion_llm_stackedbar.png'))
    create_stacked_bar_plot_mpl(data, 'ProgrammingExperience', 'ErrorsIdentified', 'LLMUsage', os.path.join(output_dir, 'experience_errors_llm_stackedbar.png'))
    create_stacked_bar_plot_mpl(data, 'ProgrammingExperience', 'FinalSelfEfficacy', 'LLMUsage', os.path.join(output_dir, 'experience_selfefficacy_llm_stackedbar.png'))
    create_stacked_bar_plot_mpl(data, 'ProgrammingExperience', 'FinalAnxiety', 'LLMUsage', os.path.join(output_dir, 'experience_anxiety_llm_stackedbar.png'))
    create_stacked_bar_plot_mpl(data, 'Gender', 'CompletionTime', 'HerbalBlend', os.path.join(output_dir, 'gender_completion_herbal_stackedbar.png'))
    create_stacked_bar_plot_mpl(data, 'Gender', 'ErrorsIdentified', 'HerbalBlend', os.path.join(output_dir, 'gender_errors_herbal_stackedbar.png'))


    # --- SEM Diagrams (Conceptual) ---

    # 1. Basic Model
    nodes_basic = {
        'LLM Usage': (1, 3),
        'Herbal Blend': (1, 1),
        'Self-Efficacy': (3, 4),
        'Anxiety': (3, 2),
        'Performance': (5, 3)
    }
    edges_basic = [
        ('LLM Usage', 'Self-Efficacy'),
        ('LLM Usage', 'Anxiety'),
        ('LLM Usage', 'Performance'),
        ('Herbal Blend', 'Anxiety'),
        ('Self-Efficacy', 'Performance'),
        ('Anxiety', 'Performance')
    ]
    create_sem_diagram_mpl("Basic Model", nodes_basic, edges_basic, os.path.join(output_dir, 'sem_diagram_basic.png'))

    # --- Regression Analysis ---
    regression_results = perform_regression_analysis(processed_data, 'Performance')
    print("\nRegression Results (Performance):\n", regression_results.summary())

    # --- Qualitative Analyses ---
    prompt_analysis_results = analyze_prompts(data)
    print("\nPrompt Analysis:\n", prompt_analysis_results)
    interview_analysis_results = analyze_interviews(data)
    print("\nInterview Analysis:\n", interview_analysis_results)

    # --- Interactive Visualizations ---
    print("\nInteractive IPMA Visualization (R):")
    interactive_IPMA_visualization(data)

    print("\nInteractive SEM Visualization:")
    interactive_sem_visualization(processed_data)

    print("\nInteractive Data Exploration:")
    interactive_data_exploration(data)
