"""
Module containing visualization functions for data analysis,
particularly suited for biochemical data representation.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes




def plot_parallel_hists(
    counts_array: np.ndarray,
    bins_array: np.ndarray,
    labels: List[str],
    colors: List[str],
    xlabel: str,
    title: str,
    filename: str,
    lines: Optional[List[Tuple[str, float]]] = None,
    xlim: Optional[Tuple[float, float]] = None
    ) -> None:
    """
    Creates a plot with multiple parallel histograms.
    
    This function generates a plot displaying multiple histograms
    one above the other, allowing direct visual comparison.
    
    Parameters
    ----------
    counts_array : np.ndarray
        2D array containing histogram values.
        Each row represents a different histogram.
    bins_array : np.ndarray
        2D array containing bin boundaries for each histogram.
    labels : List[str]
        List of labels for each histogram.
    colors : List[str]
        List of colors for each histogram.
    xlabel : str
        Label for the x-axis.
    title : str
        Title of the plot.
    filename : str
        Name of the file to save the plot.
    lines : Optional[List[Tuple[str, float]]], optional
        List of tuples (name, position) to draw vertical lines.
    xlim : Optional[Tuple[float, float]], optional
        Limits (min, max) for the x-axis.
    
    Returns
    -------
    None
        The function displays and saves the plot but doesn't return any value.
    """
    # Check input dimensions
    if not len(colors) == len(labels) == np.shape(counts_array)[0]:
        print(f"Incompatible dimensions: colors={len(colors)}, labels={len(labels)}, "
              f"counts_array={np.shape(counts_array)[0]}")
        print("Dimensions do not match. The size of colors and labels must "
              "correspond to the number of histograms to be plotted.")
        return None
    
    if not np.shape(counts_array)[1] == np.shape(bins_array)[1] - 1:
        print(f"Incompatible dimensions: counts_array={np.shape(counts_array)[1]}, "
              f"bins_array={np.shape(bins_array)[1]}")
        print("Dimensions do not match. Abscissa (number of columns in "
              "bins_array) and ordinates (number of columns in counts_array) do not match.")
        return None
    
    # Create figure
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 1, 1)
    plt.xlabel(xlabel, fontsize=15)
    plt.yticks([], fontsize=20)
    plt.xticks(fontsize=30)
    plt.title(title, fontsize=25, pad=30)
    plt.yticks([])
    
    # Define vertical offsets for labels
    vertical_offsets = [0.7] * len(labels)  # Default value of 0.7
    # Specific adjustment for first elements if necessary
    if len(labels) > 0 and len(labels) <= 4:
        vertical_offsets[:len(labels)] = [0.65] * len(labels)
    
    # Plot histograms
    num_histograms, num_bins = np.shape(counts_array)  # Number of histograms, number of bins per histogram
    gap = np.max(counts_array) + 1/num_histograms * np.max(counts_array)
    vertical_position = 0
    
    for hist_index in range(num_histograms):
        histogram = np.zeros((num_bins, 2))
        histogram[:, 0] = bins_array[hist_index][1:]
        histogram[:, 1] = counts_array[hist_index] + vertical_position
        
        # Plot baseline
        ax.plot([np.min(bins_array), np.max(bins_array)], 
                [vertical_position, vertical_position], color='black')
        
        # Plot histogram
        ax.plot(histogram[:, 0], histogram[:, 1], linewidth=2, 
                linestyle='solid', color=colors[hist_index])
        ax.fill_between(histogram[:, 0], vertical_position, histogram[:, 1], 
                        alpha=0.3, color=colors[hist_index])
        
        # Add label
        offset_index = min(hist_index, len(vertical_offsets) - 1)
        plt.text(np.min(bins_array) + 0.1, 
                 vertical_position + vertical_offsets[offset_index] * gap, 
                 labels[hist_index], fontsize=13)
        
        vertical_position -= gap
    
    # Add vertical lines if specified
    if lines is not None:
        line_colors = ['r', 'g', 'orange', 'b', 'purple', 'c']
        for line_index, line in enumerate(lines):
            color_index = line_index % len(line_colors)
            plt.axvline(line[1], label=line[0], linestyle='--', color=line_colors[color_index])
    
    # Set x-axis limits if specified
    if xlim is not None:
        plt.xlim(xlim)
    
    # Finalize plot
    plt.legend(bbox_to_anchor=(0.01, hist_index/(2*num_histograms)), fontsize=20)
    plt.grid(linestyle='--', linewidth=0.8)
    plt.savefig(filename)
    plt.show()


def polar_density_plot(
    angles: np.ndarray,
    amplitudes: np.ndarray,
    title: str = '',
    bins: int = 100,
    cmap: str = 'twilight',
    save: bool = False,
    file: str = 'polarplot.png'
    ) -> None:
    """
    Creates a polar density plot to visualize pseudorotation angles.
    
    This function is particularly useful for visualizing nucleic acid
    conformations based on phase angle P and amplitude Vmax.
    
    Parameters
    ----------
    angles : np.ndarray
        Phase angles P in degrees.
    amplitudes : np.ndarray
        Corresponding Vmax amplitudes.
    title : str, optional
        Title of the plot.
    bins : int, optional
        Number of bins for the 2D histogram (currently unused).
    cmap : str, optional
        Color palette for visualization.
    save : bool, optional
        If True, saves the plot.
    file : str, optional
        Name of the file to save the plot.
    
    Returns
    -------
    None
        The function displays and saves the plot but doesn't return any value.
    
    Notes
    -----
    Edge effects may appear at the 0°-360° junction if the number of bins is small.
    """
    # Create figure
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.grid(True, linewidth=1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    # Add colored areas
    ax.bar(x=np.radians(18), height=60, width=np.radians(36), color="blue", alpha=0.25)  # Blue area
    ax.bar(x=np.radians(162), height=60, width=np.radians(36), color="red", alpha=0.25)  # Red area
    
    # Plot points
    ax.scatter(np.radians(angles), amplitudes)
    
    # Add conformation labels
    angle_positions = [18, 54, 90, 126, 162, 198, 234, 270, 306, 342]
    label_radius = 63  # Radial position for labels
    conformations = [
        "C3' endo", "C4' exo", "O4' endo", "C1' exo", "C2' endo",
        "C3' exo", "C4' endo", "O4' exo", "C1' endo", "C2' exo"
    ]
    
    for i, angle in enumerate(angle_positions):
        text_alignment = "left" if angle < 180 else "right"
        ax.text(x=np.radians(angle), y=label_radius, s=conformations[i],
                fontweight="bold", ha=text_alignment, fontsize=15)
    
    # Configure radial ticks
    ax.set_rmax(60)
    ax.set_rticks([0, 20, 40, 60], fontsize=15)
    ax.set_rlabel_position(108)
    ax.tick_params(axis="y", which="both", labelsize=15)
    
    # Configure angular ticks
    ax.set_xticks(np.radians(np.linspace(0, 360, 10, endpoint=False)), minor=False)
    plt.setp(ax.xaxis.get_majorticklabels()[1:5], ha="left")
    plt.setp(ax.xaxis.get_majorticklabels()[6:9], ha="right")
    
    # Remove duplicates from legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right", 
               markerscale=2, fontsize=10, bbox_to_anchor=(1.2, 1))
    
    # Finalize plot
    plt.title(title, fontsize=24)
    plt.suptitle("Pseudorotation Phase angles P(polar) and Amplitude Vmax (radius)")
    plt.tight_layout()
    
    if save:
        plt.savefig(file)
    
    plt.show()


def plot_fingerprint(
    combination: List[bool],
    title: str,
    long_names: Optional[List[str]] = None
    ) -> None:
    """
    Creates a fingerprint-like visualization for binary data.
    
    This function generates a heatmap that visually represents a set of
    binary data, such as molecular interactions or structural features.
    
    Parameters
    ----------
    combination : List[bool]
        Array of boolean values representing the features to visualize.
    title : str
        Title of the plot.
    long_names : Optional[List[str]], optional
        List of full names for labels. If not provided, generic labels
        will be used.
    
    Returns
    -------
    None
        The function displays the plot but doesn't return any value.
    """
    # Check that long_names is defined
    if long_names is None:
        long_names = [f"Feature_{i}" for i in range(len(combination))]
    
    # Create data for heatmap
    heatmap_data = []
    value = 4.0
    
    for feature_present in combination:
        if feature_present == False:
            heatmap_data.append([0.0, 0.0])
        else:
            heatmap_data.append([value, value])
        value += 1.0
    
    # Convert to DataFrame for easier visualization
    data_array = np.array(heatmap_data)
    data_transposed = data_array.transpose()
    feature_df = pd.DataFrame(data_transposed, columns=long_names)
    feature_df_transposed = feature_df.transpose()
    
    # Create figure
    _, ax = plt.subplots(figsize=(11, 4))
    
    # Define color palette
    cmap = sns.color_palette("cubehelix_r", as_cmap=True)
    
    # Create heatmap
    sns.heatmap(feature_df_transposed, cmap=cmap, vmin=0, vmax=18)
    
    # Finalize plot
    plt.title(title)
    plt.show()



def categorize_data_array(data_array: List[List[float]], feature_names: List[str]) -> np.ndarray:
    """
    Categorize data values into distinct groups based on predefined thresholds.
    
    This function processes a multi-feature data array and assigns category indices
    based on value ranges specific to each feature type.
    
    Parameters
    ----------
    data_array : List[List[float]]
        Input data organized as a list of lists, where each inner list represents 
        values for a specific feature across all samples.
    feature_names : List[str]
        Names of the features corresponding to each list in data_array.
        
    Returns
    -------
    np.ndarray
        Array of categorized values where each element represents the 
        category index for a specific feature and sample.
        
    Raises
    ------
    SystemExit
        If data_array is not a list.
    """
    # Validate input data structure
    if not isinstance(data_array, list):
        print("ERROR: data_array must be a list")
        sys.exit(1)
    
    # Get dimensions of the data array
    n_features, n_samples = np.shape(data_array)
    
    # Initialize output array (samples x features)
    category_matrix = np.zeros((n_samples, n_features))
    
    # Process first 7 features with common threshold logic
    for feature_idx in range(7):
        feature_values = data_array[feature_idx]
        
        # Ensure proper variable reference
        for sample_idx in range(len(feature_values)):
            current_value = feature_values[sample_idx]
            
            # Categorize based on thresholds
            if current_value <= 3.1:
                category_matrix[sample_idx, feature_idx] = 1
            elif 3.1 < current_value <= 4.1:
                category_matrix[sample_idx, feature_idx] = 2
            elif 4.1 < current_value <= 5.1:
                category_matrix[sample_idx, feature_idx] = 3
    
    # Process feature at index 7 (PU parameter)
    pu_idx = 7
    pu_values = data_array[pu_idx]
    for sample_idx in range(len(pu_values)):
        if 0 <= pu_values[sample_idx] <= 36:
            category_matrix[sample_idx, pu_idx] = 4
        elif 144 <= pu_values[sample_idx] <= 180:
            category_matrix[sample_idx, pu_idx] = 5
    
    # Process feature at index 8 (IAA parameter)
    iaa_idx = 8
    iaa_values = data_array[iaa_idx]
    for sample_idx in range(len(iaa_values)):
        if 140 <= iaa_values[sample_idx]:
            category_matrix[sample_idx, iaa_idx] = 6
        elif 125 <= iaa_values[sample_idx] < 140:
            category_matrix[sample_idx, iaa_idx] = 7
    
    return category_matrix


def cluster_data_combinations(category_matrix: np.ndarray) -> np.ndarray:
    """
    Identify unique combinations of categories and group samples accordingly.
    
    This function analyzes a categorized data matrix and groups samples that share
    identical category patterns across all features.
    
    Parameters
    ----------
    category_matrix : np.ndarray
        Matrix of categorized values where rows represent samples and columns represent features.
        
    Returns
    -------
    np.ndarray
        Array of objects where each element contains:
        - A unique category combination
        - The count of samples with this combination
        - List of sample indices with this combination
    """
    unique_combinations = []
    combination_clusters = []
    
    n_samples, n_features = np.shape(category_matrix)
    
    # Iterate through all samples to find unique category combinations
    for sample_idx in range(n_samples):
        current_combination = list(category_matrix[sample_idx])
        
        if current_combination not in unique_combinations:
            # Add new combination to the list
            unique_combinations.append(current_combination)
            combination_clusters.append([current_combination, 1, [sample_idx]])
        else:
            # Find the existing combination and update its count
            found = False
            cluster_idx = 0
            
            while not found and cluster_idx < len(combination_clusters):
                if current_combination == list(combination_clusters[cluster_idx][0]):
                    combination_clusters[cluster_idx][1] += 1
                    combination_clusters[cluster_idx][2].append(sample_idx)
                    found = True
                cluster_idx += 1
    
    print(f"Analysis complete: found {len(unique_combinations)} unique combinations")
    
    # Convert to numpy array for consistency
    result = np.asarray(combination_clusters, dtype="object")
    return result


def generate_colored_dataframe(
    combinations_df: pd.DataFrame, 
    feature_names: List[str], 
    total_frames: int, 
    cutoff: float = 0.002
    ) -> Tuple[pd.DataFrame, List[int], int]:
    """
    Generate a colored dataframe representation of feature combinations.
    
    This function creates a visualization-ready dataframe where each combination
    of feature categories is represented with appropriate coloring.
    
    Parameters
    ----------
    combinations_df : pd.DataFrame
        DataFrame containing combination data with columns for combinations and counts.
    feature_names : List[str]
        Names of the features to be used as column labels.
    total_frames : int
        Total number of data frames/samples.
    cutoff : float, optional
        Minimum ratio of samples required to include a combination, by default 0.002.
        
    Returns
    -------
    Tuple[pd.DataFrame, List[int], int]
        - Transposed colored dataframe
        - List of x-axis positions
        - Count of combinations included
    """
    # Initialize array to hold the visualization data
    visualization_array = np.zeros((total_frames, len(feature_names)))
    x_positions = []
    
    current_position = 0
    combination_count = 0
    
    # Process combinations until reaching cutoff threshold
    while True:
        if combination_count >= len(combinations_df):
            break
            
        # Get current combination data
        count = combinations_df.iloc[combination_count, 1]
        combination = combinations_df.iloc[combination_count, 0]
        ratio = count / total_frames
        
        # Stop if below cutoff threshold
        if ratio < cutoff:
            break
            
        x_positions.append(current_position)
        
        # Fill visualization array with category values
        for feature_idx in range(len(combination)):
            for sample_idx in range(count):
                visualization_array[current_position + sample_idx, feature_idx] = combination[feature_idx]
        
        current_position += count
        combination_count += 1
    
    # Create and transpose the dataframe
    colored_df = pd.DataFrame(visualization_array, columns=feature_names)
    colored_df_transposed = colored_df.transpose()
    
    return colored_df_transposed, x_positions, combination_count



def calculate_distribution_entropy(combinations_df: pd.DataFrame, total_samples: int) -> float:
    """
    Calculate the Shannon entropy of a distribution of combinations.
    
    This function computes the entropy of the distribution of different combinations
    in the dataset, which quantifies the diversity or uncertainty of the dataset.
    Higher entropy indicates more diversity in the combinations.
    
    Parameters
    ----------
    combinations_df : pd.DataFrame
        DataFrame containing combination data with each row representing a unique
        combination and the second column containing the count of samples with that combination.
    total_samples : int
        Total number of samples/frames in the dataset.
        
    Returns
    -------
    float
        The calculated Shannon entropy value in bits (log base 2).
    """
    # Calculate the probability of each combination class
    class_probabilities = []
    for idx in range(len(combinations_df)):
        # Extract the count of samples with this combination
        count = combinations_df.iloc[idx, 1]
        # Calculate the probability as the proportion of total samples
        class_probabilities.append(count / total_samples)
    
    # Calculate Shannon entropy using the formula: -sum(p * log2(p))
    entropy = -sum(p * np.log2(p) for p in class_probabilities if p > 0)
    
    print(f"Shannon entropy: {entropy:.4f} bits")
    return entropy




def calculate_state_percentage(data_series: pd.Series, unique_states: list) -> pd.Series:
    """
    Calculate the percentage of occurrences for each unique state in the given series.
    
    Args:
        data_series (pd.Series): A Pandas Series containing categorical data.
        unique_states (list): A list of unique states to ensure all are represented.
    
    Returns:
        pd.Series: A Pandas Series containing the percentage of each unique state.
    """
    state_counts = data_series.value_counts()
    
    # Ensure all unique states are represented
    for state in unique_states:
        if state not in state_counts:
            state_counts[state] = 0
    
    # Calculate percentage
    state_percentages = (state_counts / len(data_series)) * 100
    
    return state_percentages