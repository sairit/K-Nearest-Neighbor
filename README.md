# K-Nearest Neighbors Implementation from Scratch

**Author:** Sai Yadavalli  
**Version:** 2.1

A comprehensive implementation of the K-Nearest Neighbors algorithm with automated hyperparameter optimization, k-fold cross-validation, and extensive performance analysis capabilities, built from scratch using NumPy and pandas.

## Overview

This project implements the K-Nearest Neighbors (KNN) classification algorithm without relying on scikit-learn or other machine learning frameworks, demonstrating mastery of distance-based learning, statistical validation, and hyperparameter optimization techniques. The implementation features automated k-value selection, comprehensive cross-validation, and detailed performance analytics.

## Mathematical Foundation

### Distance-Based Classification
KNN operates on the principle that similar instances have similar classifications, using distance metrics to identify nearest neighbors:

```
d(x, y) = √(Σᵢ(xᵢ - yᵢ)²)
```

Where the Euclidean distance measures similarity between feature vectors in n-dimensional space.

### Classification Decision Rule
The algorithm assigns class labels based on majority voting among k nearest neighbors:

```
ŷ = argmax(Σᵢ∈Nₖ(x) I(yᵢ = c))
```

Where:
- `Nₖ(x)` represents the k nearest neighbors of point x
- `I(·)` is the indicator function
- `c` represents possible class labels

### Hyperparameter Optimization
The implementation systematically evaluates different k values to find the optimal balance between bias and variance:
- **Low k**: High variance, sensitive to noise
- **High k**: High bias, overly smooth decision boundaries
- **Optimal k**: Minimizes generalization error through cross-validation

## Features

- **Pure NumPy/Pandas Implementation**: No external ML libraries required
- **Automated K-Value Optimization**: Systematic evaluation of k from 1 to specified maximum
- **5-Fold Cross-Validation**: Robust statistical validation with automated data splitting
- **Comprehensive Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- **Data Standardization**: Optional Z-score normalization for improved distance calculations
- **Performance Visualization**: Automated plotting of k vs. F1-Score relationships
- **Error Analysis**: Detailed error tables across different k values and folds
- **File Management**: Automated directory creation and CSV handling
- **Interactive Interface**: User-friendly command-line interaction

## Key Components

### Core Algorithm Methods

#### `distance(a, b, n)` - Euclidean Distance Computation
Implements the fundamental distance metric for neighbor identification:
- Computes element-wise squared differences
- Applies matrix operations for vectorized efficiency
- Returns scalar distance values for ranking

#### `count_neighbors(k, sdf)` - Majority Voting Classification
Determines class labels through democratic neighbor voting:
- Counts occurrences of each class among k nearest neighbors
- Implements majority rule decision making
- Handles ties through deterministic selection

#### `makeA(df, col)` - Feature Matrix Construction
Transforms DataFrames into NumPy arrays suitable for distance calculations:
- Removes target variable columns
- Converts to numerical matrix format
- Maintains feature ordering consistency

### Data Processing Pipeline

#### `split_data()` - K-Fold Cross-Validation Setup
Implements stratified 5-fold cross-validation:
- **Random Sampling**: Each fold contains 20% of data
- **Complementary Training**: Remaining 80% forms training set
- **File Persistence**: Saves splits as CSV files for reproducibility
- **Statistical Validity**: Maintains data distribution across folds

#### `standardize(df, label)` - Feature Scaling
Applies Z-score normalization to improve distance metric reliability:
```
x_scaled = (x - μ) / σ
```
- **Mean Centering**: Removes feature-specific biases
- **Variance Scaling**: Equalizes feature importance in distance calculations
- **Label Preservation**: Maintains target variable integrity

### Performance Analysis

#### `testing(positive_class, test, train, standard=False)` - Model Evaluation
Comprehensive performance assessment including:
- **Confusion Matrix**: True/False Positive/Negative counts
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Error Counting**: Misclassification tracking
- **Statistical Reporting**: Detailed performance breakdown

#### `training(standard=False)` - Hyperparameter Optimization
Automated k-value selection through systematic evaluation:
- Iterates through k values from 1 to specified maximum
- Performs cross-validation for each k value
- Generates performance visualization
- Identifies optimal k through F1-Score maximization

#### `error_table(standard=False)` - Comprehensive Error Analysis
Creates detailed error analysis across k values and folds:
- **Fold-wise Errors**: Individual fold performance tracking
- **K-value Comparison**: Systematic hyperparameter analysis
- **Total Error Computation**: Aggregate performance metrics
- **CSV Export**: Persistent storage of analysis results

## Technical Implementation

### Algorithmic Efficiency

#### Distance Computation Optimization
- **Vectorized Operations**: NumPy broadcasting for efficient calculations
- **Memory Management**: Optimized array operations
- **Numerical Stability**: Robust handling of floating-point arithmetic

#### Data Structure Design
- **Matrix Operations**: Efficient nearest neighbor identification
- **Sorting Algorithms**: Fast neighbor ranking using pandas sort_values
- **Index Management**: Proper handling of DataFrame indices during operations

### Statistical Rigor

#### Cross-Validation Implementation
- **Stratified Sampling**: Maintains class distribution across folds
- **Statistical Independence**: Proper train/validation separation
- **Reproducibility**: Consistent random seed handling

#### Performance Metrics
- **Balanced Evaluation**: Multiple complementary metrics
- **Class-specific Analysis**: Detailed breakdown by classification categories
- **Statistical Significance**: Cross-fold variance assessment

## Usage

### Basic Classification
```python
# Initialize model
model = KNearestNeighbor(k=5, featureB="target_column")

# Load and split data
model.load_data()
model.split_data()

# Perform single test
results = model.testing("positive_class", "test.csv", "train.csv", standard=True)
```

### Hyperparameter Optimization
```python
# Optimize k value through cross-validation
model.training(standard=True)

# Generate comprehensive error analysis
error_table = model.error_table(standard=True)
```

### Interactive Analysis
```python
# Complete analysis pipeline
model = KNearestNeighbor(max_k=10, target="Diagnosis")
model.split_data()           # Create cross-validation folds
model.training()             # Find optimal k
model.error_table()          # Detailed error analysis
```

## Cross-Validation Results

### K-Value Optimization
The implementation provides systematic k-value analysis:
- **Performance Curves**: F1-Score vs. k visualization
- **Optimal Selection**: Data-driven hyperparameter choice
- **Variance Analysis**: Cross-fold performance stability

### Statistical Validation
- **5-Fold Cross-Validation**: Robust performance estimation
- **Error Distribution**: Fold-wise performance variation
- **Confidence Assessment**: Statistical reliability metrics

## Performance Monitoring

### Visualization Capabilities
- **K vs. F1-Score Plots**: Hyperparameter optimization curves
- **Error Distribution Charts**: Performance analysis across folds
- **Confusion Matrix Display**: Detailed classification breakdown

### File Output Management
- **Automated Directory Creation**: Organized file structure
- **CSV Export**: Persistent storage of results
- **Figure Generation**: Publication-quality visualizations

## Requirements

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.3.0
```

## Educational Value

This implementation demonstrates expertise in:

### Machine Learning Concepts
- **Instance-Based Learning**: Non-parametric classification methods
- **Hyperparameter Optimization**: Systematic model selection
- **Cross-Validation**: Statistical model evaluation techniques
- **Distance Metrics**: Geometric similarity measures

### Statistical Analysis
- **Performance Metrics**: Comprehensive evaluation methodologies
- **Bias-Variance Tradeoff**: Understanding of model complexity effects
- **Statistical Significance**: Cross-validation and confidence assessment
- **Data Preprocessing**: Feature scaling and normalization

### Software Engineering
- **Object-Oriented Design**: Clean class structure and encapsulation
- **File System Management**: Automated directory and file handling
- **Interactive Programming**: User-friendly command-line interfaces
- **Data Pipeline**: End-to-end machine learning workflow

### Mathematical Implementation
- **Linear Algebra**: Vector operations and distance calculations
- **Algorithmic Thinking**: Efficient nearest neighbor identification
- **Numerical Methods**: Stable floating-point computations
- **Optimization Theory**: Systematic hyperparameter search

## Algorithmic Complexity

### Time Complexity
- **Training**: O(1) - Lazy learning approach
- **Prediction**: O(n·d·log(n)) where n=training examples, d=dimensions
- **Cross-Validation**: O(k·f·n·d·log(n)) where k=max k-value, f=folds

### Space Complexity
- **Storage**: O(n·d) for training data
- **Distance Matrix**: O(n) for neighbor distances
- **Temporary Storage**: O(k) for neighbor selection

## Future Enhancements

- [ ] Alternative distance metrics (Manhattan, Minkowski, Cosine)
- [ ] Weighted voting schemes (distance-weighted, inverse-distance)
- [ ] Dimensionality reduction integration (PCA, LDA)
- [ ] Approximate nearest neighbor algorithms for scalability
- [ ] Multi-class classification with probabilistic outputs
- [ ] Feature selection and importance analysis
- [ ] Parallel processing for large datasets

---

This implementation showcases a thorough understanding of distance-based learning algorithms, statistical validation techniques, and the practical considerations involved in building robust machine learning systems from first principles.
