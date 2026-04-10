```markdown
# Lab Assignment: From ID3 to CART and Random Forests

Welcome to the Decision Trees and Ensemble Learning lab! In this assignment, you will bridge the gap between basic decision tree concepts and modern ensemble machine learning techniques. 

You are provided with a working implementation of the **ID3** algorithm. Your goal is to build upon this logic to implement the **CART** (Classification and Regression Trees) algorithm, and finally, combine multiple CART trees to create a **Random Forest**.

## Setup Instructions

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. You will be using the `data/drug200.csv` dataset to test your models.

---

## Part 1: Implementing CART (`src/cart.py`)

The provided `id3.py` file uses Shannon Entropy and creates multi-way branches. CART differs in two major ways: it uses **Gini Impurity** and strictly restricts the tree to **Binary Splits**.

### 1. Gini Impurity
You must implement the Gini impurity calculation. Gini measures how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.

The formula for Gini Impurity is:
$$Gini(S) = 1 - \sum_{i=1}^{C} p_i^2$$
Where $p_i$ is the probability of an item belonging to class $i$.

### 2. Binary Splits
Instead of creating a branch for every unique value in a categorical feature, CART creates a True/False split. For example, instead of branching into *High*, *Normal*, and *Low* for Blood Pressure, the tree evaluates: `BP == HIGH`.

### 3. Pre-pruning (Stopping Criteria)
To prevent overfitting, implement the following criteria in your recursive `_build_tree` method:
* `max_depth`: Stop growing the tree if this depth is reached.
* `min_samples_split`: Do not split a node if it contains fewer than this number of samples.
* `min_samples_leaf`: Cancel a split if it results in a child node with fewer than this number of samples.
* `min_impurity_decrease`: Cancel a split if the drop in Gini impurity is less than this threshold.

---

## Part 2: Building a Random Forest (`src/random_forest.py`)

Decision trees are prone to overfitting. You will build a Random Forest to reduce variance using an ensemble of your CART models.

### 1. Bootstrapping (Bagging)
Implement the `bootstrap_sample` function. For a dataset of size $N$, you must randomly draw $N$ samples **with replacement**. Some original rows will appear multiple times, while others will be left out (Out-Of-Bag).

### 2. Feature Subsampling
When building each tree in the forest, do not evaluate all available attributes. Instead, limit the tree to a random subset of features at each split. Use the square root rule:
$$m = \lfloor \sqrt{M} \rfloor$$
Where $M$ is the total number of features and $m$ is the subset size.

### 3. Majority Voting
In your `predict` method, query every tree in your forest and return the most frequently predicted class.

---

## Part 3: Testing Your Code

You can test your implementation locally using Pytest. The autograder will run these exact same tests.

Run all tests:
```bash
pytest tests/test_trees.py
```

Run only CART tests:
```bash
pytest tests/test_trees.py -k "cart"
```

---

## Part 4: Comparative Analysis (Write-Up Submission)

Individual file submission on eLC for this assignment. The file should include the Results and the answers to the discussion questions as described in the following.

**Instructions:** Run the provided `id3.py` (excluding numerical features like Age and Na_to_K), your completed `cart.py`, and your completed `random_forest.py` on the `drug200.csv` dataset. Record your results and observations below.

### Results

* **ID3 Accuracy (Categorical Only):** [Enter percentage]
* **CART Accuracy:** [Enter percentage]
* **Random Forest Accuracy:** [Enter percentage]

### Discussion Questions
Answer the following questions and include them in your report.

**1. How did the structure of your CART tree differ from the ID3 tree?**

**2. Which pre-pruning parameter in CART had the biggest impact on preventing overfitting in your tests, and why?**

**3. Did the Random Forest outperform the single CART tree? Explain why the bootstrapping and feature subsampling steps contribute to this difference.**
```
