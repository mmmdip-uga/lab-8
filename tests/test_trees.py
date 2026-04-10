import pytest
import math
from src.cart import gini_impurity, CARTTree
from src.random_forest import bootstrap_sample, RandomForest

# ==========================================
# CART Implementation Tests (70 Points Total)
# ==========================================

# Test 1: 5 Points
def test_gini_impurity_pure():
    """A pure node should have a Gini impurity of 0.0"""
    labels = ['DrugA', 'DrugA', 'DrugA']
    assert math.isclose(gini_impurity(labels), 0.0)

# Test 2: 10 Points
def test_gini_impurity_mixed():
    """A perfectly mixed binary node should have a Gini impurity of 0.5"""
    labels = ['DrugA', 'DrugB']
    assert math.isclose(gini_impurity(labels), 0.5)

# Test 3: 15 Points
def test_cart_binary_split():
    """CART must strictly use binary splits (true/false branches), not multi-way branches."""
    X = [{'BP': 'HIGH'}, {'BP': 'NORMAL'}, {'BP': 'LOW'}]
    y = ['A', 'B', 'B']
    
    tree = CARTTree()
    tree.fit(X, y)
    
    assert isinstance(tree.root, CARTTree._BinarySplit), "Root should be a BinarySplit node"
    assert hasattr(tree.root, 'true_branch'), "Node must have a true_branch"
    assert hasattr(tree.root, 'false_branch'), "Node must have a false_branch"
    assert not hasattr(tree.root, 'branches'), "'branches' dict is forbidden in CART"

# Test 4: 10 Points
def test_cart_max_depth():
    """Tree should stop splitting when max_depth is reached."""
    X = [{'F1': 'A', 'F2': 'X'}, {'F1': 'B', 'F2': 'Y'}, {'F1': 'A', 'F2': 'Y'}]
    y = ['1', '2', '3']
    
    tree = CARTTree(max_depth=1)
    tree.fit(X, y)
    
    # The root is depth 0. Its branches should be forced into leaves.
    assert isinstance(tree.root.true_branch, CARTTree._Leaf)
    assert isinstance(tree.root.false_branch, CARTTree._Leaf)

# Test 5: 15 Points
def test_cart_min_samples_leaf():
    """Tree should refuse to split if a resulting child node has fewer than min_samples_leaf."""
    # Splitting on 'F1' == 'B' leaves 1 sample in the True branch and 2 in the False branch.
    X = [{'F1': 'A'}, {'F1': 'A'}, {'F1': 'B'}]
    y = ['Y', 'Y', 'N']
    
    # Enforce minimum 2 samples per leaf
    tree = CARTTree(min_samples_leaf=2)
    tree.fit(X, y)
    
    # The split should be aborted, resulting in a single leaf node.
    assert isinstance(tree.root, CARTTree._Leaf), "Split should be aborted due to min_samples_leaf"
    assert tree.root.value == 'Y', "Majority vote should be 'Y'"

# Test 6: 15 Points
def test_cart_min_impurity_decrease():
    """Tree should refuse to split if the Gini decrease is too small."""
    # This dataset has zero correlation between feature and label. Gini decrease will be 0.
    X = [{'F': 'A'}, {'F': 'A'}, {'F': 'B'}, {'F': 'B'}]
    y = ['Y', 'N', 'Y', 'N']
    
    # Require at least a 0.1 drop in impurity
    tree = CARTTree(min_impurity_decrease=0.1)
    tree.fit(X, y)
    
    assert isinstance(tree.root, CARTTree._Leaf), "Split should be aborted due to low impurity decrease"

# ==========================================
# Random Forest Tests (30 Points Total)
# ==========================================

# Test 7: 5 Points
def test_bootstrap_sample_length():
    """Bootstrap sample must contain the exact same number of rows as the original data."""
    X = [{'id': i} for i in range(10)]
    y = [i for i in range(10)]
    
    X_sample, y_sample = bootstrap_sample(X, y)
    
    assert len(X_sample) == 10
    assert len(y_sample) == 10

# Test 8: 10 Points
def test_bootstrap_sample_randomness():
    """Bootstrap sample must sample WITH replacement (duplicates expected)."""
    X = [{'id': i} for i in range(50)]
    y = [i for i in range(50)]
    
    X_sample, _ = bootstrap_sample(X, y)
    unique_ids = set([x['id'] for x in X_sample])
    
    # In 50 random samples with replacement, getting all 50 unique is mathematically improbable.
    assert len(unique_ids) < 50, "Sample should contain duplicates"

# Test 9: 5 Points
def test_random_forest_initialization():
    """Random Forest should store variables correctly upon initialization."""
    rf = RandomForest(n_trees=7, max_depth=3)
    assert rf.n_trees == 7
    assert rf.max_depth == 3
    assert isinstance(rf.trees, list)

# Test 10: 10 Points
def test_random_forest_e2e_training():
    """Random Forest should successfully train multiple trees and execute majority voting."""
    X = [{'F1': 'A', 'F2': 'X'}, {'F1': 'B', 'F2': 'Y'}, {'F1': 'A', 'F2': 'Y'}, {'F1': 'B', 'F2': 'X'}]
    y = ['1', '2', '1', '2']
    
    rf = RandomForest(n_trees=3)
    rf.fit(X, y)
    
    assert len(rf.trees) == 3, "Forest must build exactly n_trees"
    
    # Test a prediction
    pred = rf.predict({'F1': 'A', 'F2': 'X'})
    assert pred in ['1', '2'], "Prediction must return a valid class label"
