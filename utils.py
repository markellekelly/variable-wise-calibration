from sklearn import tree
import numpy as np

def get_lineage(tree, feature_names):
    '''
    Parse a sklearn tree and return the rules corresponding to the leaf nodes
    '''
    if tree.tree_.node_count == 1:
        return []
    left = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] if i != -2 else -2 for i in tree.tree_.feature]

    idx = np.argwhere(left == -1)[:,0]     

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    boxes = []
    new_box = []
    for child in idx:
        for node in recurse(left, right, child):
            if type(node) == np.int64:
                boxes.append(new_box)
                new_box = []
            else:
                new_box.append(node)
    return boxes

def get_splits(X_train, X_test, y_train, y_test, max_depth=2, min_samples_leaf=10):
    '''
    Given a train and test set, generate decision trees for each variable
        and return the boxes corresponding to the leaf nodes
    '''
    boxes = []
    for var in X_train.columns:
        sub_train = X_train[[var]]; sub_test = X_test[[var]]
        clf = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        clf = clf.fit(sub_train, y_train)
        clf.score(sub_test, y_test)
        boxes.extend(get_lineage(clf, [var]))
    
    s = set(np.array([item for sublist in boxes for item in sublist]).flatten())
    s.remove('l'); s.remove('r'); s.remove(var)
    return sorted([float(i) for i in s])

def bootstrap_ci(data, n=1000, func=np.mean, p=0.95):
    '''
    bootstrap confidence intervals for binned plots
    '''

    sample_size = len(data)
    simulations = [func(np.random.choice(data, size=sample_size, replace=True)) for i in range(n)]
    simulations.sort()
    u_pval = (1+p)/2.
    l_pval = (1-u_pval)
    l_indx = int(np.floor(n*l_pval))
    u_indx = int(np.floor(n*u_pval))
    
    return(simulations[l_indx],simulations[u_indx])