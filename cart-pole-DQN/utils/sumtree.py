import numpy as np

class SumTree:
    """
    SumTree data structure for efficient prioritized sampling.
    
    A SumTree is a binary tree where each parent node holds the sum of its children.
    Leaf nodes store the actual priority values.
    This allows for O(log N) updates and O(log N) sampling based on priority probability.
    
    The tree is implemented using a fixed-size numpy array for efficiency.
    For capacity N, the tree needs an array of size 2*N - 1.
    - Indices 0 to capacity-2 are parent nodes.
    - Indices capacity-1 to 2*capacity-2 are leaf nodes (storing priorities).
    """
    
    def __init__(self, capacity):
        """
        Initialize SumTree.
        
        Args:
            capacity (int): Maximum number of experiences (leaf nodes) the tree can hold.
        """
        self.capacity = capacity
        # Tree array stores priority sums. Size is 2*N - 1 for a complete binary tree.
        self.tree = np.zeros(2 * capacity - 1)
        # Data array stores the actual experience objects (transitions).
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0       # Pointer to the next location to write data (circular buffer)
        self.n_entries = 0   # Current number of elements in the tree
    
    def _propagate(self, idx, change):
        """
        Recursively propagate priority change up to the root.
        
        Args:
            idx (int): Current node index.
            change (float): The amount by which the priority changed.
        """
        parent = (idx - 1) // 2
        
        self.tree[parent] += change
        
        # Keep propagating until we reach the root (index 0)
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """
        Recursively search for the leaf node corresponding to priority value 's'.
        
        Args:
            idx (int): Current node index (start with root=0).
            s (float): The priority value we are searching for.
            
        Returns:
            int: The index of the leaf node.
        """
        left = 2 * idx + 1
        right = left + 1
        
        # If we have reached a leaf node (left child index is out of bounds)
        if left >= len(self.tree):
            return idx
        
        # If the value sought is less than the left child's sum, go left.
        # Otherwise, subtract left child's sum and go right.
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """
        Get the total sum of priorities (stored at the root).
        """
        return self.tree[0]
    
    def add(self, priority, data):
        """
        Add a new experience with a given priority.
        
        Args:
            priority (float): Initial priority of the experience.
            data (object): The experience tuple (state, action, ...).
        """
        # Calculate the index in the tree array for the new leaf node
        idx = self.write + self.capacity - 1
        
        # Store data in the separate data array
        self.data[self.write] = data
        
        # Update the tree with the new priority
        self.update(idx, priority)
        
        # Advance the write pointer (circular buffer)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        # Keep track of total entries (up to capacity)
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        """
        Update the priority of an existing leaf node.
        
        Args:
            idx (int): The index of the leaf node in the 'tree' array.
            priority (float): The new priority value.
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        # Propagate the change up to the root to maintain correct sums
        self._propagate(idx, change)
    
    def get(self, s):
        """
        Get the data and priority for a given cumulative priority value 's'.
        
        Args:
            s (float): A value between 0 and total_priority to sample.
            
        Returns:
            tuple: (leaf_index, priority_value, experience_data)
        """
        # Find the leaf index corresponding to value 's'
        idx = self._retrieve(0, s)
        # Calculate corresponding index in the data array
        data_idx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[data_idx])