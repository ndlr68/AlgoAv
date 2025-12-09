# tp2_tree_operations.py
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import time

# =============================================================================
# AVL Tree Implementation with Full Operations
# =============================================================================

class AVLNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1  # Hauteur commence à 1 pour un nœud seul

def get_height(node):
    if not node:
        return 0
    return node.height

def get_balance(node):
    if not node:
        return 0
    return get_height(node.left) - get_height(node.right)

def update_height(node):
    if node:
        node.height = 1 + max(get_height(node.left), get_height(node.right))

def right_rotate(y):
    x = y.left
    T2 = x.right

    x.right = y
    y.left = T2

    update_height(y)
    update_height(x)

    return x

def left_rotate(x):
    y = x.right
    T2 = y.left

    y.left = x
    x.right = T2

    update_height(x)
    update_height(y)

    return y

def insert_avl(node, value):
    if not node:
        return AVLNode(value)

    if value < node.value:
        node.left = insert_avl(node.left, value)
    elif value > node.value:
        node.right = insert_avl(node.right, value)
    else:
        return node  # Pas de doublons

    update_height(node)
    balance = get_balance(node)

    # Left Left Case
    if balance > 1 and value < node.left.value:
        return right_rotate(node)

    # Right Right Case
    if balance < -1 and value > node.right.value:
        return left_rotate(node)

    # Left Right Case
    if balance > 1 and value > node.left.value:
        node.left = left_rotate(node.left)
        return right_rotate(node)

    # Right Left Case
    if balance < -1 and value < node.right.value:
        node.right = right_rotate(node.right)
        return left_rotate(node)

    return node

def find_min_node(node):
    current = node
    while current.left:
        current = current.left
    return current

def delete_avl(node, value):
    if not node:
        return node

    if value < node.value:
        node.left = delete_avl(node.left, value)
    elif value > node.value:
        node.right = delete_avl(node.right, value)
    else:
        # Nœud avec un seul enfant ou aucun enfant
        if not node.left:
            return node.right
        elif not node.right:
            return node.left

        # Nœud avec deux enfants
        temp = find_min_node(node.right)
        node.value = temp.value
        node.right = delete_avl(node.right, temp.value)

    update_height(node)
    balance = get_balance(node)

    # Rééquilibrage
    # Left Left
    if balance > 1 and get_balance(node.left) >= 0:
        return right_rotate(node)

    # Left Right
    if balance > 1 and get_balance(node.left) < 0:
        node.left = left_rotate(node.left)
        return right_rotate(node)

    # Right Right
    if balance < -1 and get_balance(node.right) <= 0:
        return left_rotate(node)

    # Right Left
    if balance < -1 and get_balance(node.right) > 0:
        node.right = right_rotate(node.right)
        return left_rotate(node)

    return node

def search_avl(node, value):
    if not node:
        return False

    if node.value == value:
        return True
    elif value < node.value:
        return search_avl(node.left, value)
    else:
        return search_avl(node.right, value)

def get_depth_avl(node):
    """Retourne la profondeur de l'arbre (commence à 0)"""
    if not node:
        return -1  # Arbre vide = profondeur -1
    return node.height - 1  # Convertit hauteur en profondeur

def build_avl(values):
    root = None
    for value in values:
        root = insert_avl(root, value)
    return root

# =============================================================================
# TAS Max Implementation with Full Operations
# =============================================================================

class MaxHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def insert(self, value):
        # Vérifier si la valeur existe déjà
        if value in self.heap:
            return False  # Doublon détecté
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
        return True  # Insertion réussie

    def _heapify_up(self, i):
        while i > 0 and self.heap[self.parent(i)] < self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = (
                self.heap[self.parent(i)],
                self.heap[i],
            )
            i = self.parent(i)

    def extract_max(self):
        if not self.heap:
            return None

        max_val = self.heap[0]
        last_val = self.heap.pop()

        if self.heap:
            self.heap[0] = last_val
            self._heapify_down(0)

        return max_val

    def _heapify_down(self, i):
        max_index = i
        left = self.left_child(i)
        right = self.right_child(i)

        if left < len(self.heap) and self.heap[left] > self.heap[max_index]:
            max_index = left

        if right < len(self.heap) and self.heap[right] > self.heap[max_index]:
            max_index = right

        if i != max_index:
            self.heap[i], self.heap[max_index] = self.heap[max_index], self.heap[i]
            self._heapify_down(max_index)

    def delete_value(self, value):
        if value not in self.heap:
            return False

        index = self.heap.index(value)

        # Remplacer par le dernier élément
        self.heap[index] = self.heap[-1]
        self.heap.pop()

        # Réorganiser le tas
        if index < len(self.heap):
            self._heapify_down(index)
            self._heapify_up(index)

        return True

    def delete_batch(self, values):
        """Supprime plusieurs valeurs en une seule opération"""
        deleted_count = 0
        not_found_count = 0

        for value in values:
            if self.delete_value(value):
                deleted_count += 1
            else:
                not_found_count += 1

        return deleted_count, not_found_count

    def search(self, value):
        return value in self.heap

    def get_max(self):
        return self.heap[0] if self.heap else None

    def get_depth(self):
        """Retourne la profondeur du tas (commence à 0)"""
        if not self.heap:
            return -1  # Tas vide = profondeur -1

        n = len(self.heap)
        depth = 0
        level_size = 1

        while level_size <= n:
            depth += 1
            level_size *= 2

        return depth - 1  # Commence à 0

    def build_heap(self, values):
        self.heap = []
        unique_values = []
        for value in values:
            if value not in unique_values:  # Éviter les doublons
                unique_values.append(value)
                self.insert(value)
        return self

    def get_root_node(self):
        if not self.heap:
            return None

        class HeapNode:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None

        nodes = [HeapNode(val) for val in self.heap]

        for i in range(len(nodes)):
            left_idx = self.left_child(i)
            right_idx = self.right_child(i)

            if left_idx < len(nodes):
                nodes[i].left = nodes[left_idx]
            if right_idx < len(nodes):
                nodes[i].right = nodes[right_idx]

        return nodes[0] if nodes else None

# =============================================================================
# TAS Min Implementation with Full Operations
# =============================================================================

class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def insert(self, value):
        # Vérifier si la valeur existe déjà
        if value in self.heap:
            return False  # Doublon détecté
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
        return True  # Insertion réussie

    def _heapify_up(self, i):
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = (
                self.heap[self.parent(i)],
                self.heap[i],
            )
            i = self.parent(i)

    def extract_min(self):
        if not self.heap:
            return None

        min_val = self.heap[0]
        last_val = self.heap.pop()

        if self.heap:
            self.heap[0] = last_val
            self._heapify_down(0)

        return min_val

    def _heapify_down(self, i):
        min_index = i
        left = self.left_child(i)
        right = self.right_child(i)

        if left < len(self.heap) and self.heap[left] < self.heap[min_index]:
            min_index = left

        if right < len(self.heap) and self.heap[right] < self.heap[min_index]:
            min_index = right

        if i != min_index:
            self.heap[i], self.heap[min_index] = self.heap[min_index], self.heap[i]
            self._heapify_down(min_index)

    def delete_value(self, value):
        if value not in self.heap:
            return False

        index = self.heap.index(value)

        # Remplacer par le dernier élément
        self.heap[index] = self.heap[-1]
        self.heap.pop()

        # Réorganiser le tas
        if index < len(self.heap):
            self._heapify_down(index)
            self._heapify_up(index)

        return True

    def delete_batch(self, values):
        """Supprime plusieurs valeurs en une seule opération"""
        deleted_count = 0
        not_found_count = 0

        for value in values:
            if self.delete_value(value):
                deleted_count += 1
            else:
                not_found_count += 1

        return deleted_count, not_found_count

    def search(self, value):
        return value in self.heap

    def get_min(self):
        return self.heap[0] if self.heap else None

    def get_depth(self):
        """Retourne la profondeur du tas (commence à 0)"""
        if not self.heap:
            return -1  # Tas vide = profondeur -1

        n = len(self.heap)
        depth = 0
        level_size = 1

        while level_size <= n:
            depth += 1
            level_size *= 2

        return depth - 1  # Commence à 0

    def build_heap(self, values):
        self.heap = []
        unique_values = []
        for value in values:
            if value not in unique_values:  # Éviter les doublons
                unique_values.append(value)
                self.insert(value)
        return self

    def get_root_node(self):
        if not self.heap:
            return None

        class HeapNode:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None

        nodes = [HeapNode(val) for val in self.heap]

        for i in range(len(nodes)):
            left_idx = self.left_child(i)
            right_idx = self.right_child(i)

            if left_idx < len(nodes):
                nodes[i].left = nodes[left_idx]
            if right_idx < len(nodes):
                nodes[i].right = nodes[right_idx]

        return nodes[0] if nodes else None

# =============================================================================
# M-FUNCTIONAL HEAP Implementation (Tas M-fonctionnel)
# =============================================================================

class MFunctionalMaxHeap:
    def __init__(self, m=3):
        self.heap = []
        self.m = m  # Nombre d'enfants par nœud

    def parent(self, i):
        return (i - 1) // self.m

    def child_indices(self, i):
        """Retourne les indices des enfants du nœud i"""
        start = self.m * i + 1
        return list(range(start, start + self.m))

    def insert(self, value):
        """Insère une valeur dans le tas m-fonctionnel max"""
        if value in self.heap:
            return False  # Doublon détecté
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
        return True

    def _heapify_up(self, i):
        """Réorganise le tas vers le haut pour maintenir la propriété de tas max"""
        while i > 0 and self.heap[self.parent(i)] < self.heap[i]:
            parent_idx = self.parent(i)
            self.heap[i], self.heap[parent_idx] = self.heap[parent_idx], self.heap[i]
            i = parent_idx

    def extract_max(self):
        """Extrait et retourne l'élément maximum"""
        if not self.heap:
            return None

        max_val = self.heap[0]
        last_val = self.heap.pop()

        if self.heap:
            self.heap[0] = last_val
            self._heapify_down(0)

        return max_val

    def _heapify_down(self, i):
        """Réorganise le tas vers le bas pour maintenir la propriété de tas max"""
        max_index = i
        children = self.child_indices(i)
        
        for child in children:
            if child < len(self.heap) and self.heap[child] > self.heap[max_index]:
                max_index = child

        if i != max_index:
            self.heap[i], self.heap[max_index] = self.heap[max_index], self.heap[i]
            self._heapify_down(max_index)

    def delete_value(self, value):
        """Supprime une valeur spécifique du tas"""
        if value not in self.heap:
            return False

        index = self.heap.index(value)
        self.heap[index] = self.heap[-1]
        self.heap.pop()

        if index < len(self.heap):
            self._heapify_down(index)
            self._heapify_up(index)

        return True

    def delete_batch(self, values):
        """Supprime plusieurs valeurs en une seule opération"""
        deleted_count = 0
        not_found_count = 0

        for value in values:
            if self.delete_value(value):
                deleted_count += 1
            else:
                not_found_count += 1

        return deleted_count, not_found_count

    def search(self, value):
        return value in self.heap

    def get_max(self):
        return self.heap[0] if self.heap else None

    def get_depth(self):
        """Retourne la profondeur du tas (commence à 0) - VERSION CORRIGÉE"""
        if not self.heap:
            return -1  # Tas vide = profondeur -1
        
        n = len(self.heap)
        if n == 0:
            return -1
        elif n == 1:
            return 0  # Seulement la racine = profondeur 0
        
        # Calcul correct de la profondeur pour un tas m-fonctionnel
        depth = 0
        max_nodes_at_depth = 1  # Nombre maximum de nœuds à la profondeur d
        
        while n > 0:
            n -= max_nodes_at_depth
            if n > 0:
                depth += 1
                max_nodes_at_depth *= self.m
        
        return depth


    def build_heap(self, values):
        """Construit un tas m-fonctionnel à partir d'une liste de valeurs"""
        self.heap = []
        unique_values = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
                self.insert(value)
        return self

    def get_root_node(self):
        """Retourne la racine sous forme de nœud pour la visualisation"""
        if not self.heap:
            return None

        class MHeapNode:
            def __init__(self, value):
                self.value = value
                self.children = []  # Liste des enfants

        nodes = [MHeapNode(val) for val in self.heap]

        for i in range(len(nodes)):
            children_indices = self.child_indices(i)
            for child_idx in children_indices:
                if child_idx < len(nodes):
                    nodes[i].children.append(nodes[child_idx])

        return nodes[0] if nodes else None

class MFunctionalMinHeap:
    def __init__(self, m=3):
        self.heap = []
        self.m = m  # Nombre d'enfants par nœud

    def parent(self, i):
        return (i - 1) // self.m

    def child_indices(self, i):
        """Retourne les indices des enfants du nœud i"""
        start = self.m * i + 1
        return list(range(start, start + self.m))

    def insert(self, value):
        """Insère une valeur dans le tas m-fonctionnel min"""
        if value in self.heap:
            return False  # Doublon détecté
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
        return True

    def _heapify_up(self, i):
        """Réorganise le tas vers le haut pour maintenir la propriété de tas min"""
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]:
            parent_idx = self.parent(i)
            self.heap[i], self.heap[parent_idx] = self.heap[parent_idx], self.heap[i]
            i = parent_idx

    def extract_min(self):
        """Extrait et retourne l'élément minimum"""
        if not self.heap:
            return None

        min_val = self.heap[0]
        last_val = self.heap.pop()

        if self.heap:
            self.heap[0] = last_val
            self._heapify_down(0)

        return min_val

    def _heapify_down(self, i):
        """Réorganise le tas vers le bas pour maintenir la propriété de tas min"""
        min_index = i
        children = self.child_indices(i)
        
        for child in children:
            if child < len(self.heap) and self.heap[child] < self.heap[min_index]:
                min_index = child

        if i != min_index:
            self.heap[i], self.heap[min_index] = self.heap[min_index], self.heap[i]
            self._heapify_down(min_index)

    def delete_value(self, value):
        """Supprime une valeur spécifique du tas"""
        if value not in self.heap:
            return False

        index = self.heap.index(value)
        self.heap[index] = self.heap[-1]
        self.heap.pop()

        if index < len(self.heap):
            self._heapify_down(index)
            self._heapify_up(index)

        return True

    def delete_batch(self, values):
        """Supprime plusieurs valeurs en une seule opération"""
        deleted_count = 0
        not_found_count = 0

        for value in values:
            if self.delete_value(value):
                deleted_count += 1
            else:
                not_found_count += 1

        return deleted_count, not_found_count

    def search(self, value):
        return value in self.heap

    def get_min(self):
        return self.heap[0] if self.heap else None

    def get_depth(self):
        """Retourne la profondeur du tas (commence à 0) - VERSION CORRIGÉE"""
        if not self.heap:
            return -1  # Tas vide = profondeur -1
        
        n = len(self.heap)
        if n == 0:
            return -1
        elif n == 1:
            return 0  # Seulement la racine = profondeur 0
        
        # Calcul correct de la profondeur pour un tas m-fonctionnel
        depth = 0
        max_nodes_at_depth = 1  # Nombre maximum de nœuds à la profondeur d
        
        while n > 0:
            n -= max_nodes_at_depth
            if n > 0:
                depth += 1
                max_nodes_at_depth *= self.m
        
        return depth

    def build_heap(self, values):
        """Construit un tas m-fonctionnel à partir d'une liste de valeurs"""
        self.heap = []
        unique_values = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
                self.insert(value)
        return self

    def get_root_node(self):
        """Retourne la racine sous forme de nœud pour la visualisation"""
        if not self.heap:
            return None

        class MHeapNode:
            def __init__(self, value):
                self.value = value
                self.children = []  # Liste des enfants

        nodes = [MHeapNode(val) for val in self.heap]

        for i in range(len(nodes)):
            children_indices = self.child_indices(i)
            for child_idx in children_indices:
                if child_idx < len(nodes):
                    nodes[i].children.append(nodes[child_idx])

        return nodes[0] if nodes else None

# =============================================================================
# Tree Visualization Functions
# =============================================================================

def tree_to_nx(root, G=None, parent=None):
    if G is None:
        G = nx.DiGraph()
    if root is None:
        return G

    node_value = root.value
    G.add_node(node_value)

    if parent is not None:
        G.add_edge(parent, node_value)

    tree_to_nx(root.left, G, node_value)
    tree_to_nx(root.right, G, node_value)

    return G

def mfunctional_tree_to_nx(root, G=None, parent=None):
    """Version pour les tas m-fonctionnels"""
    if G is None:
        G = nx.DiGraph()
    if root is None:
        return G

    node_value = root.value
    G.add_node(node_value)

    if parent is not None:
        G.add_edge(parent, node_value)

    # Parcourir tous les enfants (peu importe le nombre)
    for child in root.children:
        mfunctional_tree_to_nx(child, G, node_value)

    return G

def get_tree_positions(root):
    if root is None:
        return {}

    def compute_positions(node, level=0, pos=0, positions=None, level_widths=None):
        if positions is None:
            positions = {}
        if level_widths is None:
            level_widths = {}

        if node is None:
            return pos

        # Calculer la position du sous-arbre gauche
        left_pos = compute_positions(node.left, level + 1, pos, positions, level_widths)

        # Position actuelle du nœud
        current_pos = left_pos
        positions[node.value] = (
            current_pos,
            -level,
        )  # level commence à 0 pour la racine

        # Mettre à jour la largeur du niveau
        level_widths[level] = max(level_widths.get(level, 0), current_pos + 1)

        # Calculer la position du sous-arbre droit
        right_pos = compute_positions(
            node.right, level + 1, current_pos + 1, positions, level_widths
        )

        return right_pos

    positions = {}
    level_widths = {}
    compute_positions(root, 0, 0, positions, level_widths)

    # Normaliser les positions
    if positions:
        max_level = max(-y for _, y in positions.values()) if positions else 0
        max_width = max(level_widths.values()) if level_widths else 1

        normalized_positions = {}
        for node, (x, y) in positions.items():
            norm_x = (x / max_width) * 2 - 1 if max_width > 0 else 0
            norm_y = (y / max_level) if max_level > 0 else y
            normalized_positions[node] = (norm_x, norm_y)

        return normalized_positions

    return positions

def get_mfunctional_tree_positions(root, m=3):
    """Version pour les tas m-fonctionnels"""
    if root is None:
        return {}

    def compute_positions(node, level=0, pos=0, positions=None, level_widths=None):
        if positions is None:
            positions = {}
        if level_widths is None:
            level_widths = {}

        if node is None:
            return pos

        # Pour un nœud m-fonctionnel, nous devons calculer la position de tous les enfants
        current_pos = pos
        positions[node.value] = (current_pos, -level)
        level_widths[level] = max(level_widths.get(level, 0), current_pos + 1)

        # Calculer les positions des enfants
        next_pos = current_pos
        for child in node.children:
            next_pos = compute_positions(child, level + 1, next_pos, positions, level_widths)

        return next_pos + 1

    positions = {}
    level_widths = {}
    compute_positions(root, 0, 0, positions, level_widths)

    # Normaliser les positions
    if positions:
        max_level = max(-y for _, y in positions.values()) if positions else 0
        max_width = max(level_widths.values()) if level_widths else 1

        normalized_positions = {}
        for node, (x, y) in positions.items():
            norm_x = (x / max_width) * 2 - 1 if max_width > 0 else 0
            norm_y = (y / max_level) if max_level > 0 else y
            normalized_positions[node] = (norm_x, norm_y)

        return normalized_positions

    return positions

def visualize_tree(root, title):
    if root is None:
        st.warning("L'arbre est vide")
        return None

    G = tree_to_nx(root)
    pos = get_tree_positions(root)

    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_color="lightblue",
        node_size=800,
        font_size=10,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
        edge_color="gray",
        ax=ax,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 0.1)
    ax.axis("off")

    return fig

def visualize_mfunctional_tree(root, m, title):
    """Visualisation pour les tas m-fonctionnels"""
    if root is None:
        st.warning("L'arbre est vide")
        return None

    G = mfunctional_tree_to_nx(root)
    pos = get_mfunctional_tree_positions(root, m)

    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_color="lightgreen",
        node_size=800,
        font_size=10,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
        edge_color="gray",
        ax=ax,
    )

    plt.title(f"{title} (m={m})", fontsize=14, fontweight="bold")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 0.1)
    ax.axis("off")

    return fig

# =============================================================================
# File Import Functions
# =============================================================================

def read_values_from_file(uploaded_file):
    """Lit les valeurs depuis un fichier uploadé"""
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        values = []

        for line in content.splitlines():
            line = line.strip()
            if line:
                # Supprimer les commentaires (tout après #)
                if "#" in line:
                    line = line.split("#")[0].strip()

                # Séparer par espaces, virgules ou points-virgules
                for part in line.replace(",", " ").replace(";", " ").split():
                    try:
                        value = int(part.strip())
                        values.append(value)
                    except ValueError:
                        continue  # Ignorer les parties non numériques

        return values
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {e}")
        return []

# =============================================================================
# File Import Functions - CORRIGÉ
# =============================================================================

def show_file_import_section(data_structure):
    """Affiche la section d'importation de fichier - VERSION CORRIGÉE"""
    st.markdown("---")
    st.subheader("📁 Importation depuis un fichier")

    uploaded_file = st.file_uploader(
        "Choisir un fichier texte avec des valeurs numériques",
        type=["txt", "csv"],
        key=f"file_upload_{data_structure}",
    )

    if uploaded_file is not None:
        st.info(f"Fichier sélectionné: {uploaded_file.name}")

        # Aperçu du contenu
        content = uploaded_file.getvalue().decode("utf-8")
        st.text_area(
            "Aperçu du fichier:",
            content[:500] + ("..." if len(content) > 500 else ""),
            height=100,
        )

        # Options d'importation
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📥 Importer les valeurs",
                use_container_width=True,
                key=f"import_{data_structure}",
            ):
                start_time = time.time()
                values = read_values_from_file(uploaded_file)
                if values:
                    st.success(f"✅ {len(values)} valeurs lues depuis le fichier")

                    # RECONSTRUCTION COMPLÈTE - CORRECTION DU PROBLÈME
                    if data_structure == "Tas Max":
                        # Créer un nouveau tas max avec uniquement les valeurs du fichier
                        st.session_state.max_heap = MaxHeap()
                        st.session_state.max_heap.build_heap(values)
                        st.success(f"✅ Tas Max reconstruit avec {len(values)} valeurs")
                    elif data_structure == "Tas Min":
                        # Créer un nouveau tas min avec uniquement les valeurs du fichier
                        st.session_state.min_heap = MinHeap()
                        st.session_state.min_heap.build_heap(values)
                        st.success(f"✅ Tas Min reconstruit avec {len(values)} valeurs")
                    elif data_structure == "Arbre AVL":
                        # Construire un nouvel arbre AVL avec uniquement les valeurs du fichier
                        st.session_state.avl_tree = build_avl(values)
                        st.success(
                            f"✅ Arbre AVL reconstruit avec {len(values)} valeurs"
                        )
                    elif data_structure == "Tas M-Fonctionnel Max":
                        # Créer un nouveau tas m-fonctionnel max avec uniquement les valeurs du fichier
                        st.session_state.mfunctional_max_heap = MFunctionalMaxHeap(m=st.session_state.get('m_value', 3))
                        st.session_state.mfunctional_max_heap.build_heap(values)
                        st.success(f"✅ Tas M-Fonctionnel Max reconstruit avec {len(values)} valeurs")
                    elif data_structure == "Tas M-Fonctionnel Min":
                        # Créer un nouveau tas m-fonctionnel min avec uniquement les valeurs du fichier
                        st.session_state.mfunctional_min_heap = MFunctionalMinHeap(m=st.session_state.get('m_value', 3))
                        st.session_state.mfunctional_min_heap.build_heap(values)
                        st.success(f"✅ Tas M-Fonctionnel Min reconstruit avec {len(values)} valeurs")

                    end_time = time.time()
                    execution_time = (end_time - start_time) * 1000  # en millisecondes
                    st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")

                    st.rerun()
                else:
                    st.error("❌ Aucune valeur numérique trouvée dans le fichier")

        with col2:
            if st.button(
                "🗑️ Vider la structure",
                use_container_width=True,
                key=f"clear_{data_structure}",
            ):
                start_time = time.time()
                if data_structure == "Tas Max":
                    st.session_state.max_heap = MaxHeap()
                elif data_structure == "Tas Min":
                    st.session_state.min_heap = MinHeap()
                elif data_structure == "Arbre AVL":
                    st.session_state.avl_tree = None
                elif data_structure == "Tas M-Fonctionnel Max":
                    st.session_state.mfunctional_max_heap = MFunctionalMaxHeap(m=st.session_state.get('m_value', 3))
                elif data_structure == "Tas M-Fonctionnel Min":
                    st.session_state.mfunctional_min_heap = MFunctionalMinHeap(m=st.session_state.get('m_value', 3))
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000

                st.success("✅ Structure vidée")
                st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                st.rerun()

        # Ajouter une option pour AJOUTER les valeurs du fichier (sans supprimer les anciennes)
        # st.markdown("---")
        # st.subheader("🔧 Options avancées d'importation")

        # col3, col4 = st.columns(2)

        # with col3:
        #     if st.button(
        #         "➕ Ajouter les valeurs (sans effacer)",
        #         use_container_width=True,
        #         key=f"add_{data_structure}",
        #     ):
        #         start_time = time.time()
        #         values = read_values_from_file(uploaded_file)
        #         if values:
        #             added_count = 0
        #             duplicate_count = 0

        #             if data_structure == "Tas Max":
        #                 for value in values:
        #                     if st.session_state.max_heap.insert(value):
        #                         added_count += 1
        #                     else:
        #                         duplicate_count += 1
        #                 st.success(f"✅ {added_count} valeurs ajoutées au Tas Max")

        #             elif data_structure == "Tas Min":
        #                 for value in values:
        #                     if st.session_state.min_heap.insert(value):
        #                         added_count += 1
        #                     else:
        #                         duplicate_count += 1
        #                 st.success(f"✅ {added_count} valeurs ajoutées au Tas Min")

        #             elif data_structure == "Arbre AVL":
        #                 for value in values:
        #                     if not search_avl(st.session_state.avl_tree, value):
        #                         st.session_state.avl_tree = insert_avl(
        #                             st.session_state.avl_tree, value
        #                         )
        #                         added_count += 1
        #                     else:
        #                         duplicate_count += 1
        #                 st.success(f"✅ {added_count} valeurs ajoutées à l'AVL")

        #             elif data_structure == "Tas M-Fonctionnel Max":
        #                 for value in values:
        #                     if st.session_state.mfunctional_max_heap.insert(value):
        #                         added_count += 1
        #                     else:
        #                         duplicate_count += 1
        #                 st.success(f"✅ {added_count} valeurs ajoutées au Tas M-Fonctionnel Max")

        #             elif data_structure == "Tas M-Fonctionnel Min":
        #                 for value in values:
        #                     if st.session_state.mfunctional_min_heap.insert(value):
        #                         added_count += 1
        #                     else:
        #                         duplicate_count += 1
        #                 st.success(f"✅ {added_count} valeurs ajoutées au Tas M-Fonctionnel Min")

        #             if duplicate_count > 0:
        #                 st.warning(f"⚠️ {duplicate_count} doublons ignorés")

        #             end_time = time.time()
        #             execution_time = (end_time - start_time) * 1000
        #             st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
        #             st.rerun()
        #         else:
        #             st.error("❌ Aucune valeur numérique trouvée dans le fichier")

        # with col4:
        #     # Information sur le mode actuel
        #     st.info(
        #         """
        #     **Modes d'importation:**
        #     - **📥 Importer:** Remplace toute la structure
        #     - **➕ Ajouter:** Ajoute aux valeurs existantes
        #     - **🗑️ Vider:** Supprime toutes les valeurs
        #     """
        #     )

# =============================================================================
# Operation Functions for Each Data Structure
# =============================================================================

def show_tas_max_operations():
    st.subheader("🔺 Opérations Tas Max")

    if "max_heap" not in st.session_state:
        st.session_state.max_heap = MaxHeap()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Contrôles Tas Max**")

        tas_operation = st.radio(
            "Choisir l'opération:",
            ["Insertion", "Recherche", "Suppression", "Profondeur", "Extraire Max"],
            key="tas_max_op",
        )

        if tas_operation in ["Insertion", "Recherche", "Suppression"]:
            tas_value = st.number_input("Valeur:", value=0, step=1, key="tas_max_val")

        # BOUTON RÉINITIALISER AJOUTÉ ICI
        if st.button(
            "🔄 Réinitialiser Tas Max",
            use_container_width=True,
            key="reset_tas_max_main",
        ):
            start_time = time.time()
            st.session_state.max_heap = MaxHeap()
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            st.success("✅ Tas Max réinitialisé")
            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
            st.rerun()

        if st.button(
            "Exécuter Opération",
            type="primary",
            use_container_width=True,
            key="exec_tas_max",
        ):
            start_time = time.time()

            if tas_operation == "Insertion":
                success = st.session_state.max_heap.insert(tas_value)
                if success:
                    st.success(f"✅ Valeur {tas_value} insérée dans le Tas Max")
                else:
                    st.warning(f"⚠️ Valeur {tas_value} déjà présente dans le Tas Max")

            elif tas_operation == "Recherche":
                found = st.session_state.max_heap.search(tas_value)
                if found:
                    st.success(f"✅ Valeur {tas_value} trouvée dans le Tas Max")
                else:
                    st.warning(f"❌ Valeur {tas_value} non trouvée dans le Tas Max")

            elif tas_operation == "Suppression":
                success = st.session_state.max_heap.delete_value(tas_value)
                if success:
                    st.success(f"✅ Valeur {tas_value} supprimée du Tas Max")
                else:
                    st.warning(f"❌ Valeur {tas_value} non trouvée dans le Tas Max")

            elif tas_operation == "Profondeur":
                depth = st.session_state.max_heap.get_depth()
                st.info(f"📊 Profondeur du Tas Max: {depth}")

            elif tas_operation == "Extraire Max":
                max_val = st.session_state.max_heap.extract_max()
                if max_val is not None:
                    st.success(f"✅ Maximum extrait: {max_val}")
                else:
                    st.warning("Le Tas Max est vide")

            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # en millisecondes
            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")

        # Opérations par lot - AFFICHÉES UNIQUEMENT DANS LES SECTIONS CORRESPONDANTES
        if tas_operation == "Insertion":
            st.write("**Insertion multiple**")
            tas_batch_input = st.text_input(
                "Valeurs (séparées par espaces):",
                placeholder="10 20 5 15 25",
                key="batch_tas_max_input",
            )
            if st.button(
                "Insérer lot", use_container_width=True, key="batch_tas_max_btn"
            ):
                if tas_batch_input.strip():
                    try:
                        start_time = time.time()
                        values = [
                            int(x.strip()) for x in tas_batch_input.split() if x.strip()
                        ]
                        if values:
                            inserted_count = 0
                            duplicate_count = 0
                            for value in values:
                                if st.session_state.max_heap.insert(value):
                                    inserted_count += 1
                                else:
                                    duplicate_count += 1

                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000

                            if inserted_count > 0:
                                st.success(
                                    f"✅ {inserted_count} valeurs insérées dans le Tas Max"
                                )
                            if duplicate_count > 0:
                                st.warning(f"⚠️ {duplicate_count} doublons ignorés")

                            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                            st.rerun()
                        else:
                            st.warning("❌ Aucune valeur valide trouvée")
                    except ValueError:
                        st.error(
                            "❌ Veuillez entrer des nombres entiers valides séparés par des espaces"
                        )

        elif tas_operation == "Suppression":
            st.write("**Suppression multiple**")
            tas_batch_input = st.text_input(
                "Valeurs à supprimer (séparées par espaces):",
                placeholder="10 20 5 15 25",
                key="batch_delete_tas_max_input",
            )
            if st.button(
                "Supprimer lot",
                use_container_width=True,
                key="batch_delete_tas_max_btn",
            ):
                if tas_batch_input.strip():
                    try:
                        start_time = time.time()
                        values = [
                            int(x.strip()) for x in tas_batch_input.split() if x.strip()
                        ]
                        if values:
                            deleted_count, not_found_count = (
                                st.session_state.max_heap.delete_batch(values)
                            )
                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000

                            if deleted_count > 0:
                                st.success(
                                    f"✅ {deleted_count} valeurs supprimées du Tas Max"
                                )
                            if not_found_count > 0:
                                st.warning(f"⚠️ {not_found_count} valeurs non trouvées")

                            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                            st.rerun()
                        else:
                            st.warning("❌ Aucune valeur valide trouvée")
                    except ValueError:
                        st.error(
                            "❌ Veuillez entrer des nombres entiers valides séparés par des espaces"
                        )

    with col2:
        st.write("**Information et Visualisation**")

        if st.session_state.max_heap.heap:
            tas_depth = st.session_state.max_heap.get_depth()
            tas_max = st.session_state.max_heap.get_max()

            st.info(
                f"""
                **Information Tas Max:**
                - Profondeur: {depth}
                - Maximum: {tas_max}
                - Taille: {len(st.session_state.max_heap.heap)}
                - Structure: {st.session_state.max_heap.heap}
                """
            )

            # Visualisation
            root_node = st.session_state.max_heap.get_root_node()
            fig_tas = visualize_tree(root_node, "Tas Max")
            if fig_tas:
                st.pyplot(fig_tas)
                plt.close(fig_tas)
        else:
            st.info("🌱 Le Tas Max est vide")

    # Section importation fichier pour Tas Max
    show_file_import_section("Tas Max")

def show_tas_min_operations():
    st.subheader("🔻 Opérations Tas Min")

    if "min_heap" not in st.session_state:
        st.session_state.min_heap = MinHeap()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Contrôles Tas Min**")

        tas_operation = st.radio(
            "Choisir l'opération:",
            ["Insertion", "Recherche", "Suppression", "Profondeur", "Extraire Min"],
            key="tas_min_op",
        )

        if tas_operation in ["Insertion", "Recherche", "Suppression"]:
            tas_value = st.number_input("Valeur:", value=0, step=1, key="tas_min_val")

        # BOUTON RÉINITIALISER AJOUTÉ ICI
        if st.button(
            "🔄 Réinitialiser Tas Min",
            use_container_width=True,
            key="reset_tas_min_main",
        ):
            start_time = time.time()
            st.session_state.min_heap = MinHeap()
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            st.success("✅ Tas Min réinitialisé")
            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
            st.rerun()

        if st.button(
            "Exécuter Opération",
            type="primary",
            use_container_width=True,
            key="exec_tas_min",
        ):
            start_time = time.time()

            if tas_operation == "Insertion":
                success = st.session_state.min_heap.insert(tas_value)
                if success:
                    st.success(f"✅ Valeur {tas_value} insérée dans le Tas Min")
                else:
                    st.warning(f"⚠️ Valeur {tas_value} déjà présente dans le Tas Min")

            elif tas_operation == "Recherche":
                found = st.session_state.min_heap.search(tas_value)
                if found:
                    st.success(f"✅ Valeur {tas_value} trouvée dans le Tas Min")
                else:
                    st.warning(f"❌ Valeur {tas_value} non trouvée dans le Tas Min")

            elif tas_operation == "Suppression":
                success = st.session_state.min_heap.delete_value(tas_value)
                if success:
                    st.success(f"✅ Valeur {tas_value} supprimée du Tas Min")
                else:
                    st.warning(f"❌ Valeur {tas_value} non trouvée dans le Tas Min")

            elif tas_operation == "Profondeur":
                depth = st.session_state.min_heap.get_depth()
                st.info(f"📊 Profondeur du Tas Min: {depth}")

            elif tas_operation == "Extraire Min":
                min_val = st.session_state.min_heap.extract_min()
                if min_val is not None:
                    st.success(f"✅ Minimum extrait: {min_val}")
                else:
                    st.warning("Le Tas Min est vide")

            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")

        # Opérations par lot - AFFICHÉES UNIQUEMENT DANS LES SECTIONS CORRESPONDANTES
        if tas_operation == "Insertion":
            st.write("**Insertion multiple**")
            tas_batch_input = st.text_input(
                "Valeurs (séparées par espaces):",
                placeholder="10 20 5 15 25",
                key="batch_tas_min_input",
            )
            if st.button(
                "Insérer lot", use_container_width=True, key="batch_tas_min_btn"
            ):
                if tas_batch_input.strip():
                    try:
                        start_time = time.time()
                        values = [
                            int(x.strip()) for x in tas_batch_input.split() if x.strip()
                        ]
                        if values:
                            inserted_count = 0
                            duplicate_count = 0
                            for value in values:
                                if st.session_state.min_heap.insert(value):
                                    inserted_count += 1
                                else:
                                    duplicate_count += 1

                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000

                            if inserted_count > 0:
                                st.success(
                                    f"✅ {inserted_count} valeurs insérées dans le Tas Min"
                                )
                            if duplicate_count > 0:
                                st.warning(f"⚠️ {duplicate_count} doublons ignorés")

                            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                            st.rerun()
                        else:
                            st.warning("❌ Aucune valeur valide trouvée")
                    except ValueError:
                        st.error(
                            "❌ Veuillez entrer des nombres entiers valides séparés par des espaces"
                        )

        elif tas_operation == "Suppression":
            st.write("**Suppression multiple**")
            tas_batch_input = st.text_input(
                "Valeurs à supprimer (séparées par espaces):",
                placeholder="10 20 5 15 25",
                key="batch_delete_tas_min_input",
            )
            if st.button(
                "Supprimer lot",
                use_container_width=True,
                key="batch_delete_tas_min_btn",
            ):
                if tas_batch_input.strip():
                    try:
                        start_time = time.time()
                        values = [
                            int(x.strip()) for x in tas_batch_input.split() if x.strip()
                        ]
                        if values:
                            deleted_count, not_found_count = (
                                st.session_state.min_heap.delete_batch(values)
                            )
                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000

                            if deleted_count > 0:
                                st.success(
                                    f"✅ {deleted_count} valeurs supprimées du Tas Min"
                                )
                            if not_found_count > 0:
                                st.warning(f"⚠️ {not_found_count} valeurs non trouvées")

                            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                            st.rerun()
                        else:
                            st.warning("❌ Aucune valeur valide trouvée")
                    except ValueError:
                        st.error(
                            "❌ Veuillez entrer des nombres entiers valides séparés par des espaces"
                        )

    with col2:
        st.write("**Information et Visualisation**")

        if st.session_state.min_heap.heap:
            tas_depth = st.session_state.min_heap.get_depth()
            tas_min = st.session_state.min_heap.get_min()

            st.info(
                f"""
                **Information Tas Min:**
                - Profondeur: {tas_depth}
                - Minimum: {tas_min}
                - Taille: {len(st.session_state.min_heap.heap)}
                - Structure: {st.session_state.min_heap.heap}
                """
            )

            # Visualisation
            root_node = st.session_state.min_heap.get_root_node()
            fig_tas = visualize_tree(root_node, "Tas Min")
            if fig_tas:
                st.pyplot(fig_tas)
                plt.close(fig_tas)
        else:
            st.info("🌱 Le Tas Min est vide")

    # Section importation fichier pour Tas Min
    show_file_import_section("Tas Min")

def show_avl_operations():
    st.subheader("🌳 Opérations AVL")

    if "avl_tree" not in st.session_state:
        st.session_state.avl_tree = None

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Contrôles AVL**")

        avl_operation = st.radio(
            "Choisir l'opération:",
            ["Insertion", "Recherche", "Suppression", "Profondeur"],
            key="avl_op",
        )

        if avl_operation in ["Insertion", "Recherche", "Suppression"]:
            avl_value = st.number_input("Valeur:", value=0, step=1, key="avl_val")

        # BOUTON RÉINITIALISER AJOUTÉ ICI
        if st.button(
            "🔄 Réinitialiser AVL",
            use_container_width=True,
            key="reset_avl_main",
        ):
            start_time = time.time()
            st.session_state.avl_tree = None
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            st.success("✅ Arbre AVL réinitialisé")
            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
            st.rerun()

        if st.button(
            "Exécuter Opération",
            type="primary",
            use_container_width=True,
            key="exec_avl",
        ):
            start_time = time.time()

            if avl_operation == "Insertion":
                # Vérifier si la valeur existe déjà
                if search_avl(st.session_state.avl_tree, avl_value):
                    st.warning(f"⚠️ Valeur {avl_value} déjà présente dans l'AVL")
                else:
                    st.session_state.avl_tree = insert_avl(
                        st.session_state.avl_tree, avl_value
                    )
                    st.success(f"✅ Valeur {avl_value} insérée dans l'AVL")

            elif avl_operation == "Recherche":
                found = search_avl(st.session_state.avl_tree, avl_value)
                if found:
                    st.success(f"✅ Valeur {avl_value} trouvée dans l'AVL")
                else:
                    st.warning(f"❌ Valeur {avl_value} non trouvée dans l'AVL")

            elif avl_operation == "Suppression":
                if st.session_state.avl_tree:
                    # Vérifier si la valeur existe avant suppression
                    if search_avl(st.session_state.avl_tree, avl_value):
                        st.session_state.avl_tree = delete_avl(
                            st.session_state.avl_tree, avl_value
                        )
                        st.success(f"✅ Valeur {avl_value} supprimée de l'AVL")
                    else:
                        st.warning(f"❌ Valeur {avl_value} non trouvée dans l'AVL")
                else:
                    st.warning("L'arbre AVL est vide")

            elif avl_operation == "Profondeur":
                depth = get_depth_avl(st.session_state.avl_tree)
                st.info(f"📊 Profondeur de l'AVL: {depth}")

            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")

        # Opérations par lot - AFFICHÉES UNIQUEMENT DANS LES SECTIONS CORRESPONDANTES
        if avl_operation == "Insertion":
            st.write("**Insertion multiple**")
            avl_batch_input = st.text_input(
                "Valeurs (séparées par espaces):",
                placeholder="10 20 5 15 25",
                key="batch_avl_input",
            )
            if st.button("Insérer lot", use_container_width=True, key="batch_avl_btn"):
                if avl_batch_input.strip():
                    try:
                        start_time = time.time()
                        values = [
                            int(x.strip()) for x in avl_batch_input.split() if x.strip()
                        ]
                        if values:
                            inserted_count = 0
                            duplicate_count = 0
                            for value in values:
                                # Vérifier si la valeur existe déjà
                                if not search_avl(st.session_state.avl_tree, value):
                                    st.session_state.avl_tree = insert_avl(
                                        st.session_state.avl_tree, value
                                    )
                                    inserted_count += 1
                                else:
                                    duplicate_count += 1

                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000

                            if inserted_count > 0:
                                st.success(
                                    f"✅ {inserted_count} valeurs insérées dans l'AVL"
                                )
                            if duplicate_count > 0:
                                st.warning(f"⚠️ {duplicate_count} doublons ignorés")

                            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                            st.rerun()
                        else:
                            st.warning("❌ Aucune valeur valide trouvée")
                    except ValueError:
                        st.error(
                            "❌ Veuillez entrer des nombres entiers valides séparés par des espaces"
                        )

        elif avl_operation == "Suppression":
            st.write("**Suppression multiple**")
            avl_batch_input = st.text_input(
                "Valeurs à supprimer (séparées par espaces):",
                placeholder="10 20 5 15 25",
                key="batch_delete_avl_input",
            )
            if st.button(
                "Supprimer lot", use_container_width=True, key="batch_delete_avl_btn"
            ):
                if avl_batch_input.strip():
                    try:
                        start_time = time.time()
                        values = [
                            int(x.strip()) for x in avl_batch_input.split() if x.strip()
                        ]
                        if values:
                            deleted_count = 0
                            not_found_count = 0
                            for value in values:
                                # Vérifier si la valeur existe avant suppression
                                if search_avl(st.session_state.avl_tree, value):
                                    st.session_state.avl_tree = delete_avl(
                                        st.session_state.avl_tree, value
                                    )
                                    deleted_count += 1
                                else:
                                    not_found_count += 1

                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000

                            if deleted_count > 0:
                                st.success(
                                    f"✅ {deleted_count} valeurs supprimées de l'AVL"
                                )
                            if not_found_count > 0:
                                st.warning(f"⚠️ {not_found_count} valeurs non trouvées")

                            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                            st.rerun()
                        else:
                            st.warning("❌ Aucune valeur valide trouvée")
                    except ValueError:
                        st.error(
                            "❌ Veuillez entrer des nombres entiers valides séparés par des espaces"
                        )

    with col2:
        st.write("**Information et Visualisation**")

        if st.session_state.avl_tree:
            avl_depth = get_depth_avl(st.session_state.avl_tree)
            avl_balance = get_balance(st.session_state.avl_tree)

            st.info(
                f"""
                **Information AVL:**
                - Profondeur: {avl_depth}
                - Balance racine: {avl_balance}
                - Valeur racine: {st.session_state.avl_tree.value}
                - Hauteur AVL: {st.session_state.avl_tree.height}
                """
            )

            # Visualisation
            fig_avl = visualize_tree(st.session_state.avl_tree, "Arbre AVL")
            if fig_avl:
                st.pyplot(fig_avl)
                plt.close(fig_avl)
        else:
            st.info("🌱 L'arbre AVL est vide")

    # Section importation fichier pour AVL
    show_file_import_section("Arbre AVL")

def show_mfunctional_max_operations():
    st.subheader("🔺 Opérations Tas M-Fonctionnel Max")

    if "mfunctional_max_heap" not in st.session_state:
        st.session_state.mfunctional_max_heap = MFunctionalMaxHeap(m=3)
    
    if "m_value" not in st.session_state:
        st.session_state.m_value = 3

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Contrôles Tas M-Fonctionnel Max**")

        # Sélecteur pour m
        m_value = st.number_input(
            "Valeur de m (nombre d'enfants par nœud):",
            min_value=2,
            max_value=10,
            value=st.session_state.m_value,
            key="m_max_input"
        )
        
        if m_value != st.session_state.m_value:
            st.session_state.m_value = m_value
            # Reconstruire le tas avec le nouveau m
            old_values = st.session_state.mfunctional_max_heap.heap[:] if st.session_state.mfunctional_max_heap.heap else []
            st.session_state.mfunctional_max_heap = MFunctionalMaxHeap(m=m_value)
            st.session_state.mfunctional_max_heap.build_heap(old_values)
            st.rerun()

        tas_operation = st.radio(
            "Choisir l'opération:",
            ["Insertion", "Recherche", "Suppression", "Profondeur", "Extraire Max"],
            key="mfunctional_max_op",
        )

        if tas_operation in ["Insertion", "Recherche", "Suppression"]:
            tas_value = st.number_input("Valeur:", value=0, step=1, key="mfunctional_max_val")

        # BOUTON RÉINITIALISER
        if st.button(
            "🔄 Réinitialiser Tas M-Fonctionnel Max",
            use_container_width=True,
            key="reset_mfunctional_max_main",
        ):
            start_time = time.time()
            st.session_state.mfunctional_max_heap = MFunctionalMaxHeap(m=st.session_state.m_value)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            st.success("✅ Tas M-Fonctionnel Max réinitialisé")
            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
            st.rerun()

        if st.button(
            "Exécuter Opération",
            type="primary",
            use_container_width=True,
            key="exec_mfunctional_max",
        ):
            start_time = time.time()

            if tas_operation == "Insertion":
                success = st.session_state.mfunctional_max_heap.insert(tas_value)
                if success:
                    st.success(f"✅ Valeur {tas_value} insérée dans le Tas M-Fonctionnel Max")
                else:
                    st.warning(f"⚠️ Valeur {tas_value} déjà présente dans le Tas M-Fonctionnel Max")

            elif tas_operation == "Recherche":
                found = st.session_state.mfunctional_max_heap.search(tas_value)
                if found:
                    st.success(f"✅ Valeur {tas_value} trouvée dans le Tas M-Fonctionnel Max")
                else:
                    st.warning(f"❌ Valeur {tas_value} non trouvée dans le Tas M-Fonctionnel Max")

            elif tas_operation == "Suppression":
                success = st.session_state.mfunctional_max_heap.delete_value(tas_value)
                if success:
                    st.success(f"✅ Valeur {tas_value} supprimée du Tas M-Fonctionnel Max")
                else:
                    st.warning(f"❌ Valeur {tas_value} non trouvée dans le Tas M-Fonctionnel Max")

            elif tas_operation == "Profondeur":
                depth = st.session_state.mfunctional_max_heap.get_depth()
                st.info(f"📊 Profondeur du Tas M-Fonctionnel Max: {depth}")

            elif tas_operation == "Extraire Max":
                max_val = st.session_state.mfunctional_max_heap.extract_max()
                if max_val is not None:
                    st.success(f"✅ Maximum extrait: {max_val}")
                else:
                    st.warning("Le Tas M-Fonctionnel Max est vide")

            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")

        # Opérations par lot
        if tas_operation == "Insertion":
            st.write("**Insertion multiple**")
            tas_batch_input = st.text_input(
                "Valeurs (séparées par espaces):",
                placeholder="10 20 5 15 25",
                key="batch_mfunctional_max_input",
            )
            if st.button(
                "Insérer lot", use_container_width=True, key="batch_mfunctional_max_btn"
            ):
                if tas_batch_input.strip():
                    try:
                        start_time = time.time()
                        values = [
                            int(x.strip()) for x in tas_batch_input.split() if x.strip()
                        ]
                        if values:
                            inserted_count = 0
                            duplicate_count = 0
                            for value in values:
                                if st.session_state.mfunctional_max_heap.insert(value):
                                    inserted_count += 1
                                else:
                                    duplicate_count += 1

                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000

                            if inserted_count > 0:
                                st.success(
                                    f"✅ {inserted_count} valeurs insérées dans le Tas M-Fonctionnel Max"
                                )
                            if duplicate_count > 0:
                                st.warning(f"⚠️ {duplicate_count} doublons ignorés")

                            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                            st.rerun()
                        else:
                            st.warning("❌ Aucune valeur valide trouvée")
                    except ValueError:
                        st.error(
                            "❌ Veuillez entrer des nombres entiers valides séparés par des espaces"
                        )

        elif tas_operation == "Suppression":
            st.write("**Suppression multiple**")
            tas_batch_input = st.text_input(
                "Valeurs à supprimer (séparées par espaces):",
                placeholder="10 20 5 15 25",
                key="batch_delete_mfunctional_max_input",
            )
            if st.button(
                "Supprimer lot",
                use_container_width=True,
                key="batch_delete_mfunctional_max_btn",
            ):
                if tas_batch_input.strip():
                    try:
                        start_time = time.time()
                        values = [
                            int(x.strip()) for x in tas_batch_input.split() if x.strip()
                        ]
                        if values:
                            deleted_count, not_found_count = (
                                st.session_state.mfunctional_max_heap.delete_batch(values)
                            )
                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000

                            if deleted_count > 0:
                                st.success(
                                    f"✅ {deleted_count} valeurs supprimées du Tas M-Fonctionnel Max"
                                )
                            if not_found_count > 0:
                                st.warning(f"⚠️ {not_found_count} valeurs non trouvées")

                            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                            st.rerun()
                        else:
                            st.warning("❌ Aucune valeur valide trouvée")
                    except ValueError:
                        st.error(
                            "❌ Veuillez entrer des nombres entiers valides séparés par des espaces"
                        )

    with col2:
        st.write("**Information et Visualisation**")

        if st.session_state.mfunctional_max_heap.heap:
            tas_depth = st.session_state.mfunctional_max_heap.get_depth()
            tas_max = st.session_state.mfunctional_max_heap.get_max()

            st.info(
                f"""
                **Information Tas M-Fonctionnel Max:**
                - Profondeur: {tas_depth}
                - Maximum: {tas_max}
                - Taille: {len(st.session_state.mfunctional_max_heap.heap)}
                - Valeur de m: {st.session_state.mfunctional_max_heap.m}
                - Structure: {st.session_state.mfunctional_max_heap.heap}
                """
            )

            # Visualisation
            root_node = st.session_state.mfunctional_max_heap.get_root_node()
            fig_tas = visualize_mfunctional_tree(root_node, st.session_state.mfunctional_max_heap.m, "Tas M-Fonctionnel Max")
            if fig_tas:
                st.pyplot(fig_tas)
                plt.close(fig_tas)
        else:
            st.info("🌱 Le Tas M-Fonctionnel Max est vide")

    # Section importation fichier pour Tas M-Fonctionnel Max
    show_file_import_section("Tas M-Fonctionnel Max")

def show_mfunctional_min_operations():
    st.subheader("🔻 Opérations Tas M-Fonctionnel Min")

    if "mfunctional_min_heap" not in st.session_state:
        st.session_state.mfunctional_min_heap = MFunctionalMinHeap(m=3)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Contrôles Tas M-Fonctionnel Min**")

        # Sélecteur pour m
        m_value = st.number_input(
            "Valeur de m (nombre d'enfants par nœud):",
            min_value=2,
            max_value=10,
            value=st.session_state.m_value,
            key="m_min_input"
        )
        
        if m_value != st.session_state.m_value:
            st.session_state.m_value = m_value
            # Reconstruire le tas avec le nouveau m
            old_values = st.session_state.mfunctional_min_heap.heap[:] if st.session_state.mfunctional_min_heap.heap else []
            st.session_state.mfunctional_min_heap = MFunctionalMinHeap(m=m_value)
            st.session_state.mfunctional_min_heap.build_heap(old_values)
            st.rerun()

        tas_operation = st.radio(
            "Choisir l'opération:",
            ["Insertion", "Recherche", "Suppression", "Profondeur", "Extraire Min"],
            key="mfunctional_min_op",
        )

        if tas_operation in ["Insertion", "Recherche", "Suppression"]:
            tas_value = st.number_input("Valeur:", value=0, step=1, key="mfunctional_min_val")

        # BOUTON RÉINITIALISER
        if st.button(
            "🔄 Réinitialiser Tas M-Fonctionnel Min",
            use_container_width=True,
            key="reset_mfunctional_min_main",
        ):
            start_time = time.time()
            st.session_state.mfunctional_min_heap = MFunctionalMinHeap(m=st.session_state.m_value)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            st.success("✅ Tas M-Fonctionnel Min réinitialisé")
            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
            st.rerun()

        if st.button(
            "Exécuter Opération",
            type="primary",
            use_container_width=True,
            key="exec_mfunctional_min",
        ):
            start_time = time.time()

            if tas_operation == "Insertion":
                success = st.session_state.mfunctional_min_heap.insert(tas_value)
                if success:
                    st.success(f"✅ Valeur {tas_value} insérée dans le Tas M-Fonctionnel Min")
                else:
                    st.warning(f"⚠️ Valeur {tas_value} déjà présente dans le Tas M-Fonctionnel Min")

            elif tas_operation == "Recherche":
                found = st.session_state.mfunctional_min_heap.search(tas_value)
                if found:
                    st.success(f"✅ Valeur {tas_value} trouvée dans le Tas M-Fonctionnel Min")
                else:
                    st.warning(f"❌ Valeur {tas_value} non trouvée dans le Tas M-Fonctionnel Min")

            elif tas_operation == "Suppression":
                success = st.session_state.mfunctional_min_heap.delete_value(tas_value)
                if success:
                    st.success(f"✅ Valeur {tas_value} supprimée du Tas M-Fonctionnel Min")
                else:
                    st.warning(f"❌ Valeur {tas_value} non trouvée dans le Tas M-Fonctionnel Min")

            elif tas_operation == "Profondeur":
                depth = st.session_state.mfunctional_min_heap.get_depth()
                st.info(f"📊 Profondeur du Tas M-Fonctionnel Min: {depth}")

            elif tas_operation == "Extraire Min":
                min_val = st.session_state.mfunctional_min_heap.extract_min()
                if min_val is not None:
                    st.success(f"✅ Minimum extrait: {min_val}")
                else:
                    st.warning("Le Tas M-Fonctionnel Min est vide")

            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")

        # Opérations par lot
        if tas_operation == "Insertion":
            st.write("**Insertion multiple**")
            tas_batch_input = st.text_input(
                "Valeurs (séparées par espaces):",
                placeholder="10 20 5 15 25",
                key="batch_mfunctional_min_input",
            )
            if st.button(
                "Insérer lot", use_container_width=True, key="batch_mfunctional_min_btn"
            ):
                if tas_batch_input.strip():
                    try:
                        start_time = time.time()
                        values = [
                            int(x.strip()) for x in tas_batch_input.split() if x.strip()
                        ]
                        if values:
                            inserted_count = 0
                            duplicate_count = 0
                            for value in values:
                                if st.session_state.mfunctional_min_heap.insert(value):
                                    inserted_count += 1
                                else:
                                    duplicate_count += 1

                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000

                            if inserted_count > 0:
                                st.success(
                                    f"✅ {inserted_count} valeurs insérées dans le Tas M-Fonctionnel Min"
                                )
                            if duplicate_count > 0:
                                st.warning(f"⚠️ {duplicate_count} doublons ignorés")

                            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                            st.rerun()
                        else:
                            st.warning("❌ Aucune valeur valide trouvée")
                    except ValueError:
                        st.error(
                            "❌ Veuillez entrer des nombres entiers valides séparés par des espaces"
                        )

        elif tas_operation == "Suppression":
            st.write("**Suppression multiple**")
            tas_batch_input = st.text_input(
                "Valeurs à supprimer (séparées par espaces):",
                placeholder="10 20 5 15 25",
                key="batch_delete_mfunctional_min_input",
            )
            if st.button(
                "Supprimer lot",
                use_container_width=True,
                key="batch_delete_mfunctional_min_btn",
            ):
                if tas_batch_input.strip():
                    try:
                        start_time = time.time()
                        values = [
                            int(x.strip()) for x in tas_batch_input.split() if x.strip()
                        ]
                        if values:
                            deleted_count, not_found_count = (
                                st.session_state.mfunctional_min_heap.delete_batch(values)
                            )
                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000

                            if deleted_count > 0:
                                st.success(
                                    f"✅ {deleted_count} valeurs supprimées du Tas M-Fonctionnel Min"
                                )
                            if not_found_count > 0:
                                st.warning(f"⚠️ {not_found_count} valeurs non trouvées")

                            st.info(f"⏱️ Temps d'exécution: {execution_time:.2f} ms")
                            st.rerun()
                        else:
                            st.warning("❌ Aucune valeur valide trouvée")
                    except ValueError:
                        st.error(
                            "❌ Veuillez entrer des nombres entiers valides séparés par des espaces"
                        )

    with col2:
        st.write("**Information et Visualisation**")

        if st.session_state.mfunctional_min_heap.heap:
            tas_depth = st.session_state.mfunctional_min_heap.get_depth()
            tas_min = st.session_state.mfunctional_min_heap.get_min()

            st.info(
                f"""
                **Information Tas M-Fonctionnel Min:**
                - Profondeur: {tas_depth}
                - Minimum: {tas_min}
                - Taille: {len(st.session_state.mfunctional_min_heap.heap)}
                - Valeur de m: {st.session_state.mfunctional_min_heap.m}
                - Structure: {st.session_state.mfunctional_min_heap.heap}
                """
            )

            # Visualisation
            root_node = st.session_state.mfunctional_min_heap.get_root_node()
            fig_tas = visualize_mfunctional_tree(root_node, st.session_state.mfunctional_min_heap.m, "Tas M-Fonctionnel Min")
            if fig_tas:
                st.pyplot(fig_tas)
                plt.close(fig_tas)
        else:
            st.info("🌱 Le Tas M-Fonctionnel Min est vide")

    # Section importation fichier pour Tas M-Fonctionnel Min
    show_file_import_section("Tas M-Fonctionnel Min")

# =============================================================================
# TP2 Main Function
# =============================================================================

def show_tp2():
    st.set_page_config(
        page_title="TP2 - Opérations sur les Structures de Données",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(
        "<h2 style='color: #1f77b4;'>TP2: Opérations sur les Structures de Données</h2>",
        unsafe_allow_html=True,
    )

    # Main navigation
    st.markdown("### 🎯 Choisir le type de structure de données")

    data_structure = st.radio(
        "Sélectionnez la structure à manipuler:",
        ["Tas Max", "Tas Min", "Tas M-Fonctionnel Max", "Tas M-Fonctionnel Min" ,"Arbre AVL"],
        horizontal=True,
    )

    st.markdown("---")

    # Show the selected operations
    if data_structure == "Tas Max":
        show_tas_max_operations()
    elif data_structure == "Tas Min":
        show_tas_min_operations()
    elif data_structure == "Arbre AVL":
        show_avl_operations()
    elif data_structure == "Tas M-Fonctionnel Max":
        show_mfunctional_max_operations()
    elif data_structure == "Tas M-Fonctionnel Min":
        show_mfunctional_min_operations()

    # Educational information
    st.markdown("---")
    st.subheader("📚 Informations Pédagogiques")

    if data_structure == "Tas Max":
        st.markdown(
            """
            **Tas Max:**
            - Arbre binaire complet avec propriété de tas
            - Parent ≥ enfants (pour tas max)
            - Racine = élément maximum
            - Complexité: O(log n) pour insertion/suppression
            - Utilisations: Files de priorité, tri par tas
            - **Profondeur:** Commence à 0 (racine = niveau 0)
            - **Doublons:** Non autorisés
            - **Importation fichier:** Support des fichiers .txt et .csv
            - **Temps d'exécution:** Affiché pour chaque opération
            """
        )
    elif data_structure == "Tas Min":
        st.markdown(
            """
            **Tas Min:**
            - Arbre binaire complet avec propriété de tas
            - Parent ≤ enfants (pour tas min)
            - Racine = élément minimum
            - Complexité: O(log n) pour insertion/suppression
            - Utilisations: Files de priorité, algorithme de Dijkstra
            - **Profondeur:** Commence à 0 (racine = niveau 0)
            - **Doublons:** Non autorisés
            - **Importation fichier:** Support des fichiers .txt et .csv
            - **Temps d'exécution:** Affiché pour chaque opération
            """
        )
    elif data_structure == "Arbre AVL":
        st.markdown(
            """
            **Arbre AVL:**
            - Arbre binaire de recherche auto-équilibré
            - Hauteur des sous-arbres diffère au plus de 1
            - Complexité: O(log n) pour toutes les opérations
            - Utilisations: Bases de données, systèmes de fichiers
            - **Profondeur:** Commence à 0 (racine = niveau 0)
            - **Doublons:** Non autorisés
            - **Importation fichier:** Support des fichiers .txt et .csv
            - **Temps d'exécution:** Affiché pour chaque opération
            """
        )
    elif data_structure in ["Tas M-Fonctionnel Max", "Tas M-Fonctionnel Min"]:
        st.markdown(
            f"""
            **Tas M-Fonctionnel ({'Max' if 'Max' in data_structure else 'Min'}):**
            - Généralisation du tas binaire avec m enfants par nœud
            - Chaque nœud a jusqu'à m enfants
            - Propriété de tas {'max' if 'Max' in data_structure else 'min'} maintenue
            - Complexité: O(log_m n) pour insertion/suppression
            - Utilisations: Algorithmes avancés, files de priorité optimisées
            - **Profondeur:** Commence à 0 (racine = niveau 0)
            - **Doublons:** Non autorisés
            - **Importation fichier:** Support des fichiers .txt et .csv
            - **Temps d'exécution:** Affiché pour chaque opération
            - **Valeur de m:** Configurable (2-10 enfants par nœud)
            """
        )
