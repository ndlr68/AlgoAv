import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import re


class Node:
    """Node class for ABR (Binary Search Tree)."""
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert(root, value):
    if root is None:
        return Node(value)
    if value < root.value:
        root.left = insert(root.left, value)
    elif value > root.value:
        root.right = insert(root.right, value)
    return root

def build_bst(values):
    root = None
    for value in values:
        root = insert(root, value)
    return root

# AVL Tree Implementation

class AVLNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

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
        return node

    update_height(node)
    balance = get_balance(node)

    if balance > 1 and value < node.left.value:
        return right_rotate(node)
    if balance < -1 and value > node.right.value:
        return left_rotate(node)
    if balance > 1 and value > node.left.value:
        node.left = left_rotate(node.left)
        return right_rotate(node)
    if balance < -1 and value < node.right.value:
        node.right = right_rotate(node.right)
        return left_rotate(node)

    return node

def build_avl(values):
    root = None
    for value in values:
        root = insert_avl(root, value)
    return root

# TAS (Heap) Implementation

class HeapNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None

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
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
    
    def build_heap(self, values):
        self.heap = []
        for value in values:
            self.insert(value)
        return self
    
    def get_root_node(self):
        if not self.heap:
            return None
        nodes = [HeapNode(val) for val in self.heap]
        for i in range(len(nodes)):
            left_idx = self.left_child(i)
            right_idx = self.right_child(i)
            if left_idx < len(nodes):
                nodes[i].left = nodes[left_idx]
                nodes[left_idx].parent = nodes[i]
            if right_idx < len(nodes):
                nodes[i].right = nodes[right_idx]
                nodes[right_idx].parent = nodes[i]
        return nodes[0] if nodes else None

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
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        while i > 0 and self.heap[self.parent(i)] < self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
    
    def build_heap(self, values):
        self.heap = []
        for value in values:
            self.insert(value)
        return self
    
    def get_root_node(self):
        if not self.heap:
            return None
        nodes = [HeapNode(val) for val in self.heap]
        for i in range(len(nodes)):
            left_idx = self.left_child(i)
            right_idx = self.right_child(i)
            if left_idx < len(nodes):
                nodes[i].left = nodes[left_idx]
                nodes[left_idx].parent = nodes[i]
            if right_idx < len(nodes):
                nodes[i].right = nodes[right_idx]
                nodes[right_idx].parent = nodes[i]
        return nodes[0] if nodes else None

def build_tas(values, heap_type="TAS Min"):
    if heap_type == "TAS Min":
        heap = MinHeap()
    else:
        heap = MaxHeap()
    heap.build_heap(values)
    return heap.get_root_node()

# AMR Implementation

class AMRNode:
    def __init__(self, values=None, max_children=3):
        self.values = values if values else []
        self.max_children = max_children
        self.children = []
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_full(self):
        return len(self.values) >= self.max_children - 1
    
    def insert_value(self, value):
        self.values.append(value)
        self.values.sort()
        if len(self.children) < len(self.values) + 1:
            self.children.append(None)

def insert_amr(root, value, max_children=3):
    if root is None:
        root = AMRNode(max_children=max_children)
        root.insert_value(value)
        return root
    
    current = root
    path = []
    
    while current:
        path.append(current)
        if not current.is_full():
            current.insert_value(value)
            return root
        
        found_child = False
        for i, node_value in enumerate(current.values):
            if value < node_value:
                current = current.children[i]
                found_child = True
                break
        
        if not found_child:
            current = current.children[-1]
    
    if path:
        parent = path[-1]
        new_node = AMRNode(max_children=max_children)
        new_node.insert_value(value)
        
        insert_index = 0
        for i, val in enumerate(parent.values):
            if value < val:
                insert_index = i
                break
            insert_index = i + 1
        
        parent.children.insert(insert_index, new_node)
        while len(parent.children) > len(parent.values) + 1:
            parent.children.pop()
    
    return root

def build_amr(values, max_children=3):
    root = None
    for value in values:
        root = insert_amr(root, value, max_children)
    return root

# =============================================================================
# B-Tree Implementation
# =============================================================================

class BTreeNode:
    def __init__(self, order, leaf=False):
        self.order = order
        self.keys = []
        self.children = []
        self.leaf = leaf
    
    def is_full(self):
        return len(self.keys) >= self.order - 1

class BTree:
    def __init__(self, order):
        if order < 3:
            raise ValueError("B-Tree order must be at least 3")
        self.order = order
        self.root = BTreeNode(order, leaf=True)
    
    def insert(self, key):
        root = self.root
        if root.is_full():
            new_root = BTreeNode(self.order, leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key)
    
    def _insert_non_full(self, node, key):
        if node.leaf:
            self._insert_key_sorted(node, key)
        else:
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)
    
    def _split_child(self, parent, child_index):
        child = parent.children[child_index]
        new_node = BTreeNode(self.order, leaf=child.leaf)
        
        mid = len(child.keys) // 2
        median_key = child.keys[mid]
        
        new_node.keys = child.keys[mid + 1:]
        child.keys = child.keys[:mid]
        
        if not child.leaf:
            new_node.children = child.children[mid + 1:]
            child.children = child.children[:mid + 1]
        
        parent.keys.insert(child_index, median_key)
        parent.children.insert(child_index + 1, new_node)
    
    def _insert_key_sorted(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        node.keys.insert(i, key)
    
    def get_root_for_visualization(self):
        return self.root

def build_b_tree(values, order):
    if order < 3:
        raise ValueError("B-Tree order must be at least 3")
    btree = BTree(order)
    for value in values:
        btree.insert(value)
    return btree

# =============================================================================
# Helper Functions
# =============================================================================

def check_min_heap_property(node):
    if node is None:
        return True
    if node.left and node.value > node.left.value:
        return False
    if node.right and node.value > node.right.value:
        return False
    return check_min_heap_property(node.left) and check_min_heap_property(node.right)

def check_max_heap_property(node):
    if node is None:
        return True
    if node.left and node.value < node.left.value:
        return False
    if node.right and node.value < node.right.value:
        return False
    return check_max_heap_property(node.left) and check_max_heap_property(node.right)

def count_amr_nodes(root):
    if root is None:
        return 0
    count = 1
    for child in root.children:
        if child is not None:
            count += count_amr_nodes(child)
    return count

def count_b_tree_nodes(node):
    if node is None:
        return 0
    count = 1
    for child in node.children:
        count += count_b_tree_nodes(child)
    return count

def get_root_value(root):
    if root is None:
        return "None"
    if hasattr(root, "values"):
        return f"[{','.join(map(str, root.values))}]"
    elif hasattr(root, "keys"):
        return f"[{','.join(map(str, root.keys))}]"
    else:
        return root.value

# =============================================================================
# Tree Visualization Functions
# =============================================================================

def tree_to_nx(root, G=None, parent=None):
    if G is None:
        G = nx.DiGraph()
    if root is None:
        return G

    if hasattr(root, "values"):
        node_value = f"[{','.join(map(str, root.values))}]"
    elif hasattr(root, "keys"):
        node_value = f"B[{','.join(map(str, root.keys))}]"
    else:
        node_value = root.value

    G.add_node(node_value)
    if parent is not None:
        G.add_edge(parent, node_value)

    if hasattr(root, "values"):
        for child in root.children:
            if child is not None:
                tree_to_nx(child, G, node_value)
    elif hasattr(root, "keys"):
        for child in root.children:
            if child is not None:
                tree_to_nx(child, G, node_value)
    else:
        tree_to_nx(root.left, G, node_value)
        tree_to_nx(root.right, G, node_value)

    return G

def get_depth(node):
    if node is None:
        return 0
    if hasattr(node, "values"):
        child_depths = [get_depth(child) for child in node.children if child is not None]
        return 1 + max(child_depths) if child_depths else 1
    elif hasattr(node, "keys"):
        child_depths = [get_depth(child) for child in node.children if child is not None]
        return 1 + max(child_depths) if child_depths else 1
    else:
        left_depth = get_depth(node.left)
        right_depth = get_depth(node.right)
        return 1 + max(left_depth, right_depth)

def compute_levels(node, level=0, levels=None):
    if levels is None:
        levels = {}
    if node is not None:
        if hasattr(node, "values"):
            node_value = f"[{','.join(map(str, node.values))}]"
        elif hasattr(node, "keys"):
            node_value = f"B[{','.join(map(str, node.keys))}]"
        else:
            node_value = node.value

        levels[node_value] = level

        if hasattr(node, "values"):
            for child in node.children:
                if child is not None:
                    compute_levels(child, level + 1, levels)
        elif hasattr(node, "keys"):
            for child in node.children:
                if child is not None:
                    compute_levels(child, level + 1, levels)
        else:
            compute_levels(node.left, level + 1, levels)
            compute_levels(node.right, level + 1, levels)

    return levels

def get_inorder_positions(node, positions, pos_counter):
    if node is None:
        return pos_counter

    if hasattr(node, "values"):
        for i in range(len(node.children)):
            if i < len(node.children) and node.children[i] is not None:
                pos_counter = get_inorder_positions(node.children[i], positions, pos_counter)
            if i < len(node.values):
                node_value = f"[{','.join(map(str, node.values))}]"
                if node_value not in positions:
                    positions[node_value] = pos_counter[0]
                    pos_counter[0] += 1
        if len(node.children) > len(node.values) and node.children[-1] is not None:
            pos_counter = get_inorder_positions(node.children[-1], positions, pos_counter)
        return pos_counter
    elif hasattr(node, "keys"):
        for i in range(len(node.children)):
            if i < len(node.children) and node.children[i] is not None:
                pos_counter = get_inorder_positions(node.children[i], positions, pos_counter)
            if i < len(node.keys):
                node_value = f"B[{','.join(map(str, node.keys))}]"
                if node_value not in positions:
                    positions[node_value] = pos_counter[0]
                    pos_counter[0] += 1
        if len(node.children) > len(node.keys) and node.children[-1] is not None:
            pos_counter = get_inorder_positions(node.children[-1], positions, pos_counter)
        return pos_counter
    else:
        pos_counter = get_inorder_positions(node.left, positions, pos_counter)
        positions[node.value] = pos_counter[0]
        pos_counter[0] += 1
        pos_counter = get_inorder_positions(node.right, positions, pos_counter)
        return pos_counter

def get_tree_positions(root):
    if root is None:
        return {}
    levels = compute_levels(root)
    positions = {}
    pos_counter = [0]
    get_inorder_positions(root, positions, pos_counter)

    layout_pos = {}
    if positions:
        max_pos = max(positions.values()) if positions else 0
        max_level = max(levels.values()) if levels else 0
        for node_val, in_pos in positions.items():
            x = in_pos / max_pos if max_pos > 0 else 0.5
            y = -levels[node_val] / max_level if max_level > 0 else 0
            layout_pos[node_val] = (x, y)
    else:
        if hasattr(root, "values"):
            node_value = f"[{','.join(map(str, root.values))}]"
            layout_pos[node_value] = (0.5, 0)
        elif hasattr(root, "keys"):
            node_value = f"B[{','.join(map(str, root.keys))}]"
            layout_pos[node_value] = (0.5, 0)
        else:
            layout_pos[root.value] = (0.5, 0)
    return layout_pos

def get_amr_positions(root):
    G = tree_to_nx(root)
    if len(G.nodes) == 0:
        return {}
    pos = nx.spring_layout(G, k=2, iterations=50)
    return pos

# =============================================================================
# Graph Functions
# =============================================================================

def parse_nodes(node_input):
    if not node_input.strip():
        return []
    return [n.strip() for n in node_input.split(",") if n.strip()]

def parse_edges(edge_input, is_directed, is_weighted):
    edges = []
    if not edge_input.strip():
        return edges

    edge_list = [e.strip() for e in edge_input.split(",") if e.strip()]
    for edge_str in edge_list:
        if is_weighted and ":" in edge_str:
            parts = edge_str.rsplit(":", 1)
            conn = parts[0].strip()
            try:
                weight = float(parts[1].strip())
            except ValueError:
                st.warning(f"Invalid weight in edge '{edge_str}'. Skipping.")
                continue
        else:
            conn = edge_str
            weight = None

        if is_directed:
            match = re.match(r"(\w+)\s*->\s*(\w+)", conn)
            if match:
                u, v = match.groups()
                if is_weighted:
                    edges.append((u, v, {"weight": weight}))
                else:
                    edges.append((u, v))
            else:
                st.warning(f"Invalid directed edge '{conn}'. Use A->B format.")
        else:
            match = re.match(r"(\w+)\s*-\s*(\w+)", conn)
            if match:
                u, v = match.groups()
                if is_weighted:
                    edges.append((u, v, {"weight": weight}))
                else:
                    edges.append((u, v))
            else:
                st.warning(f"Invalid undirected edge '{conn}'. Use A-B format.")
    return edges

def create_graph(nodes, edges, is_directed, is_weighted):
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(nodes)
    for edge in edges:
        if is_weighted and len(edge) == 3:
            G.add_edge(edge[0], edge[1], **edge[2])
        else:
            G.add_edge(edge[0], edge[1])
    return G

def draw_graph(G, is_directed, is_weighted, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k=3, iterations=50)

    if is_directed:
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=20, edge_color="gray", width=2)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edge_color="gray", width=2)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightgreen", node_size=1200, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight="bold")

    if is_weighted:
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=9)

    plt.title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    return fig

# =============================================================================
# TP1 Main Function
# =============================================================================

def show_tp1():
    st.markdown(
        "<h2 style='color: #1f77b4;'>TP1: Arbres et Graphes (ABR, AVL, TAS, AMR, B-Arbre + Graphes)</h2>",
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Select mode:",
        ["Arbres", "Graphes"],
        help="Arbres: Tree structures. Graphes: General graphs (directed/undirected, weighted/unweighted).",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Controls")

        if mode == "Arbres":
            tree_type = st.radio(
                "Choose tree type:",
                ["ABR", "AVL", "TAS Min", "TAS Max", "AMR", "B-Arbre"],
                help="ABR: Binary Search Tree. AVL: Self-balancing BST. TAS Min/Max: Min/Max Heap. AMR: M-ary Search Tree. B-Arbre: B-Tree implementation.",
            )

            amr_order = None
            b_tree_order = None

            if tree_type == "AMR":
                amr_order_input = st.text_input(
                    "Enter the order of the AMR (an integer ≥ 3):",
                    placeholder="3",
                    help="The order determines how many keys and children each node can have.",
                )
                if amr_order_input.strip():
                    try:
                        amr_order = int(amr_order_input.strip())
                        if amr_order < 3:
                            st.error("Order must be ≥ 3")
                            amr_order = None
                        else:
                            st.success(f"✅ AMR order set to {amr_order}")
                    except ValueError:
                        st.error("Please enter a valid integer for the order")
                        amr_order = None

            elif tree_type == "B-Arbre":
                b_tree_order_input = st.text_input(
                    "Enter the order of the B-Tree (an integer ≥ 3):",
                    placeholder="3",
                    help="The order determines the maximum number of children per node.",
                )
                if b_tree_order_input.strip():
                    try:
                        b_tree_order = int(b_tree_order_input.strip())
                        if b_tree_order < 3:
                            st.error("❌ Order must be ≥ 3")
                            b_tree_order = None
                        else:
                            max_keys = b_tree_order - 1
                            max_children = b_tree_order
                            st.success(f"✅ B-Tree order set to {b_tree_order}")
                            st.info(f"📊 Max keys per node: {max_keys}, Max children: {max_children}")
                    except ValueError:
                        st.error("Please enter a valid integer for the order")
                        b_tree_order = None

            input_values = st.text_input(
                "📥 Enter the values to insert into the tree (separated by spaces or commas):",
                placeholder="10 5 20 15 3",
                help="Values will be inserted in the order provided to build the tree.",
            )

            if st.button("Créer l'arbre", type="primary", use_container_width=True):
                if input_values.strip():
                    try:
                        values_str = input_values.replace(",", " ").split()
                        values = [int(x.strip()) for x in values_str if x.strip()]

                        if not values:
                            with col2:
                                st.warning("No valid values provided.")
                        else:
                            root = None
                            title = ""

                            if tree_type == "ABR":
                                root = build_bst(values)
                                title = f"{tree_type} (Binary Search Tree)"
                            elif tree_type == "AVL":
                                root = build_avl(values)
                                title = f"{tree_type} Tree (Self-Balancing)"
                            elif tree_type in ["TAS Min", "TAS Max"]:
                                root = build_tas(values, tree_type)
                                heap_type = "Min Heap" if tree_type == "TAS Min" else "Max Heap"
                                title = f"{tree_type} ({heap_type})"
                            elif tree_type == "AMR":
                                if amr_order is None:
                                    with col2:
                                        st.error("❌ Please first enter a valid order for the AMR (≥ 3)")
                                    root = None
                                else:
                                    root = build_amr(values, amr_order)
                                    title = f"{tree_type} ({amr_order}-ary Search Tree)"
                            elif tree_type == "B-Arbre":
                                if b_tree_order is None:
                                    with col2:
                                        st.error("❌ Please first enter a valid order for the B-Tree (≥ 3)")
                                    root = None
                                else:
                                    btree = build_b_tree(values, b_tree_order)
                                    root = btree.get_root_for_visualization()
                                    title = f"{tree_type} (Order {b_tree_order})"

                            if root:
                                G = tree_to_nx(root)
                                if tree_type in ["AMR", "B-Arbre"]:
                                    positions = get_tree_positions(root)
                                    if not positions:
                                        positions = get_amr_positions(root)
                                else:
                                    positions = get_tree_positions(root)

                                fig, ax = plt.subplots(figsize=(10, 8))
                                try:
                                    nx.draw(
                                        G,
                                        pos=positions,
                                        with_labels=True,
                                        labels={node: str(node) for node in G.nodes()},
                                        node_color="lightblue",
                                        node_size=1200,
                                        font_size=11,
                                        font_weight="bold",
                                        arrows=True,
                                        arrowsize=15,
                                        edge_color="gray",
                                        ax=ax,
                                    )
                                except nx.NetworkXError:
                                    st.warning("Using fallback layout for tree visualization.")
                                    pos = nx.spring_layout(G, k=2, iterations=50)
                                    nx.draw(
                                        G,
                                        pos=pos,
                                        with_labels=True,
                                        labels={node: str(node) for node in G.nodes()},
                                        node_color="lightblue",
                                        node_size=1200,
                                        font_size=11,
                                        font_weight="bold",
                                        arrows=True,
                                        arrowsize=15,
                                        edge_color="gray",
                                        ax=ax,
                                    )

                                plt.title(title, fontsize=14, fontweight="bold")
                                ax.set_xlim(-1.1, 1.1)
                                ax.set_ylim(-1.1, 1.1)
                                ax.axis("off")

                                with col2:
                                    st.pyplot(fig)
                                    plt.close(fig)

                                    tree_height = get_depth(root)
                                    if tree_type == "AMR":
                                        node_count = count_amr_nodes(root)
                                        st.info(f"AMR Tree (order {amr_order}) built with {len(values)} values. Height: {tree_height}. Nodes: {node_count}")
                                        st.info(f"📊 Each node can have up to {amr_order-1} keys and {amr_order} children")
                                    elif tree_type == "B-Arbre":
                                        node_count = count_b_tree_nodes(root)
                                        st.info(f"B-Tree (order {b_tree_order}) built with {len(values)} values. Height: {tree_height}. Total nodes: {node_count}")
                                        st.success("✅ B-Tree properties verified:")
                                        st.success("- All leaves at same level")
                                        st.success("- Node capacity rules respected")
                                        st.success("- Balanced structure maintained")
                                    else:
                                        st.info(f"Tree built with {len(values)} nodes. Root value: {get_root_value(root)}. Height: {tree_height}")

                                    if tree_type == "AVL":
                                        balance_info = f"AVL Balance: {get_balance(root)}"
                                        if abs(get_balance(root)) <= 1:
                                            st.success(f"✅ {balance_info} - Tree is balanced!")
                                        else:
                                            st.error(f"❌ {balance_info} - Tree is NOT balanced!")

                                    if tree_type.startswith("TAS"):
                                        if tree_type == "TAS Min":
                                            heap_ok = check_min_heap_property(root)
                                            if heap_ok:
                                                st.success("✅ Min Heap property maintained!")
                                            else:
                                                st.error("❌ Min Heap property violated!")
                                        else:
                                            heap_ok = check_max_heap_property(root)
                                            if heap_ok:
                                                st.success("✅ Max Heap property maintained!")
                                            else:
                                                st.error("❌ Max Heap property violated!")

                    except ValueError:
                        with col2:
                            st.error("Please enter valid integers separated by spaces or commas.")
                else:
                    with col2:
                        st.warning("Please enter values to create the tree.")

            if st.button("Reset Tree", use_container_width=True):
                st.rerun()

        elif mode == "Graphes":
            st.subheader("Graphe Controls")
            is_directed = st.checkbox("Orienté (Directed)", value=False)
            is_weighted = st.checkbox("Pondéré (Weighted)", value=False)
            node_input = st.text_input("Nodes (comma-separated, e.g., A,B,C,D):", placeholder="A,B,C,D")
            edge_input = st.text_input(
                "Edges:",
                placeholder="Undirected: A-B,C-D | Directed: A->B,B->C | Weighted: A-B:5",
            )

            if st.button("Créer le graphe", type="primary", use_container_width=True):
                nodes = parse_nodes(node_input)
                edges = parse_edges(edge_input, is_directed, is_weighted)

                if not nodes:
                    with col2:
                        st.warning("Please enter nodes.")
                elif not edges:
                    with col2:
                        st.warning("Please enter edges.")
                else:
                    G = create_graph(nodes, edges, is_directed, is_weighted)
                    graph_type = "Directed" if is_directed else "Undirected"
                    weight_type = "Weighted" if is_weighted else "Unweighted"
                    title = f"{graph_type} {weight_type} Graph"
                    fig = draw_graph(G, is_directed, is_weighted, title)
                    with col2:
                        st.pyplot(fig)
                        plt.close(fig)
                        st.info(f"Graph created with {len(nodes)} nodes and {len(edges)} edges.")