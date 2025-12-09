# tp3_heap_sort.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from graphviz import Digraph

# =============================================================================
# Merge Sort Implementation 
# =============================================================================

class MergeSortTree:
    """Classe pour représenter l'arbre de tri fusion"""
    def __init__(self, array, level=0, parent=None, side=None):
        self.array = array.copy()
        self.level = level
        self.parent = parent
        self.side = side  # 'left', 'right' ou None pour la racine
        self.left = None
        self.right = None
        self.merged = None
        self.operation = "initial"

def merge_sort_tree(arr, tree_node=None, steps=None, description=None):
    """
    Trie un tableau en utilisant l'algorithme de tri fusion avec construction de l'arbre
    """
    if tree_node is None:
        tree_node = MergeSortTree(arr)
    
    if steps is None:
        steps = []
    if description is None:
        description = []
    
    # Ajouter l'état initial
    if not steps:
        steps.append(arr.copy())
        description.append("Début du tri fusion")
        tree_node.operation = "début"
    
    if len(arr) > 1:
        # DIVISER : exactement comme dans le document
        # mid = len(arr) // 2

        mid = (len(arr) + 1) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        
        # Créer les nœuds enfants
        tree_node.left = MergeSortTree(left_half, tree_node.level + 1, tree_node, 'left')
        tree_node.right = MergeSortTree(right_half, tree_node.level + 1, tree_node, 'right')
        tree_node.left.operation = "division"
        tree_node.right.operation = "division"
        
        if steps is not None:
            steps.append(arr.copy())
            description.append(f"Division: {left_half} | {right_half}")
        
        # RÉGNER : appels récursifs comme dans le document

        merge_sort_tree(left_half, tree_node.left, steps, description)
        merge_sort_tree(right_half, tree_node.right, steps, description)
        
        # COMBINER : fusion comme dans le document
        i = j = k = 0
        
        while i < len(left_half) and j < len(right_half):
            if left_half[i] <= right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
        
        # Copier les éléments restants (exactement comme dans le document)
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
        
        # Enregistrer le résultat fusionné dans l'arbre
        tree_node.merged = arr.copy()
        tree_node.operation = "fusion"
        
        if steps is not None:
            steps.append(arr.copy())
            description.append(f"Fusion: {arr}")
    
    else:
        # Cas de base : tableau d'un seul élément
        tree_node.operation = "élément unique"
        if steps is not None:
            steps.append(arr.copy())
            description.append(f"Élément unique: {arr}")
    
    return tree_node

def merge_sort_wrapper(arr, steps=None, description=None):
    """
    Wrapper pour le tri fusion qui gère les étapes et l'arbre
    """
    if steps is not None:
        steps.clear()
        if description is not None:
            description.clear()
    
    arr_copy = arr.copy()
    tree_root = merge_sort_tree(arr_copy, None, steps, description)
    return arr_copy, tree_root

# =============================================================================
# Tree Visualization Functions - CORRIGÉES pour montrer TOUTES les étapes
# =============================================================================

def create_clean_merge_tree(root):
    """
    Arbre de tri fusion :
    - Division (bleu) → enfants (gauche/droite)
    - Fusion (vert) ← résultats fusionnés des enfants
    - AUCUNE flèche directe division → fusion
    """
    dot = Digraph("merge_sort_tree")
    dot.attr(rankdir="TB", nodesep="0.3", ranksep="0.4")

    def add_nodes(node):
        if node is None:
            return

        # ID du nœud division
        div_id = f"div_{id(node)}"
        dot.node(div_id,
                 label=" , ".join(map(str, node.array)),
                 shape="box",
                 style="filled",
                 fillcolor="lightblue")

        # Feuille → uniquement afficher la division (un seul élément)
        if node.left is None and node.right is None:
            return

        # Ajout des sous-arbres
        add_nodes(node.left)
        add_nodes(node.right)

        # Liens DIVISION → enfants
        dot.edge(div_id, f"div_{id(node.left)}")
        dot.edge(div_id, f"div_{id(node.right)}")

        # Nœud de fusion (vert)
        fus_id = f"fus_{id(node)}"
        dot.node(fus_id,
                 label=" , ".join(map(str, node.merged)),
                 shape="ellipse",
                 style="filled",
                 fillcolor="lightgreen")

        # Récupère la source de gauche (fusion si existe sinon division feuille)
        if node.left.merged:
            left_src = f"fus_{id(node.left)}"
        else:
            left_src = f"div_{id(node.left)}"

        # Récupère la source de droite
        if node.right.merged:
            right_src = f"fus_{id(node.right)}"
        else:
            right_src = f"div_{id(node.right)}"

        # FUSION ← enfants
        dot.edge(left_src, fus_id)
        dot.edge(right_src, fus_id)

        # 🔥 IMPORTANT : on NE relie PLUS la division → fusion
        # (c'était la flèche indésirable)

    add_nodes(root)
    return dot

# =============================================================================
# Visualization Functions (maintenues)
# =============================================================================

def visualize_sorting_steps(steps, descriptions, title):
    """Visualise les étapes du tri fusion"""
    if not steps:
        return None

    n_steps = len(steps)
    max_display_steps = 10
    
    if n_steps > max_display_steps:
        indices = [0] + list(range(1, n_steps-1, n_steps//max_display_steps)) + [n_steps-1]
        steps = [steps[i] for i in indices if i < n_steps]
        descriptions = [descriptions[i] for i in indices if i < n_steps]
        n_steps = len(steps)

    fig, axes = plt.subplots(1, n_steps, figsize=(max(15, n_steps * 3), 6))

    if n_steps == 1:
        axes = [axes]

    for idx, (step, desc) in enumerate(zip(steps, descriptions)):
        ax = axes[idx]
        bars = ax.bar(range(len(step)), step, color="lightblue", edgecolor="black")

        if idx > 0:
            prev_step = steps[idx - 1]
            for j in range(min(len(step), len(prev_step))):
                if step[j] != prev_step[j]:
                    bars[j].set_color("red")

        ax.set_title(f"Étape {idx+1}\n{desc}", fontsize=10)
        ax.set_xlabel("Index")
        ax.set_ylabel("Valeur")
        ax.grid(True, alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

def visualize_comparison(original, sorted_array, time_taken):
    """Visualise la comparaison avant/après le tri"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(range(len(original)), original, color="lightcoral", edgecolor="black")
    ax1.set_title("Avant le Tri", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Valeur")
    ax1.grid(True, alpha=0.3)

    bars2 = ax2.bar(range(len(sorted_array)), sorted_array, color="lightgreen", edgecolor="black")
    ax2.set_title("Après le Tri", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Valeur")
    ax2.grid(True, alpha=0.3)

    for bars, ax in zip([bars1, bars2], [ax1, ax2]):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle(f"Tri Fusion - Temps d'exécution: {time_taken:.6f} secondes",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

# =============================================================================
# Performance Analysis (maintenu)
# =============================================================================

def generate_test_cases():
    """Génère différents cas de test"""
    test_cases = {
        "Petit tableau aléatoire": [random.randint(1, 50) for _ in range(10)],
        "Tableau trié": list(range(1, 16)),
        "Tableau inversé": list(range(15, 0, -1)),
        "Tableau avec doublons": [random.choice([1, 2, 3, 5, 8, 13]) for _ in range(15)],
        "Moyen tableau aléatoire": [random.randint(1, 100) for _ in range(30)],
        "Grand tableau aléatoire": [random.randint(1, 200) for _ in range(50)],
    }
    return test_cases

def analyze_performance():
    """Analyse les performances du tri fusion"""
    sizes = [10, 50, 100, 200, 500, 1000, 2000]
    times = []

    for size in sizes:
        arr = [random.randint(1, 1000) for _ in range(size)]
        start_time = time.time()
        merge_sort_wrapper(arr.copy())
        end_time = time.time()
        times.append(end_time - start_time)

    return sizes, times

def plot_performance(sizes, times):
    """Trace le graphique de performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, times, "o-", linewidth=2, markersize=8, color="steelblue")
    ax.set_xlabel("Taille du tableau", fontsize=12)
    ax.set_ylabel("Temps d'exécution (secondes)", fontsize=12)
    ax.set_title("Performance du Tri Fusion", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    for i, (size, t) in enumerate(zip(sizes, times)):
        ax.annotate(f"{t:.4f}s", (size, t), textcoords="offset points",
                   xytext=(0, 10), ha="center", fontsize=8)

    return fig

# =============================================================================
# Educational Content (maintenu)
# =============================================================================

def show_educational_content():
    """Affiche le contenu pédagogique"""
    st.markdown("""
    ## 📚 Théorie du Tri Fusion
    
    ### Principe de Base
    Le **tri fusion** suit exactement l'approche **« Diviser pour Régner »** décrite dans votre document :
    
    ### Étapes de l'Algorithme
    
    1. **DIVISER** : 
       - Découper le problème en sous-problèmes
       - Diviser le tableau en deux sous-tableaux
    
    2. **RÉGNER** : 
       - Résoudre les sous-problèmes récursivement
       - Trier chaque sous-tableau
    
    3. **COMBINER** :
       - Fusionner les solutions des sous-problèmes
       - Combiner les sous-tableaux triés
    
    ### Complexités
    - **Temps**: O(n log n) dans tous les cas
    - **Espace**: O(n) pour le tableau temporaire
    
    ### Représentation Arborescente
    L'arbre montre visuellement comment le tableau est :
    - **Divisé** récursivement jusqu'à obtenir des éléments individuels
    - **Fusionné** progressivement pour former le tableau trié
    """)

# =============================================================================
# TP3 Main Function - MODIFIÉE
# =============================================================================

def show_tp3():
    st.set_page_config(
        page_title="TP3 - Tri Fusion",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("<h2 style='color: #1f77b4;'>TP3: Tri Fusion</h2>", unsafe_allow_html=True)

    mode = st.radio("Sélectionnez le mode:", ["Tri Interactif", "Analyse de Performance", "Théorie et Explications"], horizontal=True)
    st.markdown("---")

    if mode == "Tri Interactif":
        show_interactive_sorting()
    elif mode == "Analyse de Performance":
        show_performance_analysis()
    else:
        show_educational_content()

def show_interactive_sorting():
    """Affiche l'interface de tri interactif avec arbre complet"""
    st.subheader("🔄 Tri Fusion")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("### 📊 Configuration du Tri")

        input_type = st.radio("Type d'entrée:", ["Tableau personnalisé", "Exemples prédéfinis"], key="input_type")

        if input_type == "Tableau personnalisé":
            array_input = st.text_input("Entrez les valeurs (séparées par des virgules):", 
                                      placeholder="23,354,6,7,4,24,23,54,6,466,78", key="custom_array")
            
            if array_input.strip():
                try:
                    # Gérer les séparateurs espaces ou virgules
                    if ',' in array_input:
                        original_array = [int(x.strip()) for x in array_input.split(',') if x.strip()]
                    else:
                        original_array = [int(x.strip()) for x in array_input.split() if x.strip()]
                    
                    if len(original_array) > 20:
                        st.warning("⚠️ Pour une meilleure visualisation, utilisez moins de 20 éléments")
                except ValueError:
                    st.error("❌ Veuillez entrer des nombres entiers valides")
                    original_array = []
            else:
                original_array = []
        else:
            predefined_cases = {
                "[23,354,6,7,4,24,23,54,6,466,78]": [23,354,6,7,4,24,23,54,6,466,78],
                "[70,50,30,10,20,40,60]": [70,50,30,10,20,40,60],
                "[38,27,43,3,9,82,10]": [38,27,43,3,9,82,10],
                "Petit tableau désordonné": [5,2,8,1,9,3],
                "Tableau trié": [1,2,3,4,5,6],
                "Tableau inversé": [6,5,4,3,2,1],
            }

            selected_case = st.selectbox("Choisir un exemple:", list(predefined_cases.keys()))
            original_array = predefined_cases[selected_case]
            st.info(f"Tableau sélectionné: {original_array}")

        show_tree = st.checkbox("Afficher l'arbre de tri", value=True)
        show_steps = st.checkbox("Afficher les étapes détaillées", value=False)

        if st.button("🔍 Lancer le Tri Complet", type="primary", use_container_width=True):
            if original_array:
                with st.spinner("Tri en cours avec construction de l'arbre complet..."):
                    steps = []
                    descriptions = []
                    
                    start_time = time.time()
                    sorted_array, tree_root = merge_sort_wrapper(original_array, steps, descriptions)
                    end_time = time.time()
                    execution_time = end_time - start_time

                with col2:
                    st.write("### 📈 Résultats du Tri")
                    st.success(f"✅ Tri terminé en {execution_time:.6f} secondes")

                    col_result1, col_result2 = st.columns(2)
                    with col_result1:
                        st.info(f"**Tableau original:**\n{original_array}")
                    with col_result2:
                        st.info(f"**Tableau trié:**\n{sorted_array}")

                    is_sorted = all(sorted_array[i] <= sorted_array[i+1] for i in range(len(sorted_array)-1))
                    st.success(f"✅ Tableau correctement trié: {is_sorted}")

                    # Affichage de l'arbre
                    if show_tree and tree_root:
                        st.write("### 🌳 ARBRE DU TRI FUSION")
                        st.info("""
                        **Légende:**
                        - 🔵 **Rectangles bleus**: Divisions du tableau
                        - 🟢 **Ellipses vertes**: Fusions des sous-tableaux
                        """)
                        
                        try:
                            tree_viz = create_clean_merge_tree(tree_root)
                            st.graphviz_chart(tree_viz)
                        except Exception as e:
                            st.error(f"Erreur lors de la création de l'arbre: {e}")

                    # Affichage des étapes détaillées
                    if show_steps and steps:
                        st.write("### 🔍 Étapes Détaillées du Tri")
                        fig_steps = visualize_sorting_steps(steps, descriptions, "Étapes du Tri Fusion")
                        if fig_steps:
                            st.pyplot(fig_steps)
                            plt.close(fig_steps)

                        # Détail des étapes
                        st.write("#### 📋 Détail des Étapes")
                        for i, (step, desc) in enumerate(zip(steps, descriptions)):
                            with st.expander(f"Étape {i+1}: {desc}"):
                                st.write(f"Tableau: {step}")

                    # Comparaison avant/après
                    if not show_steps or len(steps) <= 2:
                        fig_comparison = visualize_comparison(original_array, sorted_array, execution_time)
                        st.pyplot(fig_comparison)
                        plt.close(fig_comparison)

            else:
                st.warning("⚠️ Veuillez entrer un tableau valide")

def show_performance_analysis():
    """Affiche l'analyse de performance"""
    st.subheader("📊 Analyse de Performance")
    
    st.write("""
    Cette section analyse les performances du tri fusion sur différentes tailles de tableaux.
    L'algorithme a une complexité théorique de **O(n log n)**.
    """)

    if st.button("🚀 Lancer l'Analyse de Performance", type="primary"):
        with st.spinner("Analyse en cours..."):
            sizes, times = analyze_performance()
            fig_perf = plot_performance(sizes, times)
            st.pyplot(fig_perf)
            plt.close(fig_perf)

            st.write("### 📋 Résultats Détaillés")
            results_data = {"Taille du tableau": sizes, "Temps (secondes)": [f"{t:.6f}" for t in times]}
            st.table(results_data)

# =============================================================================
# Integration
# =============================================================================

# Pour intégrer avec votre app principale, ajoutez dans tp_algo_app.py:
# from tp3_heap_sort import show_tp3
# puis dans la navigation: elif selected_tp == "TP3": show_tp3()

if __name__ == "__main__":
    show_tp3()
