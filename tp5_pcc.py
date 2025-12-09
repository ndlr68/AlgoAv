# tp5_pcc_ameliore.py
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def bellman_ford_detailed(vertices, edges, source):
    """
    Implémente l'algorithme de Bellman-Ford selon la description du document.
    
    Args:
        vertices: liste des sommets
        edges: liste des arêtes sous forme [(u, v, weight), ...]
        source: sommet de départ
    
    Returns:
        dict: distances finales
        dict: prédécesseurs
        bool: True si un circuit absorbant est détecté
        str: message d'erreur
        list: historique des λ^k(i) pour chaque itération
        list: historique des sommets marqués
    """
    n = len(vertices)
    
    # Initialisation
    lambda_k = {vertex: float('inf') for vertex in vertices}
    lambda_k[source] = 0
    predecessor = {vertex: None for vertex in vertices}
    
    # Pour stocker l'historique
    history = []
    marked_history = []
    
    # Étape k=0
    lambda_0 = lambda_k.copy()
    history.append(lambda_0)
    marked_history.append({source})
    
    # Marqueurs initiaux
    M = {source}  # Ensemble des sommets marqués
    
    # Itérations de 1 à n-1
    for k in range(1, n + 1):  # On va jusqu'à n pour détecter les circuits absorbants
        M_new = set()
        lambda_new = lambda_k.copy()
        
        # Pour chaque sommet marqué dans M, examiner ses successeurs
        for u in M:
            # Trouver tous les successeurs de u
            successors = [(v, w) for (x, v, w) in edges if x == u]
            
            for v, w in successors:
                new_value = lambda_k[u] + w
                if new_value < lambda_new[v]:
                    lambda_new[v] = new_value
                    predecessor[v] = u
                    M_new.add(v)
        
        # Ajouter à l'historique
        history.append(lambda_new.copy())
        marked_history.append(M_new.copy())
        
        # Si pas de nouveaux sommets marqués, on peut arrêter
        if not M_new:
            if k < n:
                # Remplir les itérations restantes avec les mêmes valeurs
                for _ in range(k, n):
                    history.append(lambda_new.copy())
                    marked_history.append(set())
            break
        
        # Mettre à jour pour la prochaine itération
        lambda_k = lambda_new
        M = M_new
    
    # Vérification des circuits absorbants
    has_negative_cycle = False
    error_msg = ""
    
    # Vérifier une dernière fois s'il y a encore des améliorations (circuit absorbant)
    if len(history) > n:  # Si on a fait n itérations
        # Comparer les valeurs de la n-ième et (n-1)-ième itération
        last = history[-1]
        second_last = history[-2] if len(history) > 1 else last
        
        for vertex in vertices:
            if last[vertex] < second_last[vertex]:
                has_negative_cycle = True
                error_msg = f"Circuit absorbant détecté au sommet {vertex}"
                break
    
    return lambda_k, predecessor, has_negative_cycle, error_msg, history, marked_history

def draw_graph_bellman_ford(vertices, edges, distances=None, source=None, predecessors=None, title="Graphe"):
    """
    Dessine le graphe avec les distances et chemins.
    """
    G = nx.DiGraph()
    
    # Ajouter les sommets
    G.add_nodes_from(vertices)
    
    # Ajouter les arêtes avec poids
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    # Positionnement
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Couleurs des nœuds
    node_colors = []
    for node in vertices:
        if node == source:
            node_colors.append('#ff6b6b')  # Source en rouge
        elif distances and distances[node] < float('inf'):
            node_colors.append('#4ecdc4')  # Accessible en vert
        else:
            node_colors.append('#c7c7c7')  # Inaccessible en gris
    
    # Dessiner les nœuds
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, 
                          alpha=0.9, ax=ax, edgecolors='black')
    
    # Dessiner les étiquettes des nœuds
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Dessiner les arêtes
    for u, v, w in edges:
        # Couleur des arêtes selon si elles font partie du chemin
        edge_color = '#2d3436'
        style = 'solid'
        width = 1
        
        if predecessors and predecessors.get(v) == u:
            edge_color = '#e17055'
            style = 'dashed'
            width = 2
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                              edge_color=edge_color, style=style,
                              width=width, arrowsize=20, ax=ax,
                              connectionstyle='arc3,rad=0.1')
        
        # Position du label de l'arête
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        
        # Petit décalage pour éviter la superposition
        offset = 0.05
        if u < v:
            y += offset
        else:
            y -= offset
        
        ax.text(x, y, str(w), bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', alpha=0.9),
               fontsize=10, ha='center', va='center')
    
    # Ajouter les distances si disponibles
    if distances and source:
        distance_labels = {}
        for node in vertices:
            if distances[node] == float('inf'):
                distance_labels[node] = "d=∞"
            else:
                distance_labels[node] = f"d={distances[node]}"
        
        label_pos = {k: (v[0], v[1] - 0.15) for k, v in pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels=distance_labels,
                               font_size=10, font_color='#d63031', ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def create_iteration_table_bellman_ford(history, marked_history, vertices):
    """
    Crée un tableau des itérations Bellman-Ford selon le format du document.
    """
    html = """
    <style>
    .bellman-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-family: 'Courier New', monospace;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .bellman-table th {
        background-color: #2c3e50;
        color: white;
        padding: 12px;
        text-align: center;
        border: 1px solid #34495e;
    }
    .bellman-table td {
        padding: 10px;
        text-align: center;
        border: 1px solid #ddd;
    }
    .iteration-header {
        background-color: #3498db !important;
        color: white !important;
        font-weight: bold;
    }
    .marked-cell {
        background-color: #ffeaa7 !important;
        color: #d63031 !important;
        font-weight: bold;
        position: relative;
    }
    .marked-cell::after {
        content: "(*)";
        font-size: 10px;
        position: absolute;
        top: 2px;
        right: 2px;
    }
    .source-cell {
        background-color: #ff7675 !important;
        color: white !important;
    }
    .infinity {
        color: #7f8c8d;
        font-style: italic;
    }
    </style>
    <table class="bellman-table">
        <thead>
            <tr>
                <th>k</th>
    """
    
    # En-têtes pour chaque sommet
    for vertex in vertices:
        html += f'<th>λᵏ({vertex})</th>'
    
    html += '</tr></thead><tbody>'
    
    # Lignes pour chaque itération
    for k, (lambda_values, marked_set) in enumerate(zip(history, marked_history)):
        # Label de l'itération
        if k == 0:
            row_label = "0 (init)"
            row_class = "iteration-header source-cell"
        elif k == len(history) - 1:
            row_label = f"{k} (final)"
            row_class = "iteration-header"
        else:
            row_label = str(k)
            row_class = ""
        
        html += f'<tr>'
        html += f'<td class="{row_class}" style="font-weight: bold;">{row_label}</td>'
        
        for vertex in vertices:
            value = lambda_values[vertex]
            cell_class = ""
            
            if vertex in marked_set:
                cell_class = "marked-cell"
            elif k == 0 and value == 0:
                cell_class = "source-cell"
            
            if value == float('inf'):
                display_value = '∞'
                if not cell_class:
                    cell_class = "infinity"
            else:
                display_value = str(value)
            
            html += f'<td class="{cell_class}">{display_value}</td>'
        
        html += '</tr>'
    
    html += '</tbody></table>'
    return html

def show_tp5_ameliore():
    """
    Interface améliorée pour le TP5 - Algorithme de Bellman-Ford
    """

    st.markdown("""
    **Objectif** : Implémenter l'algorithme de Bellman-Ford pour trouver les plus courts chemins 
    depuis un sommet source dans un graphe pondéré, avec détection des circuits absorbants.
    """)
    
    # Initialisation de l'état de session
    if 'tp5_vertices' not in st.session_state:
        st.session_state.tp5_vertices = ['A', 'B', 'C', 'D', 'E']
    if 'tp5_edges' not in st.session_state:
        st.session_state.tp5_edges = [
            ('A', 'B', 4),
            ('A', 'C', 2),
            ('B', 'C', 3),
            ('B', 'D', 2),
            ('B', 'E', 3),
            ('C', 'B', 1),
            ('C', 'D', 4),
            ('C', 'E', 5),
            ('E', 'D', -5)
        ]
    
    # Déplacer la création des tabs en dehors du bloc conditionnel
    tab1, tab2, tab3 = st.tabs(["📊 Éditeur du graphe", "🔄 Exécution", "📚 Théorie"])

    # Sidebar pour la configuration
    with tab1:
        st.header("⚙️ Configuration du graphe")
        
        # Saisie des sommets
        st.subheader("Sommets")
        vertices_input = st.text_input(
            "Liste des sommets (séparés par des virgules):",
            value=", ".join(st.session_state.tp5_vertices),
            help="Exemple: A, B, C, D, E"
        )
        
        if st.button("Mettre à jour les sommets"):
            vertices = [v.strip() for v in vertices_input.split(',') if v.strip()]
            if vertices:
                st.session_state.tp5_vertices = vertices
                st.success(f"{len(vertices)} sommets définis")
            else:
                st.error("Veuillez entrer au moins un sommet")
        
        # Exemples prédéfinis
        st.subheader("Exemples")
        examples = {
            "Exemple 1 (sans circuit négatif)": {
                'vertices': ['A', 'B', 'C', 'D', 'E'],
                'edges': [
                    ('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 3),
                    ('B', 'D', 2), ('B', 'E', 3), ('C', 'B', 1),
                    ('C', 'D', 4), ('C', 'E', 5), ('E', 'D', -5)
                ]
            },
            "Exemple 2 (avec circuit absorbant)": {
                'vertices': ['A', 'B', 'C', 'D'],
                'edges': [
                    ('A', 'B', 1), ('B', 'C', 3),
                    ('C', 'D', 2), ('D', 'B', -6)
                ]
            },
            "Exemple 3 (graphe simple)": {
                'vertices': ['1', '2', '3', '4'],
                'edges': [
                    ('1', '2', 5), ('1', '3', 3),
                    ('2', '4', 2), ('3', '2', 1),
                    ('3', '4', 6)
                ]
            }
        }
        
        selected_example = st.selectbox("Charger un exemple:", list(examples.keys()))
        if st.button("Charger cet exemple"):
            example = examples[selected_example]
            st.session_state.tp5_vertices = example['vertices']
            st.session_state.tp5_edges = example['edges']
            st.success(f"Exemple '{selected_example}' chargé!")
    
    # Section principale
    
    with tab1:
        st.header("Éditeur du graphe")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Éditeur de matrice d'adjacence
            st.subheader("Matrice d'adjacence")
            vertices = st.session_state.tp5_vertices
            n = len(vertices)
            
            # Créer une matrice vide
            matrix_data = []
            for i in range(n):
                row = []
                for j in range(n):
                    # Chercher le poids de l'arête correspondante
                    weight = None
                    for u, v, w in st.session_state.tp5_edges:
                        if u == vertices[i] and v == vertices[j]:
                            weight = w
                            break
                    row.append(weight if weight is not None else '')
                matrix_data.append(row)
            
            # Afficher la matrice éditable
            matrix_df = pd.DataFrame(matrix_data, 
                                    index=vertices, 
                                    columns=vertices)
            
            edited_matrix = st.data_editor(
                matrix_df,
                use_container_width=True,
                height=400,
                column_config={
                    col: st.column_config.NumberColumn(
                        label=col,
                        width="small",
                        min_value=-100,
                        max_value=100,
                        step=1,
                        format="%d"
                    ) for col in vertices
                }
            )
            
            if st.button("Mettre à jour les arêtes depuis la matrice"):
                new_edges = []
                for i in range(n):
                    for j in range(n):
                        value = edited_matrix.iloc[i, j]
                        if value != '' and pd.notna(value):
                            new_edges.append((vertices[i], vertices[j], int(value)))
                st.session_state.tp5_edges = new_edges
                st.success(f"{len(new_edges)} arêtes mises à jour!")
        
        with col2:
            st.subheader("Éditeur manuel")
            
            # Ajouter une nouvelle arête
            st.write("**Ajouter une arête:**")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                from_v = st.selectbox("De:", vertices, key="from_v")
            with col_b:
                to_v = st.selectbox("À:", vertices, key="to_v")
            with col_c:
                weight = st.number_input("Poids:", value=0, step=1, key="new_weight")
            
            if st.button("➕ Ajouter l'arête"):
                if from_v == to_v:
                    st.error("Impossible d'ajouter une boucle!")
                else:
                    new_edge = (from_v, to_v, weight)
                    if new_edge not in st.session_state.tp5_edges:
                        st.session_state.tp5_edges.append(new_edge)
                        st.success(f"Arête {from_v}→{to_v} (poids={weight}) ajoutée")
                    else:
                        st.warning("Cette arête existe déjà")
            
            # Supprimer une arête
            st.write("**Supprimer une arête:**")
            if st.session_state.tp5_edges:
                edge_options = [f"{u} → {v} (poids={w})" 
                              for u, v, w in st.session_state.tp5_edges]
                edge_to_delete = st.selectbox("Sélectionner une arête:", edge_options)
                
                if st.button("🗑️ Supprimer"):
                    idx = edge_options.index(edge_to_delete)
                    st.session_state.tp5_edges.pop(idx)
                    st.success("Arête supprimée!")
            
            # Bouton pour tout supprimer
            if st.button("🧹 Tout effacer"):
                st.session_state.tp5_edges = []
                st.success("Toutes les arêtes ont été supprimées!")
        
        # Visualisation du graphe
        st.subheader("Visualisation du graphe")
        if st.session_state.tp5_vertices and st.session_state.tp5_edges:
            try:
                fig = draw_graph_bellman_ford(
                    st.session_state.tp5_vertices,
                    st.session_state.tp5_edges,
                    title="Graphe défini"
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur lors du dessin: {e}")
        else:
            st.info("Définissez des sommets et des arêtes pour visualiser le graphe.")
    
    with tab2:
        st.header("Exécution de Bellman-Ford")
        
        if not st.session_state.tp5_vertices:
            st.warning("Veuillez d'abord définir des sommets dans l'onglet 'Éditeur du graphe'.")
            return
        
        # Sélection de la source
        source = st.selectbox(
            "Sélectionner le sommet source:",
            st.session_state.tp5_vertices,
            key="source_select"
        )
        
        # Bouton d'exécution
        if st.button("🚀 Exécuter Bellman-Ford", type="primary"):
            with st.spinner("Calcul en cours..."):
                distances, predecessors, has_cycle, error_msg, history, marked_history = bellman_ford_detailed(
                    st.session_state.tp5_vertices,
                    st.session_state.tp5_edges,
                    source
                )
                
                # Résultats
                st.subheader("📈 Résultats")
                
                if has_cycle:
                    st.error(f"**Circuit absorbant détecté!** {error_msg}")
                    st.warning("Les résultats peuvent être incorrects à cause d'un circuit de poids négatif.")
                else:
                    st.success("✅ Aucun circuit absorbant détecté.")
                
                # Tableau des distances finales
                st.write("**Distances finales depuis la source:**")
                results_data = []
                for vertex in st.session_state.tp5_vertices:
                    dist = distances[vertex]
                    pred = predecessors[vertex]
                    results_data.append({
                        'Sommet': vertex,
                        'Distance': dist if dist != float('inf') else '∞',
                        'Prédécesseur': pred if pred else '-',
                        'Chemin': reconstruct_path(source, vertex, predecessors)
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Tableau des itérations
                st.subheader("🔄 Tableau des itérations Bellman-Ford")
                st.markdown("""
                **Légende:**
                - **(*)** : Sommet marqué (mis à jour lors de cette itération)
                - **Cellule rouge** : Sommet source (k=0)
                - **Cellule jaune** : Sommet marqué à cette itération
                """)
                
                table_html = create_iteration_table_bellman_ford(
                    history, marked_history, st.session_state.tp5_vertices
                )
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Bouton pour afficher les explications détaillées
                with st.expander("📖 Explications détaillées des itérations"):
                    st.markdown(get_detailed_explanations(history, marked_history, source))
                
                # Visualisation du résultat
                st.subheader("🎯 Graphe avec distances finales")
                try:
                    fig_result = draw_graph_bellman_ford(
                        st.session_state.tp5_vertices,
                        st.session_state.tp5_edges,
                        distances,
                        source,
                        predecessors,
                        "Résultat de Bellman-Ford"
                    )
                    st.pyplot(fig_result)
                except Exception as e:
                    st.error(f"Erreur lors du dessin: {e}")
                
                # Chemins détaillés
                st.subheader("🗺️ Chemins les plus courts")
                for vertex in st.session_state.tp5_vertices:
                    if vertex == source:
                        continue
                    
                    path = reconstruct_path(source, vertex, predecessors)
                    if path:
                        dist = distances[vertex]
                        if dist != float('inf'):
                            path_str = " → ".join(path)
                            st.write(f"**{source} → {vertex}** : {path_str} (distance = {dist})")
                        else:
                            st.write(f"**{source} → {vertex}** : ❌ Aucun chemin accessible")
    
    with tab3:
        st.header("Théorie de l'algorithme Bellman-Ford")
        
        st.markdown("""
        ### Principe de l'algorithme
        
        L'algorithme de Bellman-Ford permet de trouver les plus courts chemins depuis un sommet source
        dans un graphe orienté pondéré, même avec des poids négatifs.
        
        **Notation :**
        - λᵏ(v) : distance minimale du sommet source à v en utilisant au plus k arcs
        - M : ensemble des sommets "marqués" (dont la distance a été améliorée)
        
        ### Étapes de l'algorithme
        
        1. **Initialisation (k=0) :**
           - λ⁰(source) = 0
           - λ⁰(v) = ∞ pour tout v ≠ source
           - M = {source}
        
        2. **Itérations (k=1 à n-1) :**
           - Pour chaque sommet u dans M :
             - Pour chaque successeur v de u :
               - Si λᵏ⁻¹(u) + w(u,v) < λᵏ⁻¹(v) :
                 - λᵏ(v) = λᵏ⁻¹(u) + w(u,v)
                 - Ajouter v à M'
           - M = M'
           - Si M est vide : arrêter
        
        3. **Détection des circuits absorbants :**
           - Si à l'itération n il y a encore des améliorations, alors il existe un circuit absorbant
        
        ### Complexité
        - Temps : O(n × m) où n = nombre de sommets, m = nombre d'arcs
        - Espace : O(n)
        
        ### Applications
        - Routage dans les réseaux
        - Détection d'arbitrage en finance
        - Calcul de distances dans les graphes avec poids négatifs
        """)
        


def reconstruct_path(source, target, predecessors):
    """Reconstruit le chemin de source à target."""
    if predecessors[target] is None and target != source:
        return []
    
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    
    return path if path[0] == source else []

def get_detailed_explanations(history, marked_history, source):
    """Génère des explications détaillées pour chaque itération."""
    explanations = []
    
    for k, (lambda_values, marked_set) in enumerate(zip(history, marked_history)):
        if k == 0:
            explanations.append(f"**Itération k={k} (Initialisation) :**")
            explanations.append(f"- λ⁰({source}) = 0 (source)")
            explanations.append(f"- λ⁰(v) = ∞ pour tous les autres sommets")
        else:
            explanations.append(f"**Itération k={k} :**")
            
            if marked_set:
                explanations.append(f"Sommets mis à jour : {', '.join(sorted(marked_set))}")
                
                for vertex in sorted(marked_set):
                    prev_val = history[k-1][vertex]
                    new_val = lambda_values[vertex]
                    if prev_val == float('inf'):
                        prev_disp = '∞'
                    else:
                        prev_disp = str(prev_val)
                    
                    explanations.append(f"- λᵏ({vertex}) = {new_val} (était {prev_disp})")
            else:
                if k < len(history) - 1:
                    explanations.append("Aucune mise à jour - l'algorithme peut s'arrêter.")
                else:
                    explanations.append("Aucune mise à jour - calcul terminé.")
        
        explanations.append("---")
    
    return "\n".join(explanations)

def show_tp5():
    # Configuration de la page
    st.set_page_config(
        page_title="TP5 - Bellman-Ford",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Style CSS personnalisé
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        padding: 20px;
        background: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background: #FF3838;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background:#8BAE66;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #3498db;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown('<h1 class="main-header"> TP5 - Algorithme de Bellman-Ford</h1>', 
                unsafe_allow_html=True)
    
    # Afficher l'interface
    show_tp5_ameliore()
    
    # Footer
    st.markdown("---")


if __name__ == "__main__":
    show_tp5()