import streamlit as st
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

# ------------------ Algorithme de Matula ------------------ #
def matula_with_history(G):
    temp = G.copy()
    degrees = {v: temp.degree(v) for v in temp.nodes()}
    initial_degrees = degrees.copy()
    order, degree_history = [], {}

    while temp.nodes():
        v = min(temp.nodes(), key=lambda x: degrees[x])
        order.append(v)
        degree_history[v] = degrees[v]
        temp.remove_node(v)
        for u in temp.nodes():
            degrees[u] = temp.degree(u)

    coloring = {}
    for v in reversed(order):
        used = {coloring[u] for u in G.neighbors(v) if u in coloring}
        c = 1
        while c in used:
            c += 1
        coloring[v] = c

    return list(reversed(order)), initial_degrees, degree_history, coloring

# ------------------ Construction du tableau ------------------ #
def build_table(order, initial_deg, degree_history, coloring):
    max_c = max(coloring.values())
    data = {
        "Sommet trié": order,
        "Degré initial": [initial_deg[v] for v in order],
        "Degré Matula": [degree_history[v] for v in order],
    }

    for c in range(1, max_c + 1):
        data[f"C{c}"] = ["✅" if coloring[v] == c else "❌" for v in order]

    df = pd.DataFrame(data)
    df_transposed = df.T
    df_transposed.columns = [f"S{v}" for v in order]
    return df_transposed

# ------------------ Interface Streamlit ------------------ #
def show_tp4():
    st.set_page_config(
        page_title="TP4 - Algorithme Matula",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("TP4 : Coloration de Graphe - Algorithme de Matula 🎨")

    n = st.number_input("Nombre de sommets :", 1, 50, 6)
    arretes_txt = st.text_area("Arêtes (format : 1-2,1-3,2-4) :", "1-2,1-3,2-4,3-5,4-6")

    if st.button("Colorer le graphe"):
        edges = []
        for part in arretes_txt.split(","):
            if "-" in part:
                a, b = map(int, part.split("-"))
                if 1 <= a <= n and 1 <= b <= n and a != b:
                    edges.append((a, b))

        G = nx.Graph()
        G.add_nodes_from(range(1, n+1))
        G.add_edges_from(edges)

        order, init_deg, mat_deg, coloring = matula_with_history(G)

        # --- Affichage du tableau ---
        df_sommets = build_table(order, init_deg, mat_deg, coloring)
        st.subheader("📌 Tableau transposé (colonnes ↔ lignes)")
        st.table(df_sommets)

        # --- Positions des sommets ---
        pos = nx.spring_layout(G, seed=42)

        # Création des arêtes
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines"
        )

        # Création des sommets
        node_x, node_y, node_color, node_text = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append(coloring[node])
            node_text.append(f"Sommet {node} - Couleur {coloring[node]}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=[str(n) for n in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale="Viridis",  # ✅ palette Plotly native
                color=node_color,
                size=30,
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=20,l=5,r=5,t=40),
                        ))

        st.plotly_chart(fig, use_container_width=True)

