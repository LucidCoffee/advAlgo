import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
from steinlib.instance import SteinlibInstance
from steinlib.parser import SteinlibParser
from steiner import *

class MySteinlibInstance(SteinlibInstance):
    def __init__ (self):
        self.my_graph = nx.Graph()
        self.terms = []
    def terminals__t (self, line, converted_token):
        self.terms.append(converted_token[0])
    def graph__e (self, line, converted_token):
        e_start = converted_token[0]
        e_end = converted_token[1]
        weight = converted_token[2]
        self.my_graph.add_edge(e_start, e_end, weight=weight)

def load(stein_file="src/data/B/b04.stp"):
    """
    load the steinerlib instance at the path `stein_file`
    """   
    my_class = MySteinlibInstance()
    with open(stein_file) as my_file :
        my_parser = SteinlibParser (my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
    return graph, terms

def load_all_from_folder(folder, bests, prefix):
    """
    return the list of the graph in the folder
    together with their terminal and best known solution
    the graph are of the form "folder + perfix + i.stp"
    """
    res = []
    for i in range(1, len(bests)):
        G, term = load(folder + prefix + f"{i:02}.stp")
        res.append((G, term, bests[i-1]))
    return res

def load_all_B():
    """
    return the list of the graph in the /data/B/ folder
    together with their terminal and best known solution
    """
    bests = [82, 83, 138, 59, 61, 122, 111, 104, 220, 86, 88, 174, 165, 235, 318, 127, 131, 218]
    return load_all_from_folder("src/data/B/", bests, prefix="b")

def load_all_C():
    """
    return the list of the graph in the /data/C/ folder
    together with their terminal and best known solution
    """
    bests = [85, 144, 754, 1079, 1579, 55, 102, 509, 707, 1093, 32, 46, 258, 323, 556, 11, 18, 113, 146, 267]
    return load_all_from_folder("src/data/C/", bests, prefix="c")

def best_worst_case(epsilon=0.1):
    """
    return the Graph and the terminals 
    which make the ratio between approximation and optimal above 2-epsilon
    """
    n = math.ceil((4-epsilon)/epsilon)
    k = math.ceil((2-epsilon)/epsilon)
    G = nx.Graph()
    G.add_node(n)
    for i in range(n):
        G.add_node(i)
        for j in range(i):
            G.add_edge(i,j, weight=2*k+1)
        G.add_edge(n, i, weight=k+1)
    term = [i for i in range(n)]
    return G, term

def random_binomial_graph(n, p):
    G = nx.gnp_random_graph(n, p)
    while not nx.is_connected(G):
        G = nx.gnp_random_graph(n, p)
    for u, v in G.edges():
        G[u][v]["weight"] = random.randint(1, 10)
    return G

def generate_random_graph_series(n, s, p):
    """
    return a list of n random graph using the binomial model
    """
    graphs = []
    for _ in range(n):
        G = random_binomial_graph(s, p)
        terminals = set(random.sample(list(G.nodes()), max(2, int(s * 0.1)))) 
        graphs.append((G, terminals))
    return graphs

def plot_graph(G, terminals):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=10, font_weight="bold", edge_color='gray')
    terminal_nodes = list(terminals)
    nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes, node_size=500, node_color="red")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def plot_avgs(graphs, terminal_percentages):
    avg_approx = []
    avg_sim_ann = []
    # avg_branch_bound = []
    avg_shortest_path_bound = []
    avg_minimum_tree_bound = []
    
    for pct in terminal_percentages:
        
        if pct == 0:
            continue

        approx_costs = []
        sim_ann_costs = []
        # branch_bound_costs = []
        sp_bounds = []
        min_tree_bounds = []
        
        for G, _ in graphs:
            n = G.number_of_nodes()
            num_terminals = max(1, int(pct * n / 100))
            terminals = set(random.sample(list(G.nodes()), num_terminals))
            
            approx_edges = two_approx_steiner_tree(G, terminals)
            approx_costs.append(eval_solution_cost(approx_edges))
            
            _, sim_ann_cost, _ = simulated_annealing_steiner(G, terminals, random_init=False)
            sim_ann_costs.append(sim_ann_cost)
            
            # branch_bound_edges = branch_and_bound(G, terminals)
            # branch_bound_costs.append(eval_solution_cost(branch_bound_edges))
            sp_bounds.append(born_inf(G, list(terminals)))
            # min_tree_bounds.append(minimum_tree_bound(G, terminals))
        
        avg_approx.append(sum(approx_costs) / len(graphs))
        avg_sim_ann.append(sum(sim_ann_costs) / len(graphs))
        # avg_branch_bound.append(sum(branch_bound_costs) / len(graphs))
        avg_shortest_path_bound.append(sum(sp_bounds) / len(graphs))
        # avg_minimum_tree_bound.append(sum(min_tree_bounds) / len(graphs))
        print("end of ", pct)
    
    plt.plot(terminal_percentages[1:], avg_approx, label="2-approx", marker='v')
    plt.plot(terminal_percentages[1:], avg_sim_ann, label="heuristic", marker='s')
    # plt.plot(terminal_percentages[1:], avg_branch_bound, label="branch_and_bound", marker='d')
    plt.plot(terminal_percentages[1:], avg_shortest_path_bound, label="born-3-term", marker='o')
    # plt.plot(terminal_percentages[1:], avg_minimum_tree_bound, label="bound_minimum_tree", marker='x')
    plt.xlabel("Percentage of terminals")
    plt.ylabel("Average weight")
    plt.title("Comparison of Solutions and Bounds with Heuristics")
    plt.legend()
    plt.show()

def plot_steinlib_approx(steinlib_graphs, prefix="b"):
    x_labels = [(prefix + f"{i:02}") for i in range(1, len(steinlib_graphs) +1)]
    x = np.arange(len(x_labels))           
    
    approx_values = [eval_solution_cost(two_approx_steiner_tree(G, term)) for (G,term,_) in steinlib_graphs]     
    born_values = [born_inf(G, term) for (G, term,_) in steinlib_graphs]
    best_values = [best for (_,_,best) in steinlib_graphs]

    # Bar width and positions
    bar_width = 0.25
    x_approx = x - bar_width 
    x_born = x + bar_width                
    x_best = x                                  

    # Plot the grouped bar graph
    plt.figure(figsize=(12, 6))
    plt.bar(x_approx, approx_values, width=bar_width, label='approx', color='skyblue')
    plt.bar(x_best, best_values, width=bar_width, label='optimal', color='salmon')
    plt.bar(x_born, born_values, width=bar_width, label='best_born', color='lightgreen')

    plt.ylabel("total weight", fontsize=12)
    plt.xticks(x, x_labels, rotation=45, ha='right', fontsize=10)  # Set x-axis labels
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_steinlib_approx_heuristic(steinlib_graphs, prefix="b"):
    x_labels = [(prefix + f"{i:02}") for i in range(1, len(steinlib_graphs) +1)]
    x = np.arange(len(x_labels))           
    
    approx_values = [eval_solution_cost(two_approx_steiner_tree(G, term)) for (G,term,_) in steinlib_graphs] 
    heuristic_values = [simulated_annealing_steiner(G, term)[1] for (G, term,_) in steinlib_graphs]
    best_values = [best for (_,_,best) in steinlib_graphs]
    # Bar width and positions
    bar_width = 0.25
    x_approx = x - bar_width 
    x_born = x + bar_width                
    x_heur = x                                  

    # Plot the grouped bar graph
    plt.figure(figsize=(12, 6))
    plt.bar(x_approx, approx_values, width=bar_width, label='approx', color='skyblue')
    plt.bar(x_heur, best_values, width=bar_width, label='optimal', color='salmon')
    plt.bar(x_born, heuristic_values, width=bar_width, label='heuristic', color='lightgreen')

    plt.ylabel("total weight", fontsize=12)
    plt.xticks(x, x_labels, rotation=45, ha='right', fontsize=10)  # Set x-axis labels
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_graphs = 50   # Graph number
    s = 25          # Graph size
    p = 0.2         # Edge probability
    graphs = generate_random_graph_series(n_graphs, s, p)
    terminal_percentages = list(range(0, 101, 10))
    
    # G, term = best_worst_case(0.6)
    # approx_sol = two_approx_steiner_tree(G, term)
    # print("approx = ", eval_solution_cost(approx_sol))
    # heur_sol, heur_cost, result = simulated_annealing_steiner(G, term)
    # print("heuristic = ", heur_cost)
    # xpoints = [i for (i,_) in result]
    # ypoints = [v for (_,v) in result]
    # plt.xlabel("nb rounds")
    # plt.ylabel("total distance")
    # plt.plot(xpoints, ypoints)
    # plt.show()
    # branch_sol = branch_and_bound(G, term)
    # print("branch =", eval_solution_cost(branch_sol))
    # plot_steinlib_approx_heuristic(load_all_B(), "b")
    # plot_avgs(graphs, terminal_percentages)
    # plot_avg_approx(load_all_C(), "c")
    # plot_graph(G, term)
    # best_sol, best_v = branchAndBound(G, term)
    # plot_graph(best_sol, term)
