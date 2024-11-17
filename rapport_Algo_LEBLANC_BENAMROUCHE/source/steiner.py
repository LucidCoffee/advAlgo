import networkx as nx
import random
import math

### Constants

PROBA_MUTA=0.75 # probability of changing a random 1 into a 0 on the solution in random_neighbor
TI = 10   # initial temperature of the simluated annealing 
TE = 0.1  # end temperature of the simluated annealing 
NBITER = 3000 # nb iteration of the simluated annealing 

#### approximation part

def complete(G, terminals):
    G_prime = nx.Graph()
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    for u in terminals:
        for v in terminals:
            if u != v:
                G_prime.add_edge(u, v, weight=lengths[u][v])
    return G_prime

def deploy(G, T):
    result = set()
    for u, v in T.edges():
        path = nx.shortest_path(G, source=u, target=v, weight='weight')
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if (a, b) not in result and (b, a) not in result: 
                result.add((a, b))
    solution_edges = []
    for u, v in result:
        solution_edges.append((u, v, G[u][v]['weight']))
    return solution_edges

def two_approx_steiner_tree(G, terminals):
    G_prime = complete(G, terminals)
    mst = nx.minimum_spanning_tree(G_prime, weight='weight')
    return list(deploy(G, mst))

def eval_solution_cost(approx):
    return sum(w for _,_,w in approx)

#### heuristic part

def force_terminals(term,node_to_index, sol):
    """
    put the terminal to 1 in the solution
    """
    for i in term:
        sol[node_to_index[i]] = 1
    return sol

def convert_node(edge_sol, node_to_index):
    """
    convert a list of edges into an array of bit, each node is associated to an index in `node_to_index`
    """
    node_sol = [0 for _ in range(len(node_to_index))]
    for u,v,_ in edge_sol:
        node_sol[node_to_index[u]] = 1
        node_sol[node_to_index[v]] = 1
    return node_sol

def eval_sol(distances, index_to_node, sol:list):
    """
    create a complete graph with all nodes in sol and the shortest path between them (stored in distances), 
    and then return the weight of a minimal spanning tree in it
    """
    graph_sol = nx.Graph()
    for (i, b) in enumerate(sol):
        if b == 1:
            graph_sol.add_node(i)
    
    for i in graph_sol.nodes():
        for j in graph_sol.nodes():
            if i != j:
                graph_sol.add_edge(i, j, weight=distances[index_to_node[i]][index_to_node[j]])
    
    span_tree = nx.algorithms.minimum_spanning_tree(graph_sol)
    
    return span_tree.size(weight="weight")

def random_sol(term,node_to_index, graph, proba=0.5):
    """
    return a random solution
    param proba -> probability of each bit to be 1
    """
    nodes = list(graph.nodes())
    sol = [0 for _ in nodes]
    for i in range(len(nodes)):
        if random.random() < proba:
            sol[i] = 1
    
    return force_terminals(term, node_to_index, sol)

def random_neighbor(term, node_to_index, sol):
    """
    create a neighbor of sol, it will still have the terminals in it
    """
    def is_full(sol): # return true if all 1
        for y in sol:
            if y == 0 : return False
        return True
    
    new_sol = list(sol)
    # if it's full, there is nothing to swap
    if not is_full(sol):
        i = random.randint(0, len(new_sol)-1)
        j = random.randint(0, len(new_sol)-1)
        while new_sol[i] == new_sol[j]:
            i = random.randint(0, len(new_sol)-1)
            j = random.randint(0, len(new_sol)-1)
        # random swap
        new_sol[i], new_sol[j] = new_sol[j], new_sol[i] 
    

    if random.random() <= PROBA_MUTA:
        # random mutation
        i = random.randint(0, len(new_sol)-1)
        while new_sol[i] == 0:
            i = random.randint(0, len(new_sol)-1)
        new_sol[i] = 0

    new_sol = force_terminals(term, node_to_index, new_sol)

    return new_sol

def update_temperature(T, Ti, Te, nb_iter):
    """
    linear update, such that at 0 we are at Ti and after nb_iter + 1 we are at Te
    """
    return T*(Te - Ti)/(nb_iter + 1) +Ti

def simulated_annealing(Ti, Te, nb_iter, G, term, random_init=True, spacing=100):
    """
    return (best_sol, best_val, result) where result is a list of all (index, tmp_best_sol) with index%spacing==0
    if random_init is False, we use the approximation as the starting point
    """
    distances = dict(nx.algorithms.all_pairs_dijkstra_path_length(G))
    index_to_node = {i:n for i,n in enumerate(list(G.nodes()))}
    node_to_index = {n:i for i,n in enumerate(list(G.nodes()))}
    result = []
    if random_init:
        s1 = random_sol(term, node_to_index, G)
    else:
        s1 = convert_node(two_approx_steiner_tree(G, term), node_to_index)
    v1 = eval_sol(distances, index_to_node, s1)
    best_sol = s1 
    best_v = v1
    T = Ti
    for i in range(nb_iter):
        if i % spacing == 0 :
            result.append((i, best_v))
        s2 = random_neighbor(term, node_to_index, s1)
        v2 = eval_sol(distances, index_to_node, s2)
        if v2 < v1 : 
            proba = 1
            if v1 < best_v:
                best_sol = s2
                best_v = v2
                
        else:
            proba = math.exp(-(v2 - v1)/T)
        if random.random() < proba:
            s1 = s2
            v1 = v2 
        T = update_temperature(T, Ti, Te, nb_iter)

    return best_sol, best_v, result

def simulated_annealing_steiner(G, term, random_init=True):
    return simulated_annealing(TI, TE, NBITER, G, term, random_init=random_init)


### exact part

def exact_sol_3term(G, distances, t1, t2, t3) -> float:
    """
    return the exact solution with 3 terminals
    """
    min_dist = distances[t1][t2] + distances[t2][t3]
    for n in G.nodes():
        sum_dist = distances[t1][n] + distances[t2][n] + distances[t3][n]
        if sum_dist <= min_dist:
            min_dist = sum_dist 
    return min_dist

def born_inf(G, term)-> float:
    """
    return a borne inf of the exact solution using the max of the exact solution with 3 terminals
    """
    if len(term) == 2:
        return nx.algorithms.dijkstra_path_length(G, term[0], term[1])
    
    distances = dict(nx.algorithms.all_pairs_dijkstra_path_length(G))
    max_triplet = 0
    for i in range(len(term)):
        for j in range(i+1, len(term)):
            for k in range(j+1, len(term)):
                sol = exact_sol_3term(G, distances, term[i],term[j],term[k])
                if sol > max_triplet:
                    max_triplet = sol
    return max_triplet

def find_not_seen(edges, edges_seen):
    for e in edges:
        if e not in edges_seen:
            return e
    return None

def eval_sol(term, current_graph):
    if term[0] not in current_graph.nodes: return -1
    component = nx.node_connected_component(current_graph, term[0])
    for i in term:
        if i not in component:
            return -1
    return current_graph.size('weight')

def lower_bound_sol(G, term, current_graph):
    extended_term = list(term)
    for n in current_graph.nodes:
        if n not in extended_term:
            extended_term.append(n)
    return born_inf(G, extended_term)

def branchAndBound(G:nx.Graph, term):
    approx = two_approx_steiner_tree(G, term)
    best_sol = approx
    upper_bound = eval_solution_cost(approx)
    stack = [(set(), nx.Graph())] # stack with (edges_seen, current_graph)
    edges = set(G.edges())
    while(len(stack) != 0):
        (edges_seen, current_graph) = stack.pop()
        current_eval = eval_sol(term, current_graph)

        # V1 : lower_bound =  lower_bound_sol(G, term, current_graph)
        # V2 :
        lower_bound = current_eval # V3 it's not making a lot of difference
        if lower_bound >= upper_bound:
            continue
            
        if 0 <= current_eval < upper_bound:
            best_sol = current_graph
            upper_bound = current_eval
            print("better sol =", upper_bound) 
            continue 
        
        if(len(edges) > len(edges_seen)):
            (u,v) = find_not_seen(edges, edges_seen)
            new_edges_seen = edges_seen.copy()
            new_edges_seen.add((u,v))
            stack.append((new_edges_seen, current_graph))
            new_graph = current_graph.copy()
            new_graph.add_edge(u,v)
            new_graph[u][v]['weight'] = G[u][v]['weight']
            stack.append((new_edges_seen, new_graph))

    return best_sol, upper_bound

