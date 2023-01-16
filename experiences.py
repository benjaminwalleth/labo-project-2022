import random
import collections
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy
import data_analysis

from local_push import reverse_local_push
from cfe_generator import CFE


# This function is used to read the graph from Amazon.com dataset
def read_graph():
    print('Read graph from dataset...\n')
    return nx.read_gml("./data/interaction_graph.gml")  # Data from Amazon


# This function is used to print stats about the given graph
def graph_stats(g, source_node, print_stats=True):
    reviews = [x for x, y in g.nodes(data=True) if y['type'] == "'review'"]
    items = [x for x, y in g.nodes(data=True) if y['type'] == "'item'"]
    users = [x for x, y in g.nodes(data=True) if y['type'] == "'user'"]
    categories = [x for x, y in g.nodes(data=True) if y['type'] == "'category'"]

    # Printing statistics
    if print_stats:
        print('\n============ Graph statistics ============')
        print(len(reviews), " reviews")
        print(len(users), " users")
        print(len(items), " items")
        print(len(categories), " categories")

        print('\nSource node (user):', source_node)

    neighbors_reviews = []
    neighbors = []
    for n in g.successors(source_node):
        if n != source_node:
            if n in reviews:
                neighbors_reviews.append(n)
            if n in items:
                items.remove(n)  # Remove neighbors items to items (we don't want to target it in recommendation)
            neighbors.append(n)

    if print_stats:
        print('The user wrote', len(neighbors_reviews), 'reviews')
        print('Books available for recommendation: ', len(items))

    # Return number of nodes, items to target and neighbors of the source node
    return len(list(g.nodes)), items, neighbors


def init(g):
    items = [x for x, y in g.nodes(data=True) if y['type'] == "'item'"]
    return len(list(g.nodes)), items


# This function is used to get a subgraph from a source node
def sub_graph_from_node(g, source_node, distance=2):
    all_descendants = nx.descendants_at_distance(g, source_node, 0)
    for i in range(0, distance):
        descendantsI = nx.descendants_at_distance(g, source_node, i)
        all_descendants = all_descendants.union(descendantsI)

    return g.subgraph(all_descendants)


# This function is used to print a graph with different nodes and edges
def print_graph(g):
    reviews = [x for x, y in g.nodes(data=True) if y['type'] == "'review'"]
    items = [x for x, y in g.nodes(data=True) if y['type'] == "'item'"]
    users = [x for x, y in g.nodes(data=True) if y['type'] == "'user'"]
    categories = [x for x, y in g.nodes(data=True) if y['type'] == "'category'"]

    pos = nx.random_layout(g)

    nx.draw_networkx_nodes(g, pos, nodelist=items, node_color='green', node_size=100)
    nx.draw_networkx_nodes(g, pos, nodelist=categories, node_color='yellow', node_size=100)
    nx.draw_networkx_nodes(g, pos, nodelist=reviews, node_color='blue', node_size=100)
    nx.draw_networkx_nodes(g, pos, nodelist=users, node_color='red', node_size=100)
    nx.draw_networkx_edges(g, pos, width=0.3, arrowsize=3)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    plt.show()


# This function is used to pick random nodes to build a reduced graph
def reduce_graph_randomly(g, target_nodes, neighbors, k=100):
    # All items in the graph
    items = [x for x, y in g.nodes(data=True) if y['type'] == "'item'"]

    # Reduce target nodes randomly
    target_nodes = random.choices(target_nodes, k=k)

    # Remove target node with no categories
    for item in target_nodes:
        category = get_categories(g, item)
        if category is None:
            target_nodes.remove(item)

    for item in items:
        if item not in target_nodes and item not in neighbors:
            g.remove_node(item)

    return g, target_nodes


# This function is used to keep nodes from a target category
def reduce_graph_by_category(g, target_nodes, target_category, neighbors):
    # All items in the graph
    items = [x for x, y in g.nodes(data=True) if y['type'] == "'item'"]

    # Remove target node with no matching
    for item in target_nodes:
        category = get_categories(g, item)
        if category != target_category:
            target_nodes.remove(item)

    for item in items:
        if item not in target_nodes and item not in neighbors:
            g.remove_node(item)

    return g, target_nodes


# This function is used to keep only users with many links - more than the 9th decile
def reduce_graph_by_high_user(g):
    users = [x for x, y in g.nodes(data=True) if y['type'] == "'user'"]
    reviews = [x for x, y in g.nodes(data=True) if y['type'] == "'review'"]

    count = 0

    for user in users:
        successors = nx.descendants_at_distance(g, user, 1)
        if len(list(successors)) < numpy.percentile(list_number_successors, 90):
            g.remove_node(user)
            count += 1
            # Remove reviews of the user
            for successor in successors:
                if successor in reviews:
                    g.remove_node(successor)

    print(count, " users removed")
    g.remove_nodes_from(list(nx.isolates(g)))

    return g


# This function is used to get categories for a given item
def get_categories(g, item):
    categories = [x for x, y in g.nodes(data=True) if y['type'] == "'category'"]
    category = None
    for node in g.successors(item):
        if node in categories:
            category = node
    return category


# This function is used to count elements in a given list
def count_categories(list_categories):
    return collections.Counter(list_categories)


# This experience observes the reduction of the original graph by keeping only users with many links
def exp_one_bis():
    list_number_successors = data_analysis.user_neighbor(g)

    user, successors = data_analysis.select_user(g, numpy.percentile(list_number_successors, 90),
                                                 numpy.max(list_number_successors))

    source_node = user

    unfrozen_g = nx.DiGraph(g)  # For modification on the graph, we unfroze it
    h = reduce_graph_by_high_user(unfrozen_g)
    h = sub_graph_from_node(h, source_node, distance=4)

    list_number_successors = data_analysis.user_neighbor(h)
    print('============ Data analysis ============')
    print("Number of neighbors")
    print("- Median: ", numpy.percentile(list_number_successors, 50))
    print("- Average :", numpy.average(list_number_successors))
    print("- Min:", numpy.min(list_number_successors))
    print("- Max:", numpy.max(list_number_successors))

    graph_stats(h, source_node)


# This experience observes the behaviour of the system depending on the type of user (about the number of links)
def exp_one():
    print('\nSelect a type of user')
    print('1) Low number of links ( < D1 with D1 = ', numpy.percentile(list_number_successors, 10), ' )')
    print('2) Number of links around the median')
    print('3) High number of links ( > D9 with D9 = ', numpy.percentile(list_number_successors, 90), ' )')
    user_type = int(input())

    if user_type == 1:
        user, successors = data_analysis.select_user(g, numpy.min(list_number_successors),
                                                     numpy.percentile(list_number_successors, 10))
    elif user_type == 3:
        user, successors = data_analysis.select_user(g, numpy.percentile(list_number_successors, 90),
                                                     numpy.max(list_number_successors))
    else:
        user, successors = data_analysis.select_user(g, numpy.percentile(list_number_successors, 45),
                                                     numpy.percentile(list_number_successors, 55))

    print("Selected user is ", user, " with ", successors, " neighbors")

    source_node = user

    # Cut the graph
    h = sub_graph_from_node(g, source_node, distance=4)
    unfrozen_h = nx.DiGraph(h)  # For modification on the graph, we unfroze it

    # We get the number of nodes in the graph and a list of item nodes
    num_nodes, target_nodes, neighbors = graph_stats(h, source_node)

    reduced_h, target_nodes = reduce_graph_randomly(unfrozen_h, target_nodes, neighbors)

    # print_graph(reduced_h)

    # Init the parameters for page rank
    alpha = 0.15
    epsilon = 0.5 / (num_nodes * 1000)

    # Compute the page rank scores
    print('\nCompute page rank...')
    p_top_org = {}
    r_top_org = {}
    p_other_org = {}
    r_other_org = {}
    top_node = target_nodes[0]
    other_node = target_nodes[1]

    p_both_org = {}
    r_both_org = {}

    node_score = {}

    for node in target_nodes:
        p_1, r_1 = reverse_local_push(reduced_h, node, {}, {}, alpha=alpha, e=epsilon)
        if p_1.get(source_node, 0.0) > p_top_org.get(source_node, 0.0):
            p_top_org = p_1
            r_top_org = r_1
            top_node = node

        node_score[node] = p_1.get(source_node, 0.0)

        p_both_org[node] = dict(p_1)
        r_both_org[node] = dict(r_1)

    print('============ Page rank ============')
    print("Ranking for each book :")
    node_score = {k: v for k, v in sorted(node_score.items(), key=lambda item: item[1])}
    print(node_score)

    print('Top item:', top_node)

    # instantiating counterfactual explanation
    print('\nInit CFE...')
    cfe_instance = CFE(reduced_h, p_both_org, r_both_org, alpha, epsilon)

    min_actions = 1000000

    # finding the explanation using PRINCE
    cfe_instance.compute_pagerank_wo_u(source_node, target_nodes)
    cfe, replacing_item, min_number = cfe_instance.cfe_item_centric_algo_poly(source_node, top_node, target_nodes)

    print('============ PRINCE Explanation ============')
    prince_exp = [(source_node, elem) for elem in cfe]
    print(prince_exp)
    print('Replacing item: ', replacing_item)

    c = get_categories(reduced_h, top_node)
    print("top item categories: ", c)
    c = get_categories(reduced_h, replacing_item)
    print("replacing item categories: ", c)


# This experience observes the trends in recommended items and replacing items
def exp_two():
    top_categories = []
    replace_categories = []
    runtimes = []

    # source_node = "'48595465'"
    source_node = "'18999820'"

    # The experience is made 20 times
    for i in range(0, 20):
        # Cut the graph
        h = sub_graph_from_node(g, source_node, distance=4)
        unfrozen_h = nx.DiGraph(h)  # For modification on the graph, we unfroze it

        # We get the number of nodes in the graph and a list of item nodes
        num_nodes, target_nodes, neighbors = graph_stats(h, source_node, print_stats=False)

        reduced_h, target_nodes = reduce_graph_randomly(unfrozen_h, target_nodes, neighbors, k=300)

        # Init the parameters for page rank
        alpha = 0.15
        epsilon = 0.5 / (num_nodes * 1000)

        # Compute the page rank scores
        print("Ranking...")
        start = time.time()
        p_top_org = {}

        top_node = target_nodes[0]

        p_both_org = {}
        r_both_org = {}

        node_score = {}

        for node in target_nodes:
            p_1, r_1 = reverse_local_push(reduced_h, node, {}, {}, alpha=alpha, e=epsilon)
            if p_1.get(source_node, 0.0) > p_top_org.get(source_node, 0.0):
                p_top_org = p_1
                top_node = node

            node_score[node] = p_1.get(source_node, 0.0)

            p_both_org[node] = dict(p_1)
            r_both_org[node] = dict(r_1)

        # instantiating counterfactual explanation
        cfe_instance = CFE(reduced_h, p_both_org, r_both_org, alpha, epsilon)

        # finding the explanation using PRINCE
        cfe_instance.compute_pagerank_wo_u(source_node, target_nodes)
        cfe, replacing_item, min_number = cfe_instance.cfe_item_centric_algo_poly(source_node, top_node, target_nodes)

        end = time.time()

        runtimes.append(end - start)

        c = get_categories(reduced_h, top_node)
        print("top item categories: ", c)
        top_categories.append(c)  # Add the category to the list

        c = get_categories(reduced_h, replacing_item)
        print("replacing item categories: ", c)
        replace_categories.append(c) # Add the category to the list

    print(count_categories(top_categories))
    print(count_categories(replace_categories))
    print("Runtime average:", format(numpy.average(runtimes), ".2f"), " sec")


# This experience observes the impact of Epsilon parameter
def exp_three():
    top_categories = []
    replace_categories = []
    runtimes = []

    source_node = "'48595465'"
    # source_node = "'18999820'"

    for i in range(0, 20):
        # Cut the graph
        h = sub_graph_from_node(g, source_node, distance=4)
        unfrozen_h = nx.DiGraph(h)  # For modification on the graph, we unfroze it

        # We get the number of nodes in the graph and a list of item nodes
        num_nodes, target_nodes, neighbors = graph_stats(h, source_node, print_stats=False)

        reduced_h, target_nodes = reduce_graph_randomly(unfrozen_h, target_nodes, neighbors, k=100)

        # Init the parameters for page rank
        alpha = 0.15
        epsilon = 0.5 / (100 * 1000000)  # --> VALUE TO EXPERIMENT

        # Compute the page rank scores
        print("Ranking...")
        start = time.time()
        p_top_org = {}

        top_node = target_nodes[0]

        p_both_org = {}
        r_both_org = {}

        node_score = {}

        for node in target_nodes:
            p_1, r_1 = reverse_local_push(reduced_h, node, {}, {}, alpha=alpha, e=epsilon)
            if p_1.get(source_node, 0.0) > p_top_org.get(source_node, 0.0):
                p_top_org = p_1
                top_node = node

            node_score[node] = p_1.get(source_node, 0.0)

            p_both_org[node] = dict(p_1)
            r_both_org[node] = dict(r_1)

        # instantiating counterfactual explanation
        cfe_instance = CFE(reduced_h, p_both_org, r_both_org, alpha, epsilon)

        # finding the explanation using PRINCE
        cfe_instance.compute_pagerank_wo_u(source_node, target_nodes)
        cfe, replacing_item, min_number = cfe_instance.cfe_item_centric_algo_poly(source_node, top_node, target_nodes)

        end = time.time()

        runtimes.append(end - start)

        c = get_categories(reduced_h, top_node)
        print("top item categories: ", c)
        top_categories.append(c)

        c = get_categories(reduced_h, replacing_item)
        print("replacing item categories: ", c)
        replace_categories.append(c)

    print(count_categories(top_categories))
    print(count_categories(replace_categories))
    print("Epsilon", epsilon)
    print("Runtime average:", format(numpy.average(runtimes), ".4f"), " sec")


# ------ START MAIN ------
print('A) Experience 1 - Observation of explanations & recommendations depending on the type of user')
print('B) Experience 1 bis - Reduction of the graph by keeping only users with many interactions')
print('C) Experience 2 - Trends in categories (recommended items and replacing items)')
print('D) Experience 3 - Observation of the impact of Epsilon')
print('\nSelect an experience (A/B/C/D):')
selected_experience = input()

# Init the original graph
g = read_graph()

# ------ DATA ANALYSIS ------
print('============ Data analysis ============')
list_number_successors = data_analysis.user_neighbor(g)
print("Number of neighbors")
print("- Median: ", numpy.percentile(list_number_successors, 50))
print("- Average :", numpy.average(list_number_successors))
print("- Min:", numpy.min(list_number_successors))
print("- Max:", numpy.max(list_number_successors))

# d = collections.Counter(list_number_successors)
# plt.bar(d.keys(), d.values())
# plt.show()

if selected_experience == 'A':
    exp_one()
elif selected_experience == 'B':
    exp_one_bis()
elif selected_experience == 'C':
    exp_two()
elif selected_experience == 'D':
    exp_three()
