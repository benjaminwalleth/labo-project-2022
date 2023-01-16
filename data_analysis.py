import random


def user_neighbor(graph):
    users = [x for x, y in graph.nodes(data=True) if y['type'] == "'user'"]

    list_number_successors = []
    for user in users:
        successors = graph.successors(user)
        list_number_successors.append(len(list(successors)))

    return list_number_successors


def select_user(graph, min_limit, max_limit):
    users = [x for x, y in graph.nodes(data=True) if y['type'] == "'user'"]

    selected_user = -1
    successors = -1

    for user in users:
        successors = graph.successors(user)
        number_successors = len(list(successors))
        if max_limit > number_successors > min_limit:
            is_selected = random.randrange(0, 10)
            if selected_user == -1:
                selected_user = user
                successors = number_successors
            else:
                if is_selected == 0:
                    selected_user = user
                    successors = number_successors
                    return selected_user, successors

    return selected_user, successors


