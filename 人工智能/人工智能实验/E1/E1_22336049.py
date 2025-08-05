import heapq

def dijkstra(graph, start):
    priority_queue = [(0, start)]
    distance = {vertex: float('infinity') for vertex in graph}
    distance[start] = 0
    predecessor = {vertex: None for vertex in graph}

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distance[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance_to_neighbor = current_distance + weight

            if distance_to_neighbor < distance[neighbor]:
                distance[neighbor] = distance_to_neighbor
                predecessor[neighbor] = current_vertex
                heapq.heappush(priority_queue, (distance_to_neighbor, neighbor))

    return distance, predecessor

def get_shortest_path(predecessor, start, target):
    path = []
    current_vertex = target

    while current_vertex is not None:
        path.insert(0, current_vertex)
        current_vertex = predecessor[current_vertex]

    return path

def main():
    m, n = map(int, input().split())

    graph = {}
    for _ in range(n):
        node1, node2, weight = input().split()
        weight = int(weight)
        
        if node1 not in graph:
            graph[node1] = {}
        if node2 not in graph:
            graph[node2] = {}

        graph[node1][node2] = weight
        graph[node2][node1] = weight

    distance, predecessor = dijkstra(graph, 'a')

    while True:
        try:
            start_node, target_node = input().split()

            shortest_distance = distance[target_node]
            shortest_path = get_shortest_path(predecessor, 'a', target_node)

            print(f"Shortest distance from {start_node} to {target_node}: {shortest_distance}")
            print(f"Shortest path: {' -> '.join(shortest_path)}")
        
        except EOFError:
            break

if __name__ == "__main__":
    main()
