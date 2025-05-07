/*

Step 1 - (if its already installed, go to step 2)
sudo apt-get update
sudo apt-get install g++

Step 2 - 
g++ -fopenmp hpc1.cpp -o hpc1

Step 3 - 
./hpc1

sample i/o 

Enter number of vertices: 6
Enter number of edges: 7
Enter 7 edges (u v):
0 1
0 2
1 3
1 4
2 4
3 5
4 5
Enter start node: 0

BFS: 0 1 2 3 4 5
DFS: 0 1 4 5 3 2 / 0 2 4 5 1 3 

1. BFSTime Complexity:
O(V + E)
Where:

V = number of vertices
E = number of edges

Reason: Every node and every edge is visited once.

2. DFS (Depth-First Search)
Time Complexity:
O(V + E)

Same as BFS, because all nodes and edges are explored once.
Recursive stack or explicit stack used in implementation.

 3. Parallel BFS
Time Complexity (Ideal Case):
O(V + E / p) or O(log V) for certain graphs

p = number of processors/threads

It performs level-by-level traversal using frontier-based parallelism.
In practice, speedup depends on graph structure (e.g., dense/sparse, branching factor).

 4. Parallel DFS
Time Complexity (Theoretical):
O(V + E) (same as sequential in worst case)

Parallel DFS is harder to implement efficiently because it’s not inherently level-based.
In practice:
Parallel DFS explores different subtrees in separate threads.
Best-case parallelism is achieved in balanced trees or graphs with independent branches.
*/

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class Graph {
    int V;  // Number of vertices
    vector<vector<int>> adj;  // Adjacency list for the graph

public:
    // Constructor to initialize graph with given number of vertices
    Graph(int V) : V(V), adj(V) {}

    // Method to add an undirected edge between vertices u and v
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);  // As the graph is undirected
    }

    // Parallel BFS using OpenMP for level-by-level traversal
    void parallelBFS(int start) {
        vector<bool> visited(V, false);  // Vector to track visited nodes
        queue<int> q;  // Queue to hold nodes for BFS traversal

        visited[start] = true;  // Mark the start node as visited
        q.push(start);  // Push start node to queue

        cout << "BFS: ";

        // While there are nodes in the queue, continue processing
        while (!q.empty()) {
            int size = q.size();  // Get the number of nodes in the current level

            #pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                int node;

                // Pop the front node from the queue inside a critical section
                #pragma omp critical
                {
                    node = q.front();
                    q.pop();
                }

                cout << node << " ";  // Print the current node

                // Explore the neighbors of the node
                for (int neighbor : adj[node]) {
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {  // If neighbor hasn't been visited
                            visited[neighbor] = true;  // Mark as visited
                            q.push(neighbor);  // Push the neighbor to the queue
                        }
                    }
                }
            }
        }

        cout << endl;
    }

    // Helper function for task-based DFS traversal
    void dfsUtil(int node, vector<bool>& visited) {
        #pragma omp critical
        {
            if (visited[node]) return;  // If the node is already visited, return
            visited[node] = true;  // Mark the current node as visited
            cout << node << " ";  // Print the node
        }

        // Create parallel tasks to visit neighbors of the current node
        for (int neighbor : adj[node]) {
            #pragma omp task firstprivate(neighbor)
            {
                dfsUtil(neighbor, visited);  // Recursive DFS on the neighbor
            }
        }
    }

    // Task-based parallel DFS using OpenMP
    void parallelDFS(int start) {
        vector<bool> visited(V, false);  // Vector to track visited nodes
        cout << "DFS: ";

        // OpenMP parallel region to start DFS traversal
        #pragma omp parallel
        {
            #pragma omp single  // Ensure that only one thread starts the DFS
            {
                dfsUtil(start, visited);  // Start DFS from the start node
            }
        }

        cout << endl;
    }
};

int main() {
    int V, E, u, v, start;

    // Input number of vertices in the graph
    cout << "Enter number of vertices: ";
    cin >> V;

    Graph g(V);  // Create a graph with V vertices

    // Input number of edges in the graph
    cout << "Enter number of edges: ";
    cin >> E;

    // Input the edges and add them to the graph
    cout << "Enter " << E << " edges (u v):" << endl;
    for (int i = 0; i < E; ++i) {
        cin >> u >> v;
        g.addEdge(u, v);  // Add edge to the graph
    }

    // Input the start node for BFS and DFS
    cout << "Enter start node: ";
    cin >> start;

    // Call parallel BFS and DFS starting from the given node
    g.parallelBFS(start);  // Perform BFS traversal
    g.parallelDFS(start);  // Perform DFS traversal

    return 0;
}
