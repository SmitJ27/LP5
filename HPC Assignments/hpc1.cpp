/*

Step 1 - (if its already installed, go to step 2)
sudo apt-get update
sudo apt-get install g++

Step 2 - 
g++ -fopenmp hpc1.cpp -o hpc1

Step 3 - 
./hpc1 - for linux
OR 
hpc1 

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
#include <omp.h>

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) : V(V), adj(V) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "BFS: ";

        while (!q.empty()) {
            int size = q.size();
            vector<int> currentLevel;

            // Extract current level
            #pragma omp critical
            {
                for (int i = 0; i < size && !q.empty(); ++i) {
                    currentLevel.push_back(q.front());
                    q.pop();
                }
            }

            #pragma omp parallel for
            for (int i = 0; i < currentLevel.size(); ++i) {
                int node = currentLevel[i];

                #pragma omp critical
                cout << node << " ";

                for (int neighbor : adj[node]) {
                    bool doVisit = false;

                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            doVisit = true;
                        }
                    }

                    if (doVisit) {
                        #pragma omp critical
                        q.push(neighbor);
                    }
                }
            }
        }

        cout << endl;
    }

void dfsUtil(int node, vector<bool>& visited) {
    visited[node] = true;
    cout << node << " ";

    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            dfsUtil(neighbor, visited);
        }
    }
}

void parallelDFS(int start) {
    vector<bool> visited(V, false);
    cout << "DFS: ";
    dfsUtil(start, visited);
    cout << endl;
}
    };

int main() {
    int V, E, u, v, start;

    cout << "Enter number of vertices: ";
    cin >> V;

    Graph g(V);

    cout << "Enter number of edges: ";
    cin >> E;

    cout << "Enter " << E << " edges (u v):" << endl;
    for (int i = 0; i < E; ++i) {
        cin >> u >> v;
        g.addEdge(u, v);
    }

    cout << "Enter start node: ";
    cin >> start;

    g.parallelBFS(start);
    g.parallelDFS(start);

    return 0;
}
