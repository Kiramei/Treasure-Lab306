#include <bits/stdc++.h>


#define INRANGE(i, a, b) (i >= a && i < b)

using namespace std;

vector<int> getMove(const vector<int>& disc, int n, int empty_slot)
{
    vector<int> movements;
    if (empty_slot / n > 0)
        movements.push_back(empty_slot - n);
    if (empty_slot % n > 0)
        movements.push_back(empty_slot - 1);
    if (empty_slot / n < n - 1)
        movements.push_back(empty_slot + n);
    if (empty_slot % n < n - 1)
        movements.push_back(empty_slot + 1);

    return movements;
}

void makeMove(vector<int>& disc, int n, int& empty_slot, const int move)
{
    int temp = empty_slot;
    empty_slot = move;
    swap(disc[temp], disc[empty_slot]);
}

bool solved(const vector<int>& disc)
{
    for (int i = 0; i < disc.size(); i++)
    {
        if (disc[i] != i)
            return false;
    }
    return true;
}

vector<vector<int>> branches(const vector<int>& disc, int n, int empty_slot)
{
    vector<int> moves = getMove(disc, n, empty_slot);
    vector<vector<int>> valid_branches;
    for (const int move : moves)
    {
        valid_branches.push_back(disc);
        makeMove(valid_branches.back(), n, empty_slot, move);
    }
    return valid_branches;
}

vector<string> solve(const vector<int>& initial_disc, int n)
{
    vector<int> disc = initial_disc;
    int empty_slot = find(disc.begin(), disc.end(), 0) - disc.begin();
    vector<string> record_move;

    auto cmp = [](const pair<vector<int>, int>& a, const pair<vector<int>, int>& b) {
        return a.second >= b.second;
    };
    priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int>>, decltype(cmp)> pq(cmp);
    set<vector<int>> explored;
    pq.push(make_pair(disc, 0));
    explored.insert(disc);

    while (!pq.empty())
    {
        pair<vector<int>, int> node = pq.top();
        pq.pop();
        disc = node.first;
        int cost = node.second;

        if (solved(disc))
        {
            return record_move;
        }

        vector<vector<int>> valid_branches = branches(disc, n, empty_slot);
        for (const vector<int>& branch : valid_branches)
        {
            if (explored.count(branch) == 0)
            {
                explored.insert(branch);
                pq.push(make_pair(branch, cost + 1));
            }
        }
    }

    // No solution found
    return vector<string>();
}

int main()
{
    int n = 4;
    // vector<int> initial_disc(n * n);

    // for (int i = 0; i < n * n; i++)
    // {
    //     cin >> initial_disc[i];
    // }

    vector<int> initial = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0};
    while (true)
    {
    shuffle(initial.begin(), initial.end(), default_random_engine(0));
    vector<string> solution = solve(initial, n);

    if (!solution.empty())
    {
        cout << solution.size() << " moves are required to solve the puzzle." << endl;
        for (const string& move : solution)
        {
            cout << move << endl;
        }
    }
    else
    {
        cout << "No solution found." << endl;
    }
    }

    return 0;
}