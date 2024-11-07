#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
#define endl '\n'
typedef long long ll;

char s[15];

// 目标节点状态
char goal[] = {'1', '2', '3', '8', '0', '4', '7', '6', '5'};

int maxdepth;

// 启发式函数构造
inline int h(char *cur) {
    int res = 0;
    for (int i = 0; i < 9; i ++ ) {
        // 对于每个位置上的数字，如果它不等于目标状态的对应位置上的数字，并且不为0（空白格），则增加步数
        if (goal[i] != cur[i] && cur[i] != 0) res++;
    }
    return res;
}

int dx[] = {0, 1, -1, 0};
int dy[] = {1, 0, 0, -1};

string ans;

// A*算法
bool A_STAR (int depth, char *a, int pre) {
    // 如果当前状态已经达到目标状态，返回true
    if (h(a) == 0) return true;
    // 如果当前深度加上启发式函数的值减1大于最大深度限制，结束搜索
    if (depth + h(a) - 1 > maxdepth) return false;
    int sx, sy;
    // 找到当前状态中空白格的位置
    for (int i = 0; i < 9; i ++ )
        if (a[i] == '0') sx = i / 3 + 1, sy = i % 3 + 1;
    // 尝试四个方向上的移动
    for (int i = 0; i < 4; i ++ ) {
        int xx = dx[i] + sx, yy = dy[i] + sy;
        // 如果移动超出了九宫格的边界，或者移动与上一次的移动相反，则跳过
        if (xx < 1 || xx > 3 || yy < 1 || yy > 3 || (pre + i == 3)) continue;
        // 交换空白格和相邻位置上的数字
        swap(a[(xx - 1) * 3 + yy - 1], a[(sx - 1) * 3 + sy - 1]);
        // 根据移动的方向确定移动操作的字符
        char moveOp = 'R';
        if (i == 1) moveOp = 'D';
        else if (i == 2) moveOp = 'U';
        else if (i == 3) moveOp = 'L';
        // 将移动操作字符添加到路径字符串中
        ans += moveOp;
        // 递归搜索
        if (A_STAR(depth + 1, a, i)) return true;
        // 恢复状态
        swap(a[(xx - 1) * 3 + yy - 1], a[(sx - 1) * 3 + sy - 1]);
        // 移除路径字符串中的最后一个字符
        ans.pop_back();
    }
    return false;
}

// 显示拼图状态
void displayPuzzle(const vector<int>& order, const vector<Mat>& pieces) {
    auto orderMap = map<int, int>();
    for (int i = 0; i < 9; i++) {
        orderMap[goal[i]-'0'] = i;
    }
    Mat img(720, 720, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < 9; i++) {
        int x = (i % 3) * 240;
        int y = (i / 3) * 240;
        // 用pieces中对应的小块替换img中的对应位置
        pieces[orderMap[order[i]]].copyTo(img(Rect(x, y, 240, 240)));
    }
    imshow("Puzzle", img);
    waitKey(0);
}

vector<vector<int>> recoverStatus(const vector<int>& firstStatus) {
    vector<vector<int>> res;
    res.push_back(firstStatus);
    for (auto step : ans) {
        auto temp = res.back();
        int zeroPos = find(temp.begin(), temp.end(), 0) - temp.begin();
        int x = zeroPos / 3, y = zeroPos % 3;
        if (step == 'U') {
            swap(temp[zeroPos], temp[zeroPos - 3]);
        } else if (step == 'D') {
            swap(temp[zeroPos], temp[zeroPos + 3]);
        } else if (step == 'L') {
            swap(temp[zeroPos], temp[zeroPos - 1]);
        } else if (step == 'R') {
            swap(temp[zeroPos], temp[zeroPos + 1]);
        }
        res.push_back(temp);
    }
    return res;
}

int main() {
    // 读取图片
    Mat img = imread("puzzle.jpg");
    resize(img, img, Size(720, 720));
    imshow("Origin", img);
    // waitKey(0);
    // 将图片分割成9个小块
    vector<Mat> pieces;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Mat piece = img(Rect(j * 240, i * 240, 240, 240));
            pieces.push_back(piece);
        }
    }
    // 将小块打乱
    vector<int> order;
    for (int i = 0; i < 9; i++) {
        order.push_back(i);
}
    // 打乱拼图，生成初始状态
    random_device rd;
    mt19937 generator(rd());
    shuffle(order.begin(), order.end(), generator);

    displayPuzzle(order, pieces);
   
    for (int i = 0; i < 9; i ++ ) s[i] = order[i] + '0';
    // 如果初始状态就是目标状态，输出步数0并返回
    if (h(s) == 0) {
        puts("0");
        return 0;
    }
    // 逐渐增加最大深度限制，直到找到解
    for (maxdepth = 1; ; maxdepth ++ ) {
        if (A_STAR(0, s, -1)) {
            // 输出最小步数和路径字符串
            cout <<"需要最少步数为："<< maxdepth <<endl;
            cout <<"空档所需移动的方法："<< ans << endl;
            break;
        }
    }

    auto rec = recoverStatus(order);
    for (auto status : rec) {
        cout << "当前状态：" ;
        for (auto i : status) 
            cout << i << " ";
        cout << endl;
        displayPuzzle(status, pieces);
    }

    return 0;
}
