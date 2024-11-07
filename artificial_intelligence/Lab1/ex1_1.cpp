#include "opencv2/opencv.hpp"
#include <bits/stdc++.h>

using namespace std;
using namespace cv;
#define endl '\n'
typedef long long ll;

char s[100];

// 目标节点状态
char goal[] = {'0' + 1,  '0' + 2,  '0' + 3,  '0' + 4,  '0' + 5,  '0' + 6,
               '0' + 7,  '0' + 8,  '0' + 9,  '0' + 10, '0' + 11, '0' + 12,
               '0' + 13, '0' + 14, '0' + 15, '0' + 0};

int maxdepth;

// 启发式函数构造
inline int h(char *cur) {
  int res = 0;
  for (int i = 0; i < 16; i++) {
    // 对于每个位置上的数字，如果它不等于目标状态的对应位置上的数字，并且不为0（空白格），则增加步数
    if (goal[i] != cur[i] && cur[i] != 0)
      res++;
  }
  for (int i = 0; i < 16; i++) {
    if (cur[i] == '0')
      continue;
    int x = (cur[i] - '1') / 4;
    int y = (cur[i] - '1') % 4;
    int xx = i / 4;
    int yy = i % 4;
    res += abs(x - xx) + abs(y - yy);
  }
  return res;
}

int dx[] = {0, 1, -1, 0};
int dy[] = {1, 0, 0, -1};

string ans;

// A*算法
bool A_STAR(int depth, char *a, int pre) {
  // 如果当前状态已经达到目标状态，返回true
  if (h(a) == 0)
    return true;
  // 如果当前深度加上启发式函数的值减1大于最大深度限制，结束搜索
  if (depth + h(a) - 1 > maxdepth)
    return false;
  int sx, sy;
  // 找到当前状态中空白格的位置
  for (int i = 0; i < 16; i++)
    if (a[i] == '0')
      sx = i / 4 + 1, sy = i % 4 + 1;
  // 尝试四个方向上的移动
  for (int i = 0; i < 4; i++) {

    int xx = dx[i] + sx, yy = dy[i] + sy;
    // 如果移动超出了九宫格的边界，或者移动与上一次的移动相反，则跳过
    if (xx < 1 || xx > 4 || yy < 1 || yy > 4 || (pre + i == 4))
      continue;
    // 交换空白格和相邻位置上的数字
    swap(a[(xx - 1) * 4 + yy - 1], a[(sx - 1) * 4 + sy - 1]);
    // 根据移动的方向确定移动操作的字符
    char moveOp = 'R';
    if (i == 1)
      moveOp = 'D';
    else if (i == 2)
      moveOp = 'U';
    else if (i == 3)
      moveOp = 'L';
    // 将移动操作字符添加到路径字符串中
    ans += moveOp;
    // 递归搜索
    if (A_STAR(depth + 1, a, i))
      return true;
    // 恢复状态
    swap(a[(xx - 1) * 4 + yy - 1], a[(sx - 1) * 4 + sy - 1]);
    // 移除路径字符串中的最后一个字符
    ans.pop_back();
  }
  return false;
}

// 显示拼图状态
void displayPuzzle(const vector<int> &order, const vector<Mat> &pieces) {
  auto orderMap = map<int, int>();
  for (int i = 0; i < 16; i++) {
    orderMap[goal[i] - '0'] = i;
  }
  Mat img(720, 720, CV_8UC3, Scalar(255, 255, 255));
  for (int i = 0; i < 16; i++) {
    int x = (i % 4) * 180;
    int y = (i / 4) * 180;
    // 用pieces中对应的小块替换img中的对应位置
    pieces[orderMap[order[i]]].copyTo(img(Rect(x, y, 180, 180)));
  }
  imshow("Puzzle", img);
  waitKey(0);
}

vector<vector<int>> recoverStatus(const vector<int> &firstStatus) {
  vector<vector<int>> res;
  res.push_back(firstStatus);
  for (auto step : ans) {
    auto temp = res.back();
    int zeroPos = find(temp.begin(), temp.end(), 0) - temp.begin();
    int x = zeroPos / 4, y = zeroPos % 4;
    if (step == 'U') {
      swap(temp[zeroPos], temp[zeroPos - 4]);
    } else if (step == 'D') {
      swap(temp[zeroPos], temp[zeroPos + 4]);
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
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      Mat piece = img(Rect(j * 180, i * 180, 180, 180));
      pieces.push_back(piece);
    }
  }
  // 将小块打乱
  vector<int> order = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0};

  // 打乱拼图，生成初始状态
  // random_device rd;
  // mt19937 generator(rd());
  // shuffle(order.begin(), order.end(), generator);

  displayPuzzle(order, pieces);

  //   正向打乱
  string move = "";
  int p = -1;
  while (true) {
    int randNum = rand() % 4;
    if (p == randNum)
      continue;
    p = randNum;
    int zeroPos = find(order.begin(), order.end(), 0) - order.begin();
    int x = zeroPos / 4, y = zeroPos % 4;
    if (randNum == 0 && x > 0) {
      swap(order[zeroPos], order[zeroPos - 4]);
      move = "D" + move;
    } else if (randNum == 1 && x < 3) {
      swap(order[zeroPos], order[zeroPos + 4]);
      move = "U" + move;
    } else if (randNum == 2 && y > 0) {
      swap(order[zeroPos], order[zeroPos - 1]);
      move = "R" + move;
    } else if (randNum == 3 && y < 3) {
      swap(order[zeroPos], order[zeroPos + 1]);
      move = "L" + move;
    }
    displayPuzzle(order, pieces);
    cout << "还原方法：" << move << endl;
  }

  //   for (int i = 0; i < 16; i++)
  //     s[i] = order[i] + '0';
  // 如果初始状态就是目标状态，输出步数0并返回
  //   if (h(s) == 0) {
  //     puts("0");
  //     return 0;
  //   }

  // 逐渐增加最大深度限制，直到找到解
  //   for (maxdepth = 1;; maxdepth++) {
  //     vector<int> temp(order);
  //     if (A_STAR(0, s, -1)) {
  //       // 输出最小步数和路径字符串
  //       cout << "需要最少步数为：" << maxdepth << endl;
  //       cout << "空档所需移动的方法：" << ans << endl;
  //       break;
  //     }
  //   }

  //   auto rec = recoverStatus(order);
  //   for (auto status : rec) {
  //     cout << "当前状态：";
  //     for (auto i : status)
  //       cout << i << " ";
  //     cout << endl;
  //     displayPuzzle(status, pieces);
  //   }
  return 0;
}
