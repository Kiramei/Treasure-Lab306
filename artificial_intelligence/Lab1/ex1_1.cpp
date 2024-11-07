#include "opencv2/opencv.hpp"
#include <bits/stdc++.h>

using namespace std;
using namespace cv;
#define endl '\n'
typedef long long ll;

char s[100];

// Ŀ��ڵ�״̬
char goal[] = {'0' + 1,  '0' + 2,  '0' + 3,  '0' + 4,  '0' + 5,  '0' + 6,
               '0' + 7,  '0' + 8,  '0' + 9,  '0' + 10, '0' + 11, '0' + 12,
               '0' + 13, '0' + 14, '0' + 15, '0' + 0};

int maxdepth;

// ����ʽ��������
inline int h(char *cur) {
  int res = 0;
  for (int i = 0; i < 16; i++) {
    // ����ÿ��λ���ϵ����֣������������Ŀ��״̬�Ķ�Ӧλ���ϵ����֣����Ҳ�Ϊ0���հ׸񣩣������Ӳ���
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

// A*�㷨
bool A_STAR(int depth, char *a, int pre) {
  // �����ǰ״̬�Ѿ��ﵽĿ��״̬������true
  if (h(a) == 0)
    return true;
  // �����ǰ��ȼ�������ʽ������ֵ��1�������������ƣ���������
  if (depth + h(a) - 1 > maxdepth)
    return false;
  int sx, sy;
  // �ҵ���ǰ״̬�пհ׸��λ��
  for (int i = 0; i < 16; i++)
    if (a[i] == '0')
      sx = i / 4 + 1, sy = i % 4 + 1;
  // �����ĸ������ϵ��ƶ�
  for (int i = 0; i < 4; i++) {

    int xx = dx[i] + sx, yy = dy[i] + sy;
    // ����ƶ������˾Ź���ı߽磬�����ƶ�����һ�ε��ƶ��෴��������
    if (xx < 1 || xx > 4 || yy < 1 || yy > 4 || (pre + i == 4))
      continue;
    // �����հ׸������λ���ϵ�����
    swap(a[(xx - 1) * 4 + yy - 1], a[(sx - 1) * 4 + sy - 1]);
    // �����ƶ��ķ���ȷ���ƶ��������ַ�
    char moveOp = 'R';
    if (i == 1)
      moveOp = 'D';
    else if (i == 2)
      moveOp = 'U';
    else if (i == 3)
      moveOp = 'L';
    // ���ƶ������ַ���ӵ�·���ַ�����
    ans += moveOp;
    // �ݹ�����
    if (A_STAR(depth + 1, a, i))
      return true;
    // �ָ�״̬
    swap(a[(xx - 1) * 4 + yy - 1], a[(sx - 1) * 4 + sy - 1]);
    // �Ƴ�·���ַ����е����һ���ַ�
    ans.pop_back();
  }
  return false;
}

// ��ʾƴͼ״̬
void displayPuzzle(const vector<int> &order, const vector<Mat> &pieces) {
  auto orderMap = map<int, int>();
  for (int i = 0; i < 16; i++) {
    orderMap[goal[i] - '0'] = i;
  }
  Mat img(720, 720, CV_8UC3, Scalar(255, 255, 255));
  for (int i = 0; i < 16; i++) {
    int x = (i % 4) * 180;
    int y = (i / 4) * 180;
    // ��pieces�ж�Ӧ��С���滻img�еĶ�Ӧλ��
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
  // ��ȡͼƬ
  Mat img = imread("puzzle.jpg");
  resize(img, img, Size(720, 720));
  imshow("Origin", img);
  // waitKey(0);
  // ��ͼƬ�ָ��9��С��
  vector<Mat> pieces;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      Mat piece = img(Rect(j * 180, i * 180, 180, 180));
      pieces.push_back(piece);
    }
  }
  // ��С�����
  vector<int> order = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0};

  // ����ƴͼ�����ɳ�ʼ״̬
  // random_device rd;
  // mt19937 generator(rd());
  // shuffle(order.begin(), order.end(), generator);

  displayPuzzle(order, pieces);

  //   �������
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
    cout << "��ԭ������" << move << endl;
  }

  //   for (int i = 0; i < 16; i++)
  //     s[i] = order[i] + '0';
  // �����ʼ״̬����Ŀ��״̬���������0������
  //   if (h(s) == 0) {
  //     puts("0");
  //     return 0;
  //   }

  // ���������������ƣ�ֱ���ҵ���
  //   for (maxdepth = 1;; maxdepth++) {
  //     vector<int> temp(order);
  //     if (A_STAR(0, s, -1)) {
  //       // �����С������·���ַ���
  //       cout << "��Ҫ���ٲ���Ϊ��" << maxdepth << endl;
  //       cout << "�յ������ƶ��ķ�����" << ans << endl;
  //       break;
  //     }
  //   }

  //   auto rec = recoverStatus(order);
  //   for (auto status : rec) {
  //     cout << "��ǰ״̬��";
  //     for (auto i : status)
  //       cout << i << " ";
  //     cout << endl;
  //     displayPuzzle(status, pieces);
  //   }
  return 0;
}
