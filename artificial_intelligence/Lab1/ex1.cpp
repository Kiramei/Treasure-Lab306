#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
#define endl '\n'
typedef long long ll;

char s[15];

// Ŀ��ڵ�״̬
char goal[] = {'1', '2', '3', '8', '0', '4', '7', '6', '5'};

int maxdepth;

// ����ʽ��������
inline int h(char *cur) {
    int res = 0;
    for (int i = 0; i < 9; i ++ ) {
        // ����ÿ��λ���ϵ����֣������������Ŀ��״̬�Ķ�Ӧλ���ϵ����֣����Ҳ�Ϊ0���հ׸񣩣������Ӳ���
        if (goal[i] != cur[i] && cur[i] != 0) res++;
    }
    return res;
}

int dx[] = {0, 1, -1, 0};
int dy[] = {1, 0, 0, -1};

string ans;

// A*�㷨
bool A_STAR (int depth, char *a, int pre) {
    // �����ǰ״̬�Ѿ��ﵽĿ��״̬������true
    if (h(a) == 0) return true;
    // �����ǰ��ȼ�������ʽ������ֵ��1�������������ƣ���������
    if (depth + h(a) - 1 > maxdepth) return false;
    int sx, sy;
    // �ҵ���ǰ״̬�пհ׸��λ��
    for (int i = 0; i < 9; i ++ )
        if (a[i] == '0') sx = i / 3 + 1, sy = i % 3 + 1;
    // �����ĸ������ϵ��ƶ�
    for (int i = 0; i < 4; i ++ ) {
        int xx = dx[i] + sx, yy = dy[i] + sy;
        // ����ƶ������˾Ź���ı߽磬�����ƶ�����һ�ε��ƶ��෴��������
        if (xx < 1 || xx > 3 || yy < 1 || yy > 3 || (pre + i == 3)) continue;
        // �����հ׸������λ���ϵ�����
        swap(a[(xx - 1) * 3 + yy - 1], a[(sx - 1) * 3 + sy - 1]);
        // �����ƶ��ķ���ȷ���ƶ��������ַ�
        char moveOp = 'R';
        if (i == 1) moveOp = 'D';
        else if (i == 2) moveOp = 'U';
        else if (i == 3) moveOp = 'L';
        // ���ƶ������ַ���ӵ�·���ַ�����
        ans += moveOp;
        // �ݹ�����
        if (A_STAR(depth + 1, a, i)) return true;
        // �ָ�״̬
        swap(a[(xx - 1) * 3 + yy - 1], a[(sx - 1) * 3 + sy - 1]);
        // �Ƴ�·���ַ����е����һ���ַ�
        ans.pop_back();
    }
    return false;
}

// ��ʾƴͼ״̬
void displayPuzzle(const vector<int>& order, const vector<Mat>& pieces) {
    auto orderMap = map<int, int>();
    for (int i = 0; i < 9; i++) {
        orderMap[goal[i]-'0'] = i;
    }
    Mat img(720, 720, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < 9; i++) {
        int x = (i % 3) * 240;
        int y = (i / 3) * 240;
        // ��pieces�ж�Ӧ��С���滻img�еĶ�Ӧλ��
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
    // ��ȡͼƬ
    Mat img = imread("puzzle.jpg");
    resize(img, img, Size(720, 720));
    imshow("Origin", img);
    // waitKey(0);
    // ��ͼƬ�ָ��9��С��
    vector<Mat> pieces;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Mat piece = img(Rect(j * 240, i * 240, 240, 240));
            pieces.push_back(piece);
        }
    }
    // ��С�����
    vector<int> order;
    for (int i = 0; i < 9; i++) {
        order.push_back(i);
}
    // ����ƴͼ�����ɳ�ʼ״̬
    random_device rd;
    mt19937 generator(rd());
    shuffle(order.begin(), order.end(), generator);

    displayPuzzle(order, pieces);
   
    for (int i = 0; i < 9; i ++ ) s[i] = order[i] + '0';
    // �����ʼ״̬����Ŀ��״̬���������0������
    if (h(s) == 0) {
        puts("0");
        return 0;
    }
    // ���������������ƣ�ֱ���ҵ���
    for (maxdepth = 1; ; maxdepth ++ ) {
        if (A_STAR(0, s, -1)) {
            // �����С������·���ַ���
            cout <<"��Ҫ���ٲ���Ϊ��"<< maxdepth <<endl;
            cout <<"�յ������ƶ��ķ�����"<< ans << endl;
            break;
        }
    }

    auto rec = recoverStatus(order);
    for (auto status : rec) {
        cout << "��ǰ״̬��" ;
        for (auto i : status) 
            cout << i << " ";
        cout << endl;
        displayPuzzle(status, pieces);
    }

    return 0;
}
