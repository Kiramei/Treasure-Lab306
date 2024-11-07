// ����ͷ�ļ�
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
typedef long long ll;

// �ӿ�����ٶȣ���������ʱ���ٺ�ʱ
inline ll read() {
	ll f = 1, x = 0; char ch;
	do {ch = getchar(); if (ch == '-')f = -1;} while (ch > '9' || ch < '0');
	do {x = x * 10 + ch - '0'; ch = getchar();} while (ch >= '0' && ch <= '9');
	return f * x;
}

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


int main() {
    scanf("%s", s);
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
			cout <<"���֡�0���ƶ��Ĺ켣��"<< ans << endl;
			return 0;
		}
	}
	return 0;
}