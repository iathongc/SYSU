#define _CRT_SECURE_NO_WARNINGS   
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>

typedef struct Polynomial {											//多項式 
	float coef; 													//係數
	int expn;   													//指數
	struct Polynomial *next;										//指針
} Polynomial, *Polyn;

//創建一個頭指針為head，項數為m的一元多項式
Polyn CreatPolyn (Polyn head, int m) {
	head = (Polyn)malloc(sizeof(struct Polynomial));
	head -> next = NULL;
	for (int i=1; i<=m; i++) {
		Polyn p = (Polyn)malloc(sizeof(struct Polynomial));
		printf ("請輸入第%d項的係數與指數：", i);
		scanf ("%f%d", &p->coef, &p->expn);
		if (p->coef == 0)
			free (p);
		else {
			Polyn q1, q2;
			q1 = head;
			q2 = head -> next;
			while (q2 != NULL && p->expn < q2->expn) {
				q1 = q2;
				q2 = q2->next;
			}
			if (q2 != NULL && p->expn == q2->expn) {
				q2->coef += p->coef;
				if (q2->coef == 0) {
					q1->next = q2->next;
					free (q2);
				}
				free (p);
			}
			else {
				p -> next = q2;
				q1 -> next = p;
			}
		}
	}
	return head;
}

void printPoLlyn (Polyn head) {											//打印
	Polyn q = head -> next;
	int f = 0;     														//記錄是否為第一項
	if (!q) {
		puts ("0");
		puts ("\n");
		return;
	}
	while (q) {
		if (q->coef > 0 && f == 1) {
			printf("+");
		}
		f = 1;
		if (q->coef != 1 && q->coef != -1) {
				printf ("%g", q->coef);
			if (q->expn == 1)
				printf ("x");
			else if (q->expn != 0)
				printf ("x^%d", q->expn);
		}
		else {
			if (q->coef == 1) {
				if (q->expn == 0)
					printf ("1");
				else if (q->expn == 1)
					printf ("x");
				else
					printf ("x^%d", q->expn);
			}
			if (q->coef == -1) {
				if (q->expn == 0)
					printf("-1");
				else if (q->expn == 1)
					printf("-x");
				else
					printf("-x^%d", q->expn);
			}
		}
		q = q->next;
	}
	printf ("\n");
}

int compare (Polyn a, Polyn b) {										//比較兩個多項式的大小
	if (a&&b) { 														//多項式a和b均不為空
		if (a->expn > b->expn)
			return 1;													//a的指數大於b的指數
		else if (a->expn < b->expn)
			return -1;
		else
			return 0;
	}
	else if (!a&&b)
		return -1; 														//a為空，b不為空
	else if (a&&!b)
		return 1;  														//b為空，a不為空 
	else if (!a&&!b)
		return 0;  														//a,b均為空
}

Polyn addPolyn (Polyn a, Polyn b) {  									//求解a+b，並返回頭結點head
	Polyn head, qc;
	Polyn qa = a->next;
	Polyn qb = b->next;
	Polyn hc = (Polyn)malloc(sizeof(Polynomial));
	hc -> next = NULL;
	head = hc;
	while (qa || qb) {
		qc = (Polyn)malloc(sizeof(Polynomial));
		if (compare(qa, qb) == 1) {
			qc->coef = qa->coef;
			qc->expn = qa->expn;
			qa = qa->next;
		}
		else if (compare(qa, qb) == 0) {								//指數相同，直接相加
			qc->coef = qa->coef + qb->coef;
			qc->expn = qa->expn ; 
			qa = qa->next;
			qb = qb->next;
		}
		else {
			qc->coef = qb->coef;
			qc->expn = qb->expn;
			qb = qb->next;
		}
		if (qc->coef != 0) {											//將該節點插入鏈表中
			qc->next = hc->next;
			hc->next = qc;
			hc = qc;
		}
		else free(qc);		
	}
	return head;
}

Polyn subPolyn (Polyn a, Polyn b) {										//求解a-b
	Polyn h = b;
	Polyn p = b->next;
	while (p) {
		p->coef *= -1;
		p = p->next;
	}
	Polyn head = addPolyn (a, h);
	for (Polyn i = h->next; i != 0; i = i->next) {
		i->coef *= -1;
	}		
	return head;
}

double value (Polyn head, int x) { 										//計算x的值
	double sum = 0;
	for (Polyn p = head->next; p != 0; p = p->next) {
		int tmp = 1;
		int expn = p->expn;
		while (expn != 0) {												//指數不為0
			if (expn < 0)
			tmp /= x, expn++;
			else if (expn>0)
			tmp *= x, expn--;
		}
		sum += p->coef*tmp;
	}	
	return sum;
}

Polyn derivative (Polyn head) {											//計算導數
    Polyn derivativeHead, p, q;
    derivativeHead = (Polyn)malloc(sizeof(Polynomial));
    derivativeHead->next = NULL;
    p = head->next;
    while (p!=NULL && q->expn!=0) {
        q = (Polyn)malloc(sizeof(Polynomial));
        q->coef = p->coef * p->expn;
        q->expn = p->expn - 1;
        q->next = NULL;
        Polyn prev = derivativeHead;									//尋找插入點，按升冪順序
        Polyn current = derivativeHead->next;
        while (current != NULL && current->expn > q->expn) {
            prev = current;
            current = current->next;
        }
        if (current != NULL && current->expn == q->expn) {				//插入節點
            current->coef += q->coef;
            free(q);
        }
		else {
            q->next = current;
            prev->next = q;
        }
        p = p->next;
    }
    return derivativeHead;
}

Polyn multiplyPolyn (Polyn a, Polyn b) {								//計算a*b
    Polyn resultHead, pa, pb, pr, temp, node;
    resultHead = (Polyn)malloc(sizeof(Polynomial));
    resultHead->next = NULL;
    pa = a->next;
    while (pa) {
        pb = b->next;
        pr = resultHead;
        while (pb) {
            temp = (Polyn)malloc(sizeof(Polynomial));
            temp->coef = pa->coef * pb->coef;
            temp->expn = pa->expn + pb->expn;
            temp->next = NULL;
            node = resultHead;
            while (node->next != NULL && node->next->expn > temp->expn) {
                node = node->next;
            }
            if (node->next != NULL && node->next->expn == temp->expn) {
                node->next->coef += temp->coef;
                free (temp);
            }
			else {
                temp->next = node->next;
                node->next = temp;
            }
            pb = pb->next;
        }
        pa = pa->next;
    }
    return resultHead;
}

int main () {
	int m;
	Polyn a=0, b=0;
	printf ("請輸入a的項數：");
	scanf ("%d", &m);
	a = CreatPolyn (a, m);
	printPoLlyn (a);
	printf ("\n");
	printf ("請輸入b的項數：");
	scanf ("%d", &m);
	b = CreatPolyn (b, m);
	printPoLlyn(b);
	printf ("\n");
	printf ("輸出 a+b：");
	printPoLlyn (addPolyn(a, b));
	printf ("輸出 a-b：");
	printPoLlyn (subPolyn(a, b));
	Polyn c = multiplyPolyn(a, b);
	printf ("輸出 a*b：");
	printPoLlyn(c);
	printf ("輸出 a'：");
	Polyn a_derivative = derivative(a);
	printPoLlyn(a_derivative);
	printf ("\n");
	int x;
	printf ("請輸入x的值：");
	scanf ("%d", &x);
	printf ("輸出a的多項式的值為：%.2lf", value(a, x));
	return 0;
}
