dic={0:'a',1:'b',2:'c',3:'d',4:'e'}
def remove_space(s1):
    l = len(s1)
    i = 2
    while i<l:
        if s1[i] == " " and s1[i - 2] != ")":
            list1 = list(s1)
            list1.pop(i)
            s1 = ''.join(list1)
            l -= 1
        i += 1
    return s1

def remove_comma(l):
    for i in range(0,len(l)):
        if l[i][-1]==',':
            a_1=l[i][0:-1]
            l[i]=a_1
    return l

def get_opposite(s2):
    ans_1 = ""
    if s2[0] != "¬":
        ans_1 = "¬" + s2
    else:
        ans_1 = s2[1:len(s2)]
    return ans_1

def weici(s3):
    ans_2 = s3[0:s3.find("(")]
    return ans_2

def get_one_case(example_1):
    return example_1[example_1.find("(") + 1:example_1.find(")")]

def get_first_part(example_2): 
    return example_2[example_2.find("(") + 1:example_2.find(",")]

def get_last_part(example_3):
    return example_3[example_3.find(",") + 1:example_3.find(")")]

def is_variable(f):
    if f in ['x', 'y', 'z', 'u', 'v', 'w']:
        return True
    else:
        return False

def count_variables(statement):
    sum = 0
    for i in range(0, len(statement)):
        if is_variable(statement[i]) == True:
            sum += 1
    return sum

def judge_case(statement1, statement2):
    ans = "none"
    if is_variable(statement1[0]):
        if statement1[1] != statement2[1]:
            return ""
        else:
            ans = "(" + statement1[0] + "=" + statement2[0] + ")"
            return ans
    elif is_variable(statement2[0]):
        if statement1[1] != statement2[1]:
            return ""
        else:
            ans = ans = "(" + statement2[0] + "=" + statement1[0] + ")"
            return ans
    elif is_variable(statement1[1]):
        if statement1[0] != statement2[0]:
            return ""
        else:
            ans = ans = "(" + statement1[1] + "=" + statement2[1] + ")"
            return ans
    elif is_variable(statement2[1]):
        if statement1[0] != statement2[0]:
            return ""
        else:
            ans = ans = "(" + statement2[1] + "=" + statement1[1] + ")"
            return ans
    elif statement1[0] == statement2[0] and statement1[1] == statement2[1]:
        return ans
    else:
        return ""

def judge(list1,list2):
    list1_index=-1
    list2_index=-1
    flag1=0
    for i in range(0,len(list1[0])):
        head_1=weici(list1[0][i])
        for j in range(0,len(list2[0])):
            head_2=weici(list2[0][j])
            if head_1==get_opposite(head_2):
                list1_index=i
                list2_index=j
                flag1=1
                break
        if flag1==1:
            break
    if list1_index==-1 and list2_index==-1: return []
    else: 
        f1=str(list1[0][list1_index])
        f2=str(list2[0][list2_index])
        length=f1.count(",")
        if length==0:
            case1=get_one_case(f1)
            case2=get_one_case(f2)
            if case1==case2:
                return [list1_index,list2_index,"none"]
            elif is_variable(case1)==True and is_variable(case2)==False:
                return [list1_index,list2_index,"("+case1+"="+case2+")"]
            elif is_variable(case1)==False and is_variable(case2)==True:
                return [list1_index,list2_index,"("+case2+"="+case1+")"]
            else:
                return []
        elif length==1:
            case1=[get_first_part(f1),get_last_part(f1)]
            if count_variables(case1)>1: return []
            case2=[get_first_part(f2),get_last_part(f2)]
            if count_variables(case2)>1: return []
            x=judge_case(case1,case2)
            if x=="":return []
            else: return[list1_index,list2_index,x]

def unify(node_1, node_2, i, j, l):
    re = judge(node_1, node_2)
    if len(re) == 0:
        return l
    l.append([])
    tail1 = dic[re[0]]
    tail2 = dic[re[1]]
    if len(node_1[0]) > 1:
        a = str(i) + str(tail1)
    else:
        a = str(i)
    if len(node_2[0]) > 1:
        b = str(j) + str(tail2)
    else:
        b = str(j)
    ss = re[2]
    t = []
    l1 = node_1[0][:]
    l2 = node_2[0][:]
    for m in range(0, len(l1)):
        l1[m] = l1[m].replace(ss[1], ss[3:len(ss) - 1])
    for m in range(0, len(l2)):
        l2[m] = l2[m].replace(ss[1], ss[3:len(ss) - 1])
    node_1[0] = l1
    node_2[0] = l2
    for n in range(0, len(node_1[0])):
        if n != re[0]:
            t.append(node_1[0][n])
    for n in range(0, len(node_2[0])):
        if n != re[1] and t.count(node_2[0][n]) == 0:
            t.append(node_2[0][n])
    l[len(l) - 1].append(t)
    l[len(l) - 1].append(a)
    l[len(l) - 1].append(b)
    l[len(l) - 1].append(ss)
    return l

def get_num(s4):
    if s4[-1] <= "z" and s4[-1] >= "a":
        ans_2 = s4[0:-1]
    else:
        ans_2 = s4
    return ans_2

def increment_variable_num(s5, num_1):
    if s5[-1] <= "z" and s5[-1] >= "a":
        ans_3 = str(int(num_1) + 1) + s5[-1]
    else:
        ans_3 = str(int(num_1) + 1)
    return ans_3

def prepare_ans(l, node, ans, n):
    front = get_num(node[1])
    behind = get_num(node[2])
    if int(front) == -1 or int(behind) == -1:
        return
    tail = ""
    if node[len(node) - 1] == "none":
        tail = ""
    else:
        tail = node[3]
    f = front
    front = increment_variable_num(node[1], f)
    be = behind
    behind = increment_variable_num(node[2], be)
    # t=str("R["+str(front)+","+str(behind)+"]"+tail+" = "+str(node[0]))
    t = [str(front), str(behind), tail, str(node[0])]
    ans.append(t)
    if int(f) >= n: prepare_ans(l, l[int(f)], ans, n)
    if int(be) >= n: prepare_ans(l, l[int(be)], ans, n)

def correct_row(l, ans_d, s6):
    n = len(s6)
    for i in range(0, len(ans_d)):
        s6.append(ans_d[i]) 
    for i in range(len(s6) - 1, n - 1, -1):
        s_1 = int(get_num(s6[i][0])) - 1
        s_2 = int(get_num(s6[i][1])) - 1
        if s_1 >= n:
            ss1 = str(l[s_1][0])
            for j in range(n, len(s6)):
                if ss1 == s6[j][3]:
                    s6[i][0] = s6[i][0].replace(get_num(s6[i][0]), str(j + 1))
                    break
        if s_2 >= n:
            ss2 = str(l[s_2][0])
            for j in range(n, len(s6)):
                if ss2 == s6[j][3]:
                    s6[i][1] = s6[i][1].replace(get_num(s6[i][1]), str(j + 1))
                    break
    for i in range(0, len(ans_d)):
        ans_d[i] = s6[i + n]

def print_ans(ans_3):
    for i in range(0, len(ans_3)):
        f_ans = str("R[" + str(ans_3[i][0]) + "," + str(ans_3[i][1]) + "]" + ans_3[i][2] + " = " + str(ans_3[i][3]))
        print(f_ans)

def final(list_f, s):
    flag = 0
    sum = 0
    while True:
        l = len(list_f)
        for i in range(0, l):
            if sum == 0:
                j = i + 1
            else:
                j = l1
            for j in range(j, l):
                list1 = list_f[i]
                list2 = list_f[j]
                list_f = unify(list1[:], list2[:], i, j, list_f)
                if len(list_f[len(list_f) - 1][0]) == 0:
                    flag = 1
                    break
            if flag == 1:
                break

        if flag == 1:
            break
        sum += 1
        l1 = l
    ans = []
    prepare_ans(list_f, list_f[len(list_f) - 1], ans, np)
    ans.reverse()
    correct_row(list_f, ans, s)
    print_ans(ans)

np=int(input())
l=[]
for i in range(np):
    l.append([])
    t=input()
    if t[0]=="(":
        t=str(t[1:len(t)-1])
    t=remove_space(t)
    t=remove_comma(t.split())
    l[i].append(t)
    l[i].append("-1")
    l[i].append("-1")
    l[i].append("")
S=l[:]
final(l, S)
# print(end-start)
