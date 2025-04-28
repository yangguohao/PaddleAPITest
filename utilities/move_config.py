import os

path = '/PaddleAPITest/tester/api_config/'

def getnum(dst):
    s=set()
    with open(path + dst,'r') as f:
        data=f.readlines()
        for j in data:
            tmp=j.split('(')[0]
            if tmp not in s:
                s.add(tmp)
    return len(s)

def isclean(config):
    list=os.listdir(path)
    merge_list=[]
    for i in list:
        if i.find('merge')!=-1:
            merge_list.append(i)

    cnt = 0
    exist_list = set()
    exist_nums = {}
    for i in merge_list:
        cnt2 = 0
        with open(path + i,'r',encoding='utf8') as f:
            data=f.readlines()
            for j in data:
                if config in j:
                    cnt += 1
                    cnt2 += 1
                    exist_list.add(i)
        exist_nums[i] = cnt2

    if cnt:
        print(config[:len(config)-1]+' is still exist, number of times : ', cnt)
        print(config[:len(config)-1]+' is still exist in these files : ')
        for i in exist_list:
            print(i, exist_nums[i])
        return 0
    else:
        print('clean')
        return 1

def add(config, dst):
    list=os.listdir(path)
    merge_list=[]
    for i in list:
        if i.find('merge')!=-1:
            merge_list.append(i)

    for i in merge_list:
        with open(path + i,'r') as f:
            lines=f.readlines()

        matched_lines = [line for line in lines if config in line]

        with open(path + dst, 'a+') as f:
            f.writelines(matched_lines)

        if len(matched_lines):
            print("add", len(matched_lines), "lines in" , i , "to ", dst,'successfully')

def remove(config):
    path=os.getcwd()
    list=os.listdir(path)
    merge_list=[]
    for i in list:
        if i.find('merge')!=-1:
            merge_list.append(i)

    for i in merge_list:
        with open(path + i,'r') as f:
            lines=f.readlines()

        matched_lines = [line for line in lines if config in line]
        remaining_lines = [line for line in lines if config not in line]

        with open(path + i, 'w') as f:
            f.writelines(remaining_lines)

        if len(matched_lines):
            print("remove", len(matched_lines), "lines in" , i , 'successfully')

def main():
    config = "paddle.pow"
    config+='('
    dst='mytmp.txt'        #替换为你想要的临时文件名

    if not isclean(config):
        add(config, dst)   #向指定临时文件以a+方式添加写入，不会改变原有配置

    # remove(config)       #这一步要删除配置，请慎重

if __name__ == '__main__':
    main()