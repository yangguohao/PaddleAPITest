import os

def getnum(dst):
    s=set()
    with open(dst,'r') as f:
        data=f.readlines()
        for j in data:
            tmp=j.split('(')[0]
            if tmp not in s:
                s.add(tmp)
    return len(s)

def isclean(config):
    path=os.getcwd()
    list=os.listdir(path)
    merge_list=[]
    for i in list:
        if i.find('merge')!=-1:
            merge_list.append(i)

    cnt=0
    exist_list=set()
    for i in merge_list:
        with open(i,'r',encoding='utf8') as f:
            data=f.readlines()
            for j in data:
                if config in j:
                    cnt+=1
                    exist_list.add(i)

    if cnt:
        print(config[:len(config)-1]+' is still exist, number of times : ', cnt)
        print(config[:len(config)-1]+' is still exist in these files : ')
        for i in exist_list:
            print(i)
        return 0
    else:
        print('clean')
        return 1

def movefile(config, dst):
    path=os.getcwd()
    list=os.listdir(path)
    merge_list=[]
    for i in list:
        if i.find('merge')!=-1:
            merge_list.append(i)

    for i in merge_list:
        with open(i,'r') as f:
            lines=f.readlines()

        matched_lines = [line for line in lines if config in line]
        remaining_lines = [line for line in lines if config not in line]

        with open(i, 'w') as f:
            f.writelines(remaining_lines)

        with open(dst, 'a+') as f:
            f.writelines(matched_lines)

        if len(matched_lines):
            print("move", len(matched_lines), "lines in" , i , "to ", dst,'successfully')

def main():
    index=3
    config = "paddle.nn.functional.adaptive_log_softmax_with_loss"
    config+='('
    # dst = "api_config_support2torch_"+str(index)+".txt"
    dst='mytmp.txt'
    # print(getnum(dst))
    if not isclean(config):
        movefile(config, dst)

if __name__ == '__main__':
    main()