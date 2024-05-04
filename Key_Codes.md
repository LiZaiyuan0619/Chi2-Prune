# 5.4_YOLO_Chi2_Prune

## load_yolov5_model

```python
def load_yolov5_model(model, oristate_dict, device):

    if torch.cuda.device_count() > 1:
        name_base = ''
    else:
        name_base = ''

    state_dict = model.state_dict()

    # 用于加载秩文件
    cnt = 0
    prefix = 'rank_conv/yolov5_limit6/rank_conv'
    subfix = ".npy"

    # 处理Conv顺带遍历处理BN部分
    bn_part_name = ['.weight', '.bias', '.running_mean', '.running_var']

    # Conv index selected in the previous layer
    last_select_index = None
    last_name = None

    # 处理C3模块，保存cv1和cv2的select_index,前者在bottleneck，后者和bottleneck得到的select_index一起用于cv3
    last_select_index1 = None
    last_select_index2 = None

    last_select_index_concat1 = None
    last_select_index_concat2 = None

    # 处理C3模块，保存cv3的select_index，当前C3的后续卷积层使用
    last_select_index_cv3 = None

    # bottleneck得到的select_index，就是n个bottleneck其中最后一个卷积层的select_index
    last_select_index_bottleneck = None

    last4_select_index = None
    last6_select_index = None
    last10_select_index = None
    last14_select_index = None

    last17_select_index = None
    last20_select_index = None
    last23_select_index = None

    # bottleneck标志位为0时说明当前在bottleneck的第一个卷积层，为1时说明当前在bottleneck的第二个卷积层
    flag_bottleneck = 0

    # while循环标志位为0时说明C3中的cv3还没进行回溯，为1时说明C3中的cv3已经进行回溯
    flag_while = 0
    # 记录cv3的while的i，回溯时跳转到这个i
    k_cv3 = 0
    # 回溯并处理cv3后我们不继续处理bottleneck部分，而是直接跳转到bottleneck最后一个卷积层
    k_bottleneck = 0
    # 记录cv3的cnt，回溯时使用这个cnt加载秩
    cnt_cv3 = 0

    # 这样处理方便在当前循环获得下一个卷积层的名称特征
    named_modules_list = list(model.named_modules())

    k = 0
    # while循环获得的内容比参数文件要多得多，所以我们到 model.9.cv2.conv就退出
    # 用while方便我们回溯或跳转
    while k < len(named_modules_list):
        name, module = named_modules_list[k]

        # 这里取巧了，其实这里并不一定是下一个卷积层，只是刚刚好匹配了bottleneck后面是卷积块的情况
        # 后面是用'm' in next_conv_name来判断的,当不是前面说的刚刚好情况时（也就是n>1个bottleneck情况下）仍然可以正常进入bottleneck处理
        # 我们可以识别这是n>1个bottleneck，是因为在当前层——bottleneck的第二个卷积层——的下一个卷积层中有 m，比如model.4.m.1.cv1
        if k + 4 < len(named_modules_list):
            next_conv_name, next_conv_module = named_modules_list[k + 4]
        # 处理多GPU
        name = name.replace('module.', '')

        next_conv_name = next_conv_name.replace('module.', '')

        bn_name = name.replace('conv', 'bn')

        if name == "model.24":
            print("================================================================")
            print("anchors")
            state_dict[name_base + "model.24.anchors"] = oristate_dict["model.24.anchors"]
            print("state_dict[name_base + model.24.anchors]", state_dict[name_base + "model.24.anchors"])
            print("================================================================")


        # # 倒数第二层，此时名称已经不再和参数文件层一致了，但是可以替换后按照卷积层来处理，last_select_index由于C3中cv1结构也是正常的
        # if name == 'model.9.cv1.conv':
        #     bn_name = name.replace('conv', 'bn')
        #     name = name.replace('model.9.cv1.conv', 'model.9.conv.conv')
        #
        # # 最后一层，参数文件是全连接，直接加载后break
        # if name == 'model.9.cv2.conv':
        #     name = name.replace('cv2.conv', 'linear')
        #     state_dict[name_base + name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base + name + '.bias'] = oristate_dict[name + '.bias']
        #     break
        # 最后一层，参数文件是全连接，直接加载后break

        # if name == 'model.9.conv.conv':
        #     state_dict[name_base + name + '.weight'] = oristate_dict[name + '.weight']
        #     bn_name = 'model.9.conv.bn'
        #     for bn_part in bn_part_name:
        #         state_dict[name_base + bn_name + bn_part] = oristate_dict[bn_name + bn_part]
        #     linear_name = 'model.9.linear'
        #
        #     state_dict[name_base + linear_name + '.weight'] = oristate_dict[linear_name + '.weight']
        #     state_dict[name_base + linear_name + '.bias'] = oristate_dict[linear_name + '.bias']
        #     print("load_yolov5_model finished")
        #     break
        
        
        # 只有卷积层才有后续处理，这一步过滤很多情况比如 bn act等
        if isinstance(module, nn.Conv2d):
            print("#################################################")
            print("k:", k)
            # print("Processing name:", name)
            print("Processing Conv2d Name: ", name)
            print("Processing Conv2d module: ", module)
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            print("orifilter_num: ", orifilter_num)
            currentfilter_num = curweight.size(0)
            print("currentfilter_num: ", currentfilter_num)

            cov_id = cnt
            # 当现在是回溯情况时，加载的秩文件是cv3的，所以要用cv3的cnt
            # 由于continue直接忽略了回溯过程中cnt++，所以不用对cnt操作
            if flag_while == 1:
                print("Now it's backtracking", name)
                cov_id = cnt_cv3


            # 第一种大情况：当前层需要进行剪枝，这一种最复杂，要分为两种情况：前面的层剪枝，前面的层没剪枝
            # 如果前面的层已经剪枝，需要继续细分为两种情况：cv3和正常卷积层；如果前面的层没剪枝，相同
            # cv3需要last_select_index1和last_select_index2选择通道，而正常卷积层只需要last_select_index选择通道
            if orifilter_num != currentfilter_num:
                print("The current layer needs to be pruned")
                # logger.info('loading rank from: ' + prefix + str(cov_id) + subfix)
                print('loading rank from: ' + prefix + str(cov_id) + subfix)

                rank = np.load(prefix + str(cov_id) + subfix)
                select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                select_index.sort()
                print("select_index: ", select_index)

                # 当前一层（cv3的话就是前面两层）已经剪枝并且当前层需要剪枝的情况
                if (last_select_index1 is not None and last_select_index2 is not None) or last_select_index is not None:
                    if last_select_index is not None:
                        print("last_select_index is not None")
                    elif last_select_index1 is not None and last_select_index2 is not None:
                        print("last_select_index1 is not None and last_select_index2 is not None")
                    print("last conv layer has been pruned")

                    # cv3的情况
                    # if ('cv3.conv' in name) and ('m' not in name):
                    if ('cv3.conv' in name):
                        print("Now determine whether is backtracking")
                        # 对每一个cv3层，跑的第二次（回溯时）才会真正加载参数，第一次来并没有加载，而是直接进行后续的操作
                        # 只有在last_select_index_bottleneck is not None时也就是获得了bottleneck的select_index时才会加载参数
                        if last_select_index_bottleneck is not None:
                            print("Now it's the second time to come to cv3, we prune")
                            print("last_name: ", last_name)
                            print("last_select_index1 from bottleneck: ", last_select_index1)
                            print("last_select_index2: ", last_select_index2)
                            print("last_select_index_bottleneck: ", last_select_index_bottleneck)
                            last_select_index_bottleneck = None

                            # 在上一层末尾的处理情况中已经将last_select_index1赋值为了last_select_index_bottleneck
                            # 因为concat，所以这里是一半用last_select_index1，一半用last_select_index2
                            for index_i, i in enumerate(select_index):
                                for index_j in range(2 * len(last_select_index1)):
                                    if index_j < len(last_select_index1):
                                        state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                            oristate_dict[name + '.weight'][i][last_select_index1[index_j]]

                                    else:
                                        state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                            oristate_dict[name + '.weight'][i][
                                                last_select_index2[index_j - len(last_select_index1)]]
                            # bn部分都是正常处理
                            for bn_part in bn_part_name:
                                for index_i, i in enumerate(select_index):
                                    state_dict[name_base + bn_name + bn_part][index_i] = \
                                        oristate_dict[bn_name  + bn_part][i]
                            # 当走到这里时说明已经回溯完了，我们不再继续处理last_select_index类，而是直接跳回开始准备处理cv3后面的卷积层
                            k = k_bottleneck+1
                            k_bottleneck = 0
                            # 对于cv3下一层来说，last_select_index就是last_select_index_cv3
                            last_select_index = last_select_index_cv3
                            last_select_index1 = None
                            last_select_index2 = None

                            last_name = name
                            print("next last_select_index is last_select_index_cv3 for next Conv is CBS")
                            print("next last_name is cv3")

                            if "model.4" in name:
                                last4_select_index = last_select_index_cv3
                                print(f"Now name is: {name}, so update last4_select_index: {last4_select_index}")
                            elif "model.6" in name:
                                last6_select_index = last_select_index_cv3
                                print(f"Now name is:: {name}, so update last6_select_index: {last6_select_index}")

                            # 回溯结束，进入正常情况，“第一次”
                            flag_while = 0
                            print("backtracking finished, continue!!!!!!!!!!!!")
                            continue
                        else:
                            print('Now it is the first time to come to cv3, we do nothing')


                    # 针对特定的拼接后四层cv1和cv2处理
                    elif("13" in name or "17" in name or "20" in name or "23" in name) and ("cv1" in name or "cv2" in name) and ("m." not in name):
                        print(f"Concat before. Currently processing {name}")
                        last_select_index_concat1 = last_select_index
                        print(f"last_select_index_concat1 or last_select_index: {last_select_index_concat1}")
                        if "23" in name:
                            last_select_index_concat2 = last10_select_index
                            print(f"last_select_index_concat2 or last10_select_index: {last_select_index_concat2}")
                        elif "20" in name:
                            last_select_index_concat2 = last14_select_index
                            print(f"last_select_index_concat2 or last14_select_index: {last_select_index_concat2}")
                        elif "17" in name:
                            last_select_index_concat2 = last4_select_index
                            print(f"last_select_index_concat2 or last4_select_index: {last_select_index_concat2}")
                        elif "13" in name:
                            last_select_index_concat2 = last6_select_index
                            print(f"last_select_index_concat2 or last6_select_index: {last_select_index_concat2}")

                        for index_i, i in enumerate(select_index):
                            for index_j in range(len(last_select_index_concat1) + len(last_select_index_concat2)):
                                if index_j < len(last_select_index_concat1):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][last_select_index_concat1[index_j]]
                                else:
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][
                                            last_select_index_concat2[index_j - len(last_select_index_concat1)]]

                        for bn_part in bn_part_name:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]
                    # 一般卷积层的情况
                    else:
                        print("For non-cv3 convolutional layers, we prune")
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                            for bn_part in bn_part_name:
                                state_dict[name_base + bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]

                        if "model.10" in name:
                            last10_select_index = select_index
                            print(f"now name is {name}, so update last10_select_index{last10_select_index}")
                        elif "model.14" in name:
                            last14_select_index = select_index
                            print(f"now name is {name}, so update last14_select_index{last14_select_index}")

                # 当前层需要剪枝但是前面层没剪枝
                else:
                    print("last conv layer has not been pruned but we need to prune now")
                    # cv3的情况
                    # if ('cv3.conv' in name) and ('m' not in name):
                    if ('cv3.conv' in name):
                        print("Now determine whether is backtracking")
                        # 对每一个cv3层，跑的第二次（回溯时）才会真正加载参数，第一次来并没有加载，而是直接进行后续的操作
                        # 只有在last_select_index_bottleneck is not None时也就是获得了bottleneck的select_index时才会加载参数
                        if last_select_index_bottleneck is not None:
                            print("Now it's the second time to come to cv3, we prune")
                            print("last_name: ", last_name)
                            print("last_select_index1 from bottleneck: ", last_select_index1)
                            print("last_select_index2: ", last_select_index2)
                            print("last_select_index_bottleneck: ", last_select_index_bottleneck)
                            last_select_index_bottleneck = None

                            # 在上一层末尾的处理情况中已经将last_select_index1赋值为了last_select_index_bottleneck
                            # 因为concat，所以这里是一半用last_select_index1，一半用last_select_index2
                            for index_i, i in enumerate(select_index):
                                for index_j in range(2 * len(last_select_index1)):
                                    if index_j < len(last_select_index1):
                                        state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                            oristate_dict[name + '.weight'][i][last_select_index1[index_j]]

                                    else:
                                        state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                            oristate_dict[name + '.weight'][i][
                                                last_select_index2[index_j - len(last_select_index1)]]
                            # bn部分都是正常处理
                            for bn_part in bn_part_name:
                                for index_i, i in enumerate(select_index):
                                    state_dict[name_base + bn_name + bn_part][index_i] = \
                                        oristate_dict[bn_name  + bn_part][i]

                            # 当走到这里时说明已经回溯完了，我们不再继续处理last_select_index类，而是直接跳回开始准备处理cv3后面的卷积层
                            k = k_bottleneck+1
                            k_bottleneck = 0
                            # 对于cv3下一层来说，last_select_index就是last_select_index_cv3
                            last_select_index = last_select_index_cv3
                            last_name = name
                            print("next last_select_index is last_select_index_cv3 for next Conv is CBS")
                            print("next last_name is cv3")

                            last_select_index1 = None
                            last_select_index2 = None

                            if "model.4" in name:
                                last4_select_index = last_select_index_cv3
                                print(f"Now name is: {name}, so update last4_select_index: {last4_select_index}")
                            elif "model.6" in name:
                                last6_select_index = last_select_index_cv3
                                print(f"Now name is:: {name}, so update last6_select_index: {last6_select_index}")

                            # 回溯结束，进入正常情况，“第一次”
                            flag_while = 0
                            print("backtracking finished, continue!!!!!!!!!!!!")
                            continue
                        else:
                            print('Now it is the first time to come to cv3, we do nothing')

                    else:
                        for index_i, i in enumerate(select_index):

                           state_dict[name_base+name + '.weight'][index_i] = \
                                oristate_dict[name + '.weight'][i]

                           for bn_part in bn_part_name:
                               state_dict[name_base + bn_name + bn_part][index_i] = \
                                   oristate_dict[bn_name + bn_part][i]
                        if "model.10" in name:
                            last10_select_index = select_index
                            print(f"now name is  {name}, so update last10_select_index {last10_select_index}")
                        elif "model.14" in name:
                            last14_select_index = select_index
                            print(f"now name is  {name}, so update last14_select_index{last14_select_index}")

                # 无论怎样当前层剪枝了，就要处理给下一层留last_select_index
                # 如果是cv1就是获取last_select_index1留给bottleneck用，此时last_select_index不动，留给cv2用
                print("last_select_index: ", last_select_index)
                print("last_name: ", last_name)

                # 根据while遍历顺序，当第一次cv3后就进入bottleneck。由于每个bottleneck有两个卷积层，并且bottleneck数量不确定，所以需要特殊处理
                if 'm.' in name:
                    print("bottleneck")
                    # 如果是bottleneck的第一个卷积层，正常进行
                    if flag_bottleneck == 0:
                        print("bottleneck 0")
                        last_select_index = select_index
                        last_name = name
                        flag_bottleneck = 1
                    # 如果是bottleneck的第二个卷积层，需要继续判定，到底下面是进入C3后的卷积层，还是回溯，还是进入下一个bottleneck
                    else:
                        print("bottleneck 1")
                        # 下一层是bottoleneck
                        if 'm.' in next_conv_name:
                            print("next bottleneck still")
                            last_select_index = select_index
                            last_name = name
                        # 下一层是C3后的卷积层
                        else:
                            # 第一次来，就要进行回溯，这里其实也仅仅会在第一次来，第二次直接continue进入C3后的卷积层了
                            print("prepare for backtracking")
                            if flag_while == 0:
                                # 获取当前序号，即bottleneck一个卷积层的序号
                                k_bottleneck = k
                                # last_select_index_bottleneck就是回溯时last_select_index1，回溯时才会真正加载参数
                                last_select_index_bottleneck = select_index
                                print("get last_select_index_bottleneck: ", last_select_index_bottleneck)
                                last_select_index1 = last_select_index_bottleneck
                                last_select_index = None
                                print("next last_select_index is None for next Conv is cv3")
                                last_name = name
                                # 进行回溯，没有continue
                                k = k_cv3 - 1
                                flag_while = 1
                        # 无论怎样保证这是第二层处理结束了
                        flag_bottleneck = 0
                elif 'cv1.conv' in name:
                    print("cv1")
                    last_select_index1 = select_index
                    print("get last_select_index1 or current select_index: ", last_select_index1)
                    last_select_index = last_select_index
                    print("next last_select_index still")
                    last_name = last_name
                    print("next last_name still")
                    if "model.9" in name:
                        last_select_index = select_index
                        last_name = name
                        last_select_index1 = None
                # 如果是cv2就是获取last_select_index2留给cv3用，此时last_select_index是上一层的select_index,不用再保留了
                elif 'cv2.conv' in name:
                    print("cv2")
                    last_select_index2 = select_index
                    print("get last_select_index2 or current select_index: ", last_select_index2)
                    last_select_index = None
                    print("next last_select_index is None for next Conv is cv3")
                    last_name = "Should be m.cv2, but from cv2"
                    if "model.9" in name:
                        last_select_index = select_index
                        last_name = name
                        last_select_index2 = None
                # 每个cv3只可能进入这个分支一次，也就是第一次，此时根据while遍历顺序下面是bottleneck部分
                # 我们保留last_select_index_cv3用于C3模块后面的卷积层用，last_select_index因为后面处理bottleneck而变为last_select_index1
                # 为了保证不会误出现last_select_index1 is not None and last_select_index2 is not None，将last_select_index1置为None
                # last_select_index2必须保留，我们还要用它来处理cv3，记录当前i和cnt回溯使用
                elif 'cv3.conv' in name:
                    print("cv3")
                    last_select_index_cv3 = select_index
                    print("get last_select_index_cv3 or current select_index for backtracking: ", last_select_index_cv3)
                    last_select_index = last_select_index1
                    print("next last_select_index is last_select_index1 for next Conv is from bottleneck")
                    last_name = "Should be cv1, but from cv3"
                    print("next last_name is cv1")
                    last_select_index1 = None
                    k_cv3 = k
                    cnt_cv3 = cnt
                    print("get k and cnt for backtracking")
                    if "17" in name:
                        last17_select_index = select_index
                        print(f"Now name is: {name}, so update last17_select_index {last17_select_index}")
                    elif "20" in name:
                        last20_select_index = select_index
                        print(f"Now name is: {name}, so update last20_select_index {last20_select_index}")
                    elif "23" in name:
                        last23_select_index = select_index
                        print(f"Now name is: {name}, so update last23_select_index {last23_select_index}")

                # 最一般加载参数情况，线性堆叠的卷积层
                else:
                    print("Normal Conv2d")
                    last_select_index = select_index
                    last_name = name


            # 第二大情况中的第一种情况：当前层不需要剪枝，但是前面层已经剪枝，我们同样分为两种情况来处理：当前层是cv3；当前层是正常卷积层
            elif (last_select_index1 is not None and last_select_index2 is not None) or last_select_index is not None:
                if last_select_index is not None:
                    print("last_select_index is not None")
                elif last_select_index1 is not None and last_select_index2 is not None:
                    print("last_select_index1 is not None and last_select_index2 is not None")
                print("last conv layer has been pruned but we don't need to prune now")
                # cv3的情况，这里就没什么特定的处理，前面已经剪枝了就一半一半地直接加载
                if 'cv3.conv' in name:
                    # print("cv3")
                    # print("last_select_index1: ", last_select_index1)
                    # print("last_select_index2: ", last_select_index2)
                    # print("last_name: ", last_name)

                    print("Now determine whether is backtracking")
                    # 对每一个cv3层，跑的第二次（回溯时）才会真正加载参数，第一次来并没有加载，而是直接进行后续的操作
                    # 只有在last_select_index_bottleneck is not None时也就是获得了bottleneck的select_index时才会加载参数
                    if last_select_index_bottleneck is not None:
                        print("Now it's the second time to come to cv3, we prune")
                        print("last_name: ", last_name)
                        print("last_select_index1 from bottleneck: ", last_select_index1)
                        print("last_select_index2: ", last_select_index2)
                        print("last_select_index_bottleneck: ", last_select_index_bottleneck)
                        last_select_index_bottleneck = None

                        for i in range(orifilter_num):
                            index_j_total = 0
                            for index_j, j in enumerate(last_select_index1):
                                state_dict[name_base + name + '.weight'][i][index_j_total] = \
                                    oristate_dict[name + '.weight'][i][j]
                                index_j_total += 1
                            for index_j, j in enumerate(last_select_index2):
                                state_dict[name_base + name + '.weight'][i][index_j_total] = \
                                    oristate_dict[name + '.weight'][i][j + len(last_select_index1)]
                                index_j_total += 1

                        # 加载BN层参数
                        for bn_part in bn_part_name:
                            state_dict[name_base + bn_name + bn_part] = \
                                oristate_dict[bn_name + bn_part]

                        if "model.4" in name:
                            last4_select_index = list(range(currentfilter_num))
                            print(f"Now name is : {name}, so update last4_select_index {last4_select_index}")
                        elif "model.6" in name:
                            last6_select_index = list(range(currentfilter_num))
                            print(f"Now name is {name}, so update last6_select_index {last6_select_index}")

                        # 当走到这里时说明已经回溯完了，我们不再继续处理last_select_index类，而是直接跳回开始准备处理cv3后面的卷积层
                        k = k_bottleneck + 1
                        k_bottleneck = 0
                        # 对于cv3下一层来说，last_select_index就是last_select_index_cv3
                        last_select_index = last_select_index_cv3
                        last_select_index1 = None
                        last_select_index2 = None

                        last_name = name
                        print("next last_select_index is last_select_index_cv3 for next Conv is CBS")
                        print("next last_name is cv3")

                        # 回溯结束，进入正常情况，“第一次”
                        flag_while = 0
                        print("backtracking finished, continue!!!!!!!!!!!!")
                        continue
                    else:
                        print('Now it is the first time to come to cv3, we do nothing')

                        print("cv3")
                        last_select_index_cv3 = list(range(currentfilter_num))
                        print("get last_select_index_cv3 or current select_index for backtracking: ",
                              last_select_index_cv3)
                        last_select_index = last_select_index1
                        print("next last_select_index is last_select_index1 for next Conv is from bottleneck")
                        last_name = "Should be cv1, but from cv3"
                        print("next last_name is cv1")
                        last_select_index1 = None
                        k_cv3 = k
                        cnt_cv3 = cnt
                        print("get k and cnt for backtracking")

                        if "17" in name:
                            last17_select_index = list(range(currentfilter_num))
                            print(
                                f"Now name is: {name}, so update last17_select_index {last17_select_index}")
                        elif "20" in name:
                            last20_select_index = list(range(currentfilter_num))
                            print(
                                f"Now name is: {name}, so update last20_select_index {last20_select_index}")
                        elif "23" in name:
                            last23_select_index = list(range(currentfilter_num))
                            print(
                                f"Now name is: {name}, so update last23_select_index  {last23_select_index}")

                elif ("13" in name or "17" in name or "20" in name or "23" in name) and ("cv1" in name or "cv2" in name) and ("m." not in name):
                    print(f"Concat before. Currently processing {name}")
                    print("last_name: ", last_name)

                    # Determine which index list to use based on the layer name
                    if "23" in name:
                        last_select_index_concat2 = last10_select_index
                    elif "20" in name:
                        last_select_index_concat2 = last14_select_index
                    elif "17" in name:
                        last_select_index_concat2 = last4_select_index
                    elif "13" in name:
                        last_select_index_concat2 = last6_select_index

                    last_select_index_concat1 = last_select_index
                    # Begin the weight reassignment
                    for i in range(orifilter_num):
                        index_j_total = 0  # This keeps track of the position in the concatenated layer

                        # Assign weights from the first part of the concatenation (up-sampled layer)
                        for index_j, j in enumerate(last_select_index_concat1):
                            state_dict[name_base + name + '.weight'][i][index_j_total] = \
                                oristate_dict[name + '.weight'][i][j]
                            index_j_total += 1

                        # Assign weights from the second part of the concatenation (specific layer)
                        for index_j, j in enumerate(last_select_index_concat2):
                            state_dict[name_base + name + '.weight'][i][index_j_total] = \
                                oristate_dict[name + '.weight'][i][j + len(last_select_index_concat1)]
                            index_j_total += 1

                    for bn_part in bn_part_name:
                        state_dict[name_base + bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]
                    if "cv1" in name:
                        last_select_index = last_select_index
                        last_select_index1 = list(range(currentfilter_num))
                    else:
                        last_select_index = None
                        last_select_index2 = list(range(currentfilter_num))
                    last_name = name

                elif "24" in name:
                    print("Detect or last layer")
                    print("Whatever last layer is pruned, we just consider it pruned to make it easier")
                    if "24.m.0" in name:
                        last_select_index = last17_select_index
                        print("last_select_index is from last17_select_index")
                    elif "24.m.1" in name:
                        last_select_index = last20_select_index
                        print("last_select_index is from last20_select_index")
                    elif "24.m.2" in name:
                        last_select_index = last23_select_index
                        print("last_select_index is from last23_select_index")
                    print("last_select_index: ", last_select_index)

                    # 在这里需要对Detect进行赋值处理，就是说将没有剪枝的权重参数更新为剪枝后模型的权重参数。
                    # 输出是保持不变的，只有输入通道数减少，需要选择last_select_index的对应通道
                    # 最后请输出权重对应的size，显示出剪枝后权重的维度
                    for i in range(orifilter_num):  # 遍历输出通道
                        # 初始化新权重的当前输出通道的存储空间
                        new_weights = torch.zeros((state_dict[name_base + name + '.weight'].size(1),
                                                   state_dict[name_base + name + '.weight'].size(2),
                                                   state_dict[name_base + name + '.weight'].size(3)))

                        for index, j in enumerate(last_select_index):
                            # print("index: ", index)
                            # print("j: ", j)
                            if index < state_dict[name_base + name + '.weight'].size(1):
                                new_weights[index] = oristate_dict[name + '.weight'][i][j]


                        # 将新权重赋值给剪枝后的状态字典
                        state_dict[name_base + name + '.weight'][i] = new_weights

                    state_dict[name_base + name + '.bias'] = oristate_dict[name + '.bias']
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("state_dict[name_base + name + '.bias']: ", state_dict[name_base + name + '.bias'])
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                # 一般卷积层的情况
                else:
                    print("For non-cv3 convolutional layers, we don't need to prune now")
                    print("last_select_index: ", last_select_index)
                    print("last_name: ", last_name)

                    for i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+name + '.weight'][i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                    for bn_part in bn_part_name:
                        state_dict[name_base + bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]

                    last_name = name

                    if "model.10" in name:
                        last10_select_index = list(range(currentfilter_num))
                        print(f"Now name is : {name}, so update last10_select_index {last10_select_index}")
                    elif "model.14" in name:
                        last14_select_index = list(range(currentfilter_num))
                        print(f"Now name is : {name}, so update last14_select_index {last14_select_index}")

                    if "cv1" in name and "9" not in name and "m." not in name:
                        last_select_index1 = list(range(currentfilter_num))
                        last_select_index = last_select_index
                        print(f"Now name is : {name}, next last_select_index still {last_select_index}")
                    elif "cv2" in name and "9" not in name and "m." not in name:
                        last_select_index2 = list(range(currentfilter_num))
                        last_select_index = None
                        print(f"Now name is : {name}, next last_select_index is None")
                    else:
                        last_select_index = None

                    if ("10" in name or "14" in name or "18" in name or "21" in name):
                        last_select_index = list(range(currentfilter_num))
                        print(f"Now name is: {name}, so update last_select_index {last_select_index}")

                    if 'm.' in name:
                        print("bottleneck")
                        # 如果是bottleneck的第一个卷积层，正常进行
                        if flag_bottleneck == 0:
                            print("bottleneck 0")
                            last_select_index = list(range(currentfilter_num))
                            last_name = name
                            flag_bottleneck = 1
                        # 如果是bottleneck的第二个卷积层，需要继续判定，到底下面是进入C3后的卷积层，还是回溯，还是进入下一个bottleneck
                        else:
                            print("bottleneck 1")
                            # 下一层是bottoleneck
                            if 'm.' in next_conv_name:
                                print("next bottleneck still")
                                last_select_index = list(range(currentfilter_num))
                                last_name = name
                            # 下一层是C3后的卷积层
                            else:
                                # 第一次来，就要进行回溯，这里其实也仅仅会在第一次来，第二次直接continue进入C3后的卷积层了
                                print("prepare for backtracking")
                                if flag_while == 0:
                                    # 获取当前序号，即bottleneck一个卷积层的序号
                                    k_bottleneck = k
                                    # last_select_index_bottleneck就是回溯时last_select_index1，回溯时才会真正加载参数
                                    last_select_index_bottleneck = list(range(currentfilter_num))
                                    print("get last_select_index_bottleneck: ", last_select_index_bottleneck)
                                    last_select_index1 = last_select_index_bottleneck
                                    last_select_index = None
                                    print("next last_select_index is None for next Conv is cv3")
                                    last_name = name
                                    # 进行回溯，没有continue
                                    k = k_cv3 - 1
                                    flag_while = 1
                            # 无论怎样保证这是第二层处理结束了
                            flag_bottleneck = 0

            # 第二大情况中的第二种情况：当前层不需要剪枝，前面层也没剪枝，直接加载
            else:
                print("last conv layer has not been pruned and we don't need to prune now")

                state_dict[name_base+name + '.weight'] = oriweight
                for bn_part in bn_part_name:
                    state_dict[name_base + bn_name + bn_part] = \
                        oristate_dict[bn_name + bn_part]

                if "model.10" in name:
                    last10_select_index = list(range(currentfilter_num))
                    print(f"Now name is : {name}, so update last10_select_index{last10_select_index}")
                elif "model.14" in name:
                    last14_select_index = list(range(currentfilter_num))
                    print(f"Now name is: {name}, so update last14_select_index {last14_select_index}")

                if "17" in name and "m." not in name and "cv2" in name:
                    last17_select_index = list(range(currentfilter_num))
                    print(f"Now name is: {name}, so update last17_select_index {last17_select_index}")
                elif "20" in name and "m." not in name and "cv2" in name:
                    last20_select_index = list(range(currentfilter_num))
                    print(f"Now name is: {name}, so update last20_select_index {last20_select_index}")
                elif "23" in name and "m." not in name and "cv2" in name:
                    last23_select_index = list(range(currentfilter_num))
                    print(f"Now name is: {name}, so update last23_select_index {last23_select_index}")

                # 根据while遍历顺序，当第一次cv3后就进入bottleneck。由于每个bottleneck有两个卷积层，并且bottleneck数量不确定，所以需要特殊处理
                if 'm.' in name and "cv2" in name:
                    print("bottleneck 1")
                    # 下一层是bottoleneck
                    if 'm.' in next_conv_name:
                        print("next bottleneck still")
                        print("last_select_index is None")
                        last_select_index = None
                        last_name = name
                    # 下一层是C3后的卷积层
                    else:
                        print("prepare for backtracking")
                        # 获取当前序号，即bottleneck一个卷积层的序号
                        k_bottleneck = k
                        # last_select_index_bottleneck就是回溯时last_select_index1，回溯时才会真正加载参数
                        last_select_index_bottleneck = list(range(currentfilter_num))
                        print("get last_select_index_bottleneck: ", last_select_index_bottleneck)
                        last_select_index1 = last_select_index_bottleneck
                        last_select_index = None
                        print("next last_select_index is None for next Conv is cv3")
                        last_name = name
                        # 进行回溯，没有continue
                        k = k_cv3 - 1
                        flag_while = 1
                elif "cv1" in name and "9" not in name and "m." not in name:
                    last_select_index1 = list(range(currentfilter_num))
                    last_select_index = last_select_index
                    print(f"Now name is : {name}, next last_select_index still {last_select_index}")
                elif "cv2" in name and "9" not in name and "m." not in name:
                    last_select_index2 = list(range(currentfilter_num))
                    last_select_index = None
                    print(f"Now name is : {name}, next last_select_index is None")
                else:
                    print("Normal Conv2d")
                    last_select_index = None
                    if ("10" in name or "14" in name or "18" in name or "21" in name):
                        last_select_index = list(range(currentfilter_num))
                        print(f"Now name is: {name}, so update last_select_index {last_select_index}")
                    last_name = name

            # 只要是卷积层就要++

            cnt += 1
        else:
            print("Not Conv2d Name:", name)

        print("#################################################")
        k = k + 1

    model.load_state_dict(state_dict)
```

## parse_model

```python
def parse_model(compress_rate, d, ch):  # model_dict, input_channels(3)

    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 从模型配置文件中读取 'anchors'、'nc'、'depth_multiple' 和 'width_multiple' 的值，如果没有，则使用默认值。
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # 每个尺度的锚点数量（na）和输出数量（no）。输出数量是基于类别数和每个锚点的额外属性（如对象置信度等）。
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    print("no ", no)


    # 初始化存储网络层的列表（layers）、保存列表（save）和输出通道数（c2）。
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    overall_channel, mid_channel_c3, mid_channel_sppf, mid_channel_bottleneck = adapt_channel(
        compress_rate)
    print("Now it is Parse Model")
    # logger.info("overall_channel: %s", overall_channel)
    # logger.info("mid_channel_c3: %s", mid_channel_c3)
    # logger.info("mid_channel_sppf: %s", mid_channel_sppf)
    # logger.info("mid_channel_bottleneck: %s", mid_channel_bottleneck)
    # 遍历配置字典中的 backbone（主干网络）和 head（检测头），其中 f 是输入层索引，n 是重复次数，m 是模块类型，args 是模块参数。
    cnt_C3 = 0
    # 使用 tqdm 包装原来的迭代器
    for i, (f, n, m, args) in enumerate(tqdm(d['backbone'] + d['head'], desc="Processing Progress")): # from, number, module, args
        # 动态指定模块

        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            # 忽略任何 NameError 异常，这可能发生在字符串不对应当前作用域中的有效名称
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        # n 根据 depth_multiple（gd）进行调整。这允许模型动态地缩放网络深度（层数）。
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 将 c1 和 c2 设置为输入和输出通道。同时，除非是输出层，否则根据 width_multiple（gw）调整 c2。
        if m in (Conv, Bottleneck, SPPF, C3 ):
            # 特殊处理某些模块的参数
            if m == Conv:
                if i == 0:
                    c1, c2 = 3, overall_channel[0]
                elif i == 1:
                    c1, c2 = overall_channel[0], overall_channel[1]
                elif i == 3:
                    c1, c2 = overall_channel[2], overall_channel[3]
                elif i == 5:
                    c1, c2 = overall_channel[4], overall_channel[5]
                elif i == 7:
                    c1, c2 = overall_channel[6], overall_channel[7]

                elif i == 10:
                    c1, c2 = overall_channel[9], overall_channel[10]
                elif i == 14:
                    c1, c2 = overall_channel[11], overall_channel[12]
                elif i == 18:
                    c1, c2 = overall_channel[13], overall_channel[14]
                elif i == 21:
                    c1, c2 = overall_channel[15], overall_channel[16]

                args = [c1, c2, *args[1:]]
            elif m == C3:
                # print("C3 ", i)
                if i < 10:
                    c1, c2 = overall_channel[i - 1], overall_channel[i]
                if i == 2:
                    mid1, mid2 = mid_channel_c3[cnt_C3], mid_channel_bottleneck[0]
                elif i == 4:
                    mid1, mid2 = mid_channel_c3[cnt_C3], mid_channel_bottleneck[1:3]
                elif i == 6:
                    mid1, mid2 = mid_channel_c3[cnt_C3], mid_channel_bottleneck[3:6]
                elif i == 8:
                    mid1, mid2 = mid_channel_c3[cnt_C3], mid_channel_bottleneck[6]

                if i == 13:
                    mid1, mid2 = mid_channel_c3[cnt_C3], mid_channel_bottleneck[7]
                    c1, c2 = overall_channel[6] + overall_channel[10], overall_channel[11]
                elif i == 17:
                    mid1, mid2 = mid_channel_c3[cnt_C3], mid_channel_bottleneck[8]
                    c1, c2 = overall_channel[4] + overall_channel[12], overall_channel[13]
                    out17_channel = c2
                elif i == 20:
                    mid1, mid2 = mid_channel_c3[cnt_C3], mid_channel_bottleneck[9]
                    c1, c2 = overall_channel[12] + overall_channel[14], overall_channel[15]
                    out20_channel = c2
                elif i == 23:
                    mid1, mid2 = mid_channel_c3[cnt_C3], mid_channel_bottleneck[10]
                    c1, c2 = overall_channel[10] + overall_channel[16], overall_channel[17]
                    out23_channel = c2
                #
                # logger.info("mid_channel_c3: %s", mid1)
                # logger.info("mid_channel_bottleneck: %s", mid2)
                if i < 10:
                    args = [c1, c2, n, True, 1, 0.5] + [mid1, mid2]
                else:
                    args = [c1, c2, n, False, 1, 0.5] + [mid1, mid2]
                n = 1
                cnt_C3 += 1
            elif m == SPPF:
                c1, c2 = overall_channel[i - 1], overall_channel[i]
                mid = mid_channel_sppf[0]
                args = [c1, c2, 5, mid]
            else:
                c1, c2 = ch[f], args[0]
                # if c2 != no:  # if not output
                    # c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3TR, C3Ghost, C3x]:
                    args.insert(2, n)  # number of repeats
                    n = 1

        # 如果模块是批量归一化层（nn.BatchNorm2d），则参数设置为输入层的特征数
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            # print("================Detect=====================")
            # print([ch[x] for x in f])
            # print("===========================================")
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]


        # 构建模块（可能重复多次），并记录模块的类型、参数数量等信息。使用日志记录每层的详细信息。\
        # 如果 n（重复次数）大于1，则创建一个由重复的 m 模块组成的序列（nn.Sequential）。这意味着相同类型的层会根据 n 的值重复多次。
        # 如果 n 等于1，则直接创建一个 m 类型的模块。
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 提取模块 m 的类型名称作为字符串。它移除了字符串中的一些不必要的部分，比如模块名称前的 '__main__.'。
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 计算模块 m_ 的参数总数。numel() 函数用于获取每个参数的元素数量，然后将它们相加得到总数。
        np = sum(x.numel() for x in m_.parameters())  # number params
        # 这里给模块 m_ 附加了额外的信息，包括它在网络中的索引 i，它的“来源”索引 f（即输入来自哪一层），模块类型 t，以及参数数量 np。
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 更新保存列表（用于跟踪哪些层的输出需要被保存）和层列表（添加新创建的模块）
        # save 列表用于跟踪哪些层的输出需要被保存。这行代码将当前层的索引添加到保存列表中。
        # 将新创建的模块 m_ 添加到层列表 layers 中
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 如果是第一层（i == 0），则重置通道列表 ch。
        # 将输出通道数 c2 添加到通道列表中。
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    print("===========================================")
    print(ch)
    print("===========================================")

    # 将所有层组合成一个顺序模型并返回，同时返回排序后的保存列表
    return nn.Sequential(*layers), sorted(save)
```

## rank_generation

```python
def get_feature_hook(self, input, output):
    global feature_result
    global entropy
    global total
    a = output.shape[0]  # batchsize 有多少张图片
    b = output.shape[1]  # 通道数
    # 创建一个形状为(a*b,)的一维张量c，其中每个元素表示对应通道的输出矩阵的秩（rank）。
    # 对于每个通道，使用torch.matrix_rank函数计算输出矩阵的秩，并将其转换为Python数值。
    # output 是一个四维张量，i 表示批次中的图像索引，j 表示通道索引。
    # 每个通道的秩信息存储在一维张量 c 中，并将其视为一个形状为 (a * b,) 的一维张量，其中 a 是批次大小，b 是通道数
    c = torch.tensor([chi2_cal(output[i, j, :, :]) for i in range(a) for j in range(b)])
    # c 变形为形状为 (a, b) 的张量，并对第一个维度求和，以获得每个通道的秩的累积信息。这部分的目的是统计每个通道的秩，以便后续计算平均特征信息
    c = c.view(a, -1).float()#变成a*b的张量
    c = c.sum(0)#第几个维度变成1 就设置几 返回通道数大小即filter个数
    # feature_result 变量保存累积的特征信息，通过乘以 total 并加上 c 来累积每个通道的秩信息。然后，更新 total，该变量用于累积批次大小。
    # 更新feature_result，将其乘以total并加上c，以累积特征信息
    # feature_result 存储了之前所有批次的通道秩的累积信息。通过乘以 total，将之前的累积信息按照批次大小进行加权。然后，将当前批次的通道秩信息 c 加到上述加权的累积信息中。
    # 这种处理方式的好处是在处理连续的批次时，能够有效地累积和平均特征信息。通过乘以之前的总批次大小，并添加当前批次的信息，
    # 确保了之前的信息不会被遗忘，而是按照新的批次大小进行加权。这对于一些统计或特征分析任务是有帮助的，可以更好地反映模型在不同批次上的行为
    feature_result = feature_result * total + c
    # 累积批次大小
    # total 被更新为之前的 total 加上当前批次的大小。这是为了在下一次计算中使用新的批次大小。
    total = total + a#对每个通道求平均值
    # 计算每个通道的平均特征信息
    feature_result = feature_result / total
```

## adapt_channel

```python
n_num = [1, 2, 3, 1, 1, 1, 1, 1]
def adapt_channel(compress_rate):
    print("Now compress_rate", compress_rate)

    # backbone的输出通道数，第一个Conv模块的输出通道数为64，余下的Conv+C3组合的输出通道数依次为128, 256, 512, 1024
    # 此外，SPPF模块的输出通道数这里不体现
    # overall_out_channel = [64] + [128] * 2 + [256] * 2 + [512] * 2 + [1024] * 2
    # overall_out_channel = [32] + [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    overall_out_channel = [32] + [64] * 2 + [128] * 2 + [256] * 2 + [512] * 3 + [256] * 2 + [128] * 3 + [256] * 2 + [512]

    # 存储整体模块的输出通道数的压缩率
    overall_oup_cprate = []

    overall_channel = []

    # 这三部分压缩率用于获取C3模块的压缩率、SPPF模块的压缩率、Bottleneck模块的压缩率
    mid_cprate_c3 = []
    mid_cprate_sppf = []
    mid_cprate_bottleneck = []

    ori_mid_channel_c3 = [32, 64, 128, 256, 128, 64, 128, 256]
    ori_mid_channel_sppf = [256]
    ori_mid_channel_bottleneck = [32] + [64] * 2 + [128] * 3 + [256] + [128, 64, 128, 256]

    mid_channel_c3 = []
    mid_channel_sppf = []
    mid_channel_bottleneck = []

    # 解析compress_rate，构成依次是整体的压缩率、C3模块的压缩率、SPPF模块的压缩率、Bottleneck模块的压缩率
    # compress_rate依次是1+2*4，4*1，1，（1+2+3+1）*1个小数
    # 整体的压缩率包括第一个Conv模块的压缩率和余下的Conv+C3组合的压缩率
    # C3模块的压缩率只有一个压缩率，但是有4个C3模块
    # SPPF模块的压缩率只有一个压缩率，也只有1个SPPF模块
    # Bottleneck模块的压缩率有一个压缩率，但是有1+2+3+1个Bottleneck模块

    # 解析整体模块的压缩率
    idx = 0
    for i in range(len(overall_out_channel)):
        overall_oup_cprate.append(compress_rate[idx])
        idx += 1

    # 解析C3模块的压缩率
    c3_cprate = compress_rate[idx]
    for _ in n_num:
        mid_cprate_c3.append(c3_cprate)
        idx += 1

    # 解析SPPF模块的压缩率
    sppf_cprate = compress_rate[idx]
    mid_cprate_sppf.append(sppf_cprate)
    idx += 1

    # 解析Bottleneck模块的压缩率
    for _ in range(sum(n_num)):
        bottleneck_cprate = compress_rate[idx]
        mid_cprate_bottleneck.append(bottleneck_cprate)
        idx += 1

    # 计算调整后的总体通道数和中间层的通道数
    for i, cprate in enumerate(overall_oup_cprate):
        overall_channel.append(int(overall_out_channel[i] * (1 - cprate)))

    # i_cp = 1
    # for cprate in mid_cprate_c3:
    #     mid_channel_c3.append(int(overall_channel[i_cp] / 2 * (1 - cprate)))
    #     i_cp += 2
    #
    # for cprate in mid_cprate_sppf:
    #     mid_channel_sppf.append(int(overall_channel[-2] / 2 * (1 - cprate)))
    #
    # for i, cprate in enumerate(mid_cprate_bottleneck):
    #     if i < n_num[0]:
    #         mid_channel_bottleneck.append(int(overall_channel[0] * (1 - cprate)))
    #     elif i < sum(n_num[:2]):
    #         mid_channel_bottleneck.append(int(overall_channel[1] * (1 - cprate)))
    #     elif i < sum(n_num[:3]):
    #         mid_channel_bottleneck.append(int(overall_channel[3] * (1 - cprate)))
    #     else:
    #         mid_channel_bottleneck.append(int(overall_channel[5] * (1 - cprate)))

    # 假设这里是直接应用压缩率更新中间通道数
    mid_channel_c3 = [int(c * (1 - cprate)) for c, cprate in zip(ori_mid_channel_c3, mid_cprate_c3)]
    mid_channel_sppf = [int(c * (1 - cprate)) for c, cprate in zip(ori_mid_channel_sppf, mid_cprate_sppf)]
    # 对于bottleneck部分，如果处理逻辑可以简化，可以直接应用压缩率
    mid_channel_bottleneck = [int(c * (1 - cprate)) for c, cprate in zip(ori_mid_channel_bottleneck, mid_cprate_bottleneck)]

    logger.info("overall_channel: %s", overall_channel)
    logger.info("mid_channel_c3: %s", mid_channel_c3)
    logger.info("mid_channel_sppf: %s", mid_channel_sppf)
    logger.info("mid_channel_bottleneck: %s", mid_channel_bottleneck)
    # 返回计算结果
    return overall_channel, mid_channel_c3, mid_channel_sppf, mid_channel_bottleneck
```

