# TracHard

步骤有些繁琐
PS:有些地方写死了，复用性不够好，代码中提取50%的数据，另外并没有shuffle的方式，避免了图像号码可能错位或者对不上。基本都是在原train_loader下操作提取。

第一步：正常训练保存检查点，order_examples.py统计出那些一直错误的数据；
第二步：forget_pic.py从test_loader中提取出第一步中找到的hard sample，方便下一步计算tracin值；
第三步：rank_tracin.py把每个训练数据与提取出来的数据计算tracin（同类的才计算），找出每一类数据的top,由于这里只记录了是某类的第几个，所以还要对编号进行一次转化，从而确定是数据集中的第几个；
第四步：num_conversion.py把每一类的top数据从train_loader中提取出来方便训练；
第五步：将找到的数据进行训练。
