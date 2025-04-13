import sys
import getopt
dataAugmentation = False
unbalancedDataset = False
 
try:
    opts, args = getopt.getopt(sys.argv[1:], "auh",  
                              ["augmentation",
                               "unbalanced-dataset",
                               "help"])  # 长选项模式
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("用法: script.py [-a] [-u <value>]")
            sys.exit()
        elif opt in ('-a', '--augmentation'):
            dataAugmentation = True
        elif opt in ('-u', '--unbalanced-dataset'):
            unbalancedDataset = True
    
except:
    print("Usage: train.py -a [--augmentation] -u [--unbalanced-dataset]")
    sys.exit(1)

print(f'Data Augmentation: {"On" if dataAugmentation else "Off"}')
print(f'Unbalanced Dataset: {"On" if unbalancedDataset else "Off"}')