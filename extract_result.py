import os
import re

if __name__ == '__main__':
    folder_path = 'logs/minghaocil/omnibenchmark/0/30'
    pattern = r'minghaocil_lr_(\d+\.\d+)_wd_(\d+\.\d+)_opt_sgd_vt_deep_loss_cross_entropy.log'
    text_pattern = r"Loss (\d+\.\d+), Train_accy (\d+\.\d+), Test_accy (\d+\.\d+)"
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        match=re.match(pattern, file_name)
        if match:
            lr= match.group(1)
            wd= match.group(2)
            print("file_name:", file_name)
            # read the file
            with open(file_path, 'r') as f:
                text = f.read()
            matches = re.search(text_pattern, text)
            if matches:
                loss = matches.group(1)
                train_accy = matches.group(2)
                test_accy = matches.group(3)
                print("Loss:", loss)
                print("Train_accy:", train_accy)
                print("Test_accy:", test_accy)
                
                result = [lr, wd, loss, train_accy, test_accy]
                
            else:
                print("wrong file")
                import pdb; pdb.set_trace()
            
        