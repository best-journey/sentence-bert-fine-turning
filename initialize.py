import os  
import io  
import time

def split_file(big_file_path, small_file_pathes_path, chunk_size=1024*1024*20):  
    # 打开大文件  
    with open(big_file_path, 'rb') as f:  
        # 读取数据块  
        chunk = f.read(chunk_size)  
        # 创建临时文件名列表  
        temp_files = []  
        i = 1  
        while chunk:  
            # 创建临时文件并写入数据块  
            temp_file = f"{os.path.splitext(big_file_path)[0]}_{i}.tmp"  
            with open(temp_file, 'wb') as temp:  
                temp.write(chunk)  
            # 将临时文件名添加到列表中  
            temp_files.append(temp_file)  
            # 读取下一个数据块  
            chunk = f.read(chunk_size)  
            i += 1  
    
    # 存储临时文件名列表  
    with open(small_file_pathes_path, 'w', encoding='utf-8') as wfd :
        str_out = ""
        for path in temp_files:
            str_out += path + '\t'
            
        wfd.write(str_out[0:-1])
    
    # 删除大文件
    os.remove(big_file_path)

def merge_files(file_paths_path, output_path):  
    with open(file_paths_path, 'r', encoding='utf-8') as rfd:
        file_paths = rfd.readlines()[0].split('\t')
    
    # 打开输出文件  
    with open(output_path, 'wb') as out_file:  
        # 遍历所有小文件  
        for file_path in file_paths:  
            # 打开小文件  
            with open(file_path, 'rb') as in_file:  
                # 读取数据块并写入输出文件  
                chunk = in_file.read(4096)  
                while chunk:  
                    out_file.write(chunk)  
                    chunk = in_file.read(4096)  
    
    # 删除临时小文件
    for path in file_paths:
        time.sleep(0.5)
        os.remove(path)
    
    os.remove(file_paths_path) 
    
if __name__ == "__main__":
    # Part A: 切割、合并model_after_fine_turning\model_after_fine_turing.pt大文件
    # Part A: 01 这里我上传前已经执行完成，你直接运行执行下面的代码即可
    #split_file ("model_after_fine_turning\model_after_fine_turing.pt", "model_after_fine_turning\pathes.txt")
    #print("done")
    
    # Part A: 02 你直接运行执行这里的代码即可
    merge_files('model_after_fine_turning\pathes.txt', "model_after_fine_turning\model_after_fine_turing.pt")
    print("done")
    
    
    # Part B: 切割、合并model\pytorch_model.bin大文件
    # Part B: 01 这里我上传前已经执行完成，你直接运行执行下面的代码即可
    #split_file ("model\pytorch_model.bin", "model\pathes.txt")
    #print("done")
    
    # Part B: 02 你直接运行执行这里的代码即可
    merge_files('model\pathes.txt', "model\pytorch_model.bin")
    print("done")
