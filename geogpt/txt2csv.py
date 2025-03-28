import os
import csv

# 根目录和文本目录的路径
root_directory = 'D:/code/geogpt'
text_directory = 'D:/code/geogpt/文本'

# 创建保存csv文件的文件夹
output_directory = os.path.join(root_directory, 'data512')
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 遍历txt文件夹中的所有文件
for filename in os.listdir(text_directory):
    if filename.endswith('.txt'):
        # 读取txt文件内容
        with open(os.path.join(text_directory, filename), 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按指定数量的汉字切分文本
        # chunk_size = 999999999999999
        chunk_size = 512
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        if len(content) % chunk_size != 0:
            chunks.append(content[-(len(content) % chunk_size):])
        
        # 创建csv文件并写入内容
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        with open(os.path.join(output_directory, csv_filename), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['content'])
            for chunk in chunks[:-1]:
                writer.writerow([chunk])
        
        print(f'{csv_filename} 已保存')