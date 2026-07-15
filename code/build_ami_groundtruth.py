import xml.etree.ElementTree as ET
import glob
import re

print("==== 正在解析 AMI 会议多角色 XML 标注 ====")

# 自动抓取目录下所有的 EN2001a 角色 words 文件
xml_files = glob.glob("../common_data/EN2001a.*.words.xml")

if not xml_files:
    print("❌ 未找到 XML 文件，请确认是否上传到了 common_data 目录！")
    exit()

all_words = []

for xml_file in xml_files:
    print(f"正在读取: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 遍历所有的单词标签 (AMI 格式通常是 <w starttime="..." ...>word</w>)
    for elem in root.iter('w'):
        if elem.text:
            # 提取时间戳（兼容 starttime 或 start 属性）
            start_time_str = elem.attrib.get('starttime') or elem.attrib.get('start')
            if start_time_str:
                start_time = float(start_time_str)
                word = elem.text.strip()
                # 过滤掉非字母的无效字符（有些标注可能带有符号）
                word = re.sub(r'[^\w\s\']', '', word)
                if word:
                    all_words.append((start_time, word))

# 核心逻辑：按照每个单词说出的精确时间（秒）进行全局排序！
all_words.sort(key=lambda x: x[0])

# 缝合成最终的一整段标准答案
full_transcript = " ".join([item[1] for item in all_words])

# 写入到一个 txt 文件中备用
output_path = "../common_data/EN2001a_ground_truth.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_transcript)

print(f"\n✅ 缝合成功！共提取了 {len(all_words)} 个单词。")
print(f"标准答案已保存至: {output_path}")