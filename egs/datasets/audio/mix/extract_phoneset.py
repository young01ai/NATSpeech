import xml.etree.ElementTree as ET

# 读取并解析XML文件
tree = ET.parse('phoneset_mixlingual.xml')
root = tree.getroot()

# 定义命名空间
namespace = {'ns': 'http://schemas.microsoft.com/tts'}

# 遍历所有的phone元素
for phone in root.findall('ns:phone', namespace):
    name = phone.get('name')
    features = phone.find('ns:feature', namespace).text
    print(f"Phone name: {name}, Features: {features}")
