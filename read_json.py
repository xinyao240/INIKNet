import json
with open('model_config.json','r') as file:
    str = file.read()
    data = json.loads(str)
    print(data)
    print(type(data))#python列表的类型
