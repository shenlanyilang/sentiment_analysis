# -*- coding:utf-8 -*-
from opencc import OpenCC
import json

OP = OpenCC('t2s')


def fan2jian(content):
    return OP.convert(content)


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
