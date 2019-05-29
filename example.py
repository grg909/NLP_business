# coding: utf-8

# @Date    : 2019/5/29
# @Author  : WANG JINGE
# @Site    :
# @File    : example.py
# @Software: PyCharm
"""

"""

from seg_tool import SegTool
import pandas as pd
import pickle

# 输入dataframe处理
raw_data = pd.read_csv('data/train_11_880000.csv', encoding='utf8', index_col=0)
input_data = raw_data.fillna('')

# seg_and_rm_stopwords
print('seg_and_rm_stopwords: ')
wj = SegTool(input_data[:10000], data_name='行业分类数据', content_column_number=1)
with_class_list = wj.seg_and_rm_stopwords(
    seg_flags=['n', 'vn', 'an'], stopwords_relative_pos='lib/hlt_stop_words.txt')
print(with_class_list)
print('-'*50)

# 直接读取上次分词结果
fid_segword_result = pickle.load(open('data/行业分类数据.pkl', 'rb'))
words_df = pd.DataFrame(fid_segword_result, columns=['FID', 'words'])
print(words_df)
print('-'*50)

# merge_with_raw_data
print('merge_with_raw_data: ')
total_data = wj.merge_with_raw_data(
    seg_flags=['n', 'vn', 'an'], stopwords_relative_pos='lib/hlt_stop_words.txt')
print(total_data)
print('-'*50)

# iter_seg_and_rm_stopwords
print('iter_seg_and_rm_stopwords: ')
word_result_list = []
stopwords = wj._read_stopwords('lib/hlt_stop_words.txt')
for i in wj.iter_seg_and_rm_stopwords(['n', 'vn', 'an'], stopwords):
    word_result_list.append(i)
print(word_result_list)
