# coding: utf-8

# @Time    : 2019/05/29
# @Author  : WANG JINGE
# @Site    :
# @File    : seg_tool.py
# @Software: PyCharm
"""
    此类提供基本的分词功能
"""

import pandas as pd
from jieba import posseg as pseg
import jieba
import pickle
from tqdm import tqdm


jieba.setLogLevel(20)
try:
    jieba.enable_parallel()
    print('启动多进程加速。。。')
except:
    print('不支持多进程加速，跳过。。。')


class SegTool:

    def __init__(
            self,
            data,
            data_name,
            content_column_number):
        """
        传入dataframe
        :param data: 数据pandas.dataframe, 第一列必须为源数据的FID（便于分词结果与源数据合并）
        :param data_name: 数据集名称
        :param content_column: 待分词的那一列的列号（不计index）
        """
        self.data = data
        self.data_name = data_name
        self.content_column_number = content_column_number

    def _iter_segment(self, flags):
        """
        分词生成器，保留指定flags列表中的分词
        :param flags: 指定保留的分词flags列表
        :return: 每次返回一行FID加分词的结果，['53', '小明‘，’来自‘，‘成都’, '啊']
        """
        for row in tqdm(self.data.itertuples(), total=self.data.shape[0]):
            seg_list = []
            description_seg = pseg.cut(row[self.content_column_number].replace('\n', ''))
            words_seg = [
                word for word,
                flag in description_seg if flag in flags]
            seg_list.append(row[0])
            seg_list.extend(words_seg)
            yield seg_list

    @staticmethod
    def _remove_stopwords(seg_list, stopwords):
        """
        去除输入分词中的停用词。生成器
        :param seg_list: 分词结果字符串
        :param stopwords_line: 停用词列表
        :return: 每次返回一个经过筛选后的分词
        """
        for word in seg_list:
            if word not in stopwords:
                yield word

    @staticmethod
    def _read_stopwords(stopwords_relative_pos):
        """
        读取停用词表（修改停用词表不可用windows自带notepad打开，否则编码异常）
        :param stopwords_relative_pos: 停用词路径
        :return:
        """
        try:
            with open(stopwords_relative_pos, encoding='utf-8') as sp:
                stopwords = sp.read()
        except Exception as e:
            print('请确保停用词库在正确目录下')
            raise e

        return stopwords

    def seg_and_rm_stopwords(self, seg_flags, stopwords_relative_pos, enable_pickle=True):
        """
        分词和去除停用词
        :param seg_flags: 指定保留的分词flags列表
        :param stopwords_relative_pos: 停用词相对路径位置
        :return: 返回fid和分词结果组成的dataframe
        """
        stopwords = self._read_stopwords(stopwords_relative_pos)

        fid_segword_result = []
        seg_list = self._iter_segment(seg_flags)
        for seg_raw in seg_list:
            fid, seg_words = seg_raw[0], seg_raw[1:]
            seg_rm = [i for i in self._remove_stopwords(seg_words, stopwords)]
            fid_segword_result.append((fid, ' '.join(seg_rm)))

        if enable_pickle:
            try:
                pickle.dump(fid_segword_result, open('data/{}.pkl'.format(self.data_name), 'wb'))
            except:
                print('请在主目录为保存缓存文件新建一个data文件夹')

        words_df = pd.DataFrame(fid_segword_result, columns=['FID', 'words'])

        return words_df

    def merge_with_raw_data(self, seg_flags, stopwords_relative_pos):
        """
        和源数据合并的dataframe
        :param seg_flags: 指定保留的分词flags列表
        :param stopwords_relative_pos: 停用词相对路径位置
        :return: 返回合并的dataframe
        """
        seg_words = self.seg_and_rm_stopwords(seg_flags, stopwords_relative_pos)
        merge_data = self.data.join(seg_words.set_index('FID'))

        return merge_data

    def iter_seg_and_rm_stopwords(self, seg_flags, stopwords):
        """
        分词和去除停用词，生成器版本（选用）
        :param seg_flags: 指定保留的分词flags列表
        :param stopwords: 停用词相对路径位置
        :return: 每次返回一个fid和分词结果的元祖
        """
        seg_list = self._iter_segment(seg_flags)
        for seg_raw in seg_list:
            fid, seg_words = seg_raw[0], seg_raw[1:]
            seg_rm = [i for i in self._remove_stopwords(seg_words, stopwords)]
            yield (fid, ' '.join(seg_rm))
