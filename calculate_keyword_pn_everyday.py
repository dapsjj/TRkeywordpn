# -*- coding: UTF-8 -*-
import re
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import MeCab
import random
import numpy as np
import pymssql
import collections
import datetime
import logging
import os
import configparser
import decimal #不加打包成exe会出错


'''
参考ページ
http://www.statsbeginner.net/entry/2017/05/07/091435
'''

IGNORE_WORDS = set([])  # 重要度計算外とする語
no_need_words = ["これ","ここ","こと","それ","ため","よう","さん","そこ","たち","ところ","それぞれ","これら","どれ","br"]

# ひらがな
JP_HIRA = set([chr(i) for i in range(12353, 12436)])
# カタカナ
JP_KATA = set([chr(i) for i in range(12449, 12532+1)])
#要忽略的字符
# "ー"特殊
MULTIBYTE_MARK = set([
    '、', ',', '，', '。', '．','\'', '”', '“', '《', '》', '：', '（', '）', '(', ')', '；', '.', '・', '～', '`',
    '%', '％', '$', '￥', '~', '■', '●', '◆', '×', '※', '►', '▲', '▼', '‣', '·', '∶', ':', '‐', '_', '‼', '≫',
    '－','−', ';', '･', '〈', '〉', '「', '」', '『', '』', '【', '】', '〔', '〕', '?', '？', '!', '！', '+', '-',
    '*', '÷', '±', '…', '‘', '’', '／', '/', '<', '>', '><', '[', ']', '#', '＃', '゛', '゜',
    # '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # '０','１', '２', '３', '４', '５', '６', '７', '８', '９',
    '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨',
    '⑩', '⑪', '⑫', '⑬', '⑭', '⑮', '⑯', '⑰', '⑱', '⑲', '⑳',
    '➀', '➁', '➂', '➃', '➄', '➅', '➆', '➇', '➈', '➉',
    '⑴', '⑵', '⑶', '⑷', '⑸', '⑹', '⑺', '⑻', '⑼', '⑽',
    '⑾', '⑿', '⒀', '⒁', '⒂', '⒃', '⒄', '⒅', '⒆', '⒇',
    '⒈', '⒉', '⒊', '⒋', '⒌', '⒍', '⒎', '⒏', '⒐', '⒑',
    '⒒', '⒓', '⒔', '⒕', '⒖', '⒗', '⒘', '⒙', '⒚', '⒛',
    'ⅰ', 'ⅱ', 'ⅲ', 'ⅳ', 'ⅴ', 'ⅵ', 'ⅶ', 'ⅷ', 'ⅸ', 'ⅹ',
    'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ', 'Ⅵ', 'Ⅶ', 'Ⅷ', 'Ⅸ', 'Ⅹ',
    'Ⅺ', 'Ⅻ', '❶', '❷', '❸', '❹', '❺', '❻', '❼', '❽', '❾', '❿',
    '⓫', '⓬', '⓭', '⓮', '⓯', '⓰', '⓱', '⓲', '⓳', '⓴',
    '㈠', '㈡', '㈢', '㈣', '㈤', '㈥', '㈦', '㈧', '㈨', '㈩',
    '㊀', '㊁', '㊂', '㊃', '㊄', '㊅', '㊆', '㊇', '㊈', '㊉',
    'Ⓐ', 'Ⓑ', 'Ⓒ', 'Ⓓ', 'Ⓔ', 'Ⓕ', 'Ⓖ', 'Ⓗ', 'Ⓘ', 'Ⓙ',
    'Ⓚ', 'Ⓛ', 'Ⓜ', 'Ⓝ', 'Ⓞ', 'Ⓟ', 'Ⓠ', 'Ⓡ', 'Ⓢ', 'Ⓣ',
    'Ⓤ', 'Ⓥ', 'Ⓦ', 'Ⓧ', 'Ⓨ', 'Ⓩ', 'ⓐ', 'ⓑ', 'ⓒ', 'ⓓ',
    'ⓔ', 'ⓕ', 'ⓖ', 'ⓗ', 'ⓘ', 'ⓙ', 'ⓚ', 'ⓛ', 'ⓜ', 'ⓝ',
    'ⓞ', 'ⓟ', 'ⓠ', 'ⓡ', 'ⓢ', 'ⓣ', 'ⓤ', 'ⓥ', 'ⓦ', 'ⓧ',
    'ⓨ', 'ⓩ', '⒜', '⒝', '⒞', '⒟', '⒠', '⒡', '⒢', '⒣',
    '⒤', '⒥', '⒦', '⒧', '⒨', '⒩', '⒪', '⒫', '⒬', '⒭',
    '⒮', '⒯', '⒰', '⒱', '⒲', '⒳', '⒴', '⒵',
    '\r\n', '\t', '\n', '\\',
    '◇', '＜', '＞', '＊', '＝', '◍', '＋', '○', '―', 'ˇ', 'ˉ',
    '¨', '〃', '—', '‖', '∧', '∨', '∑', '∏', '∪', '∩', '∈',
    '∷', '√', '⊥', '∥', '∠', '⌒', '⊙', '∫', '∮', '≡', '≌',
    '≈', '∽', '∝', '≠', '≮', '≯', '≤', '≥', '∞', '∵', '∴',
    '♂', '♀', '°', '′', '″', '℃', '＄', '¤', '￠', '￡', '‰',
    '§', '№', '☆', '★', '□', '〓', '〜', '⬜', '〇', '＿',
    '▢', '∟', '⇒', '◯', '△', '✕', '＆', '|', '＠', '@', '&',
    '〖', '〗', '◎', '〒', '℉', '﹪', '﹫', '㎡', '㏕', '㎜',
    '㎝', '㎞', '㏎', 'm', '㎎', '㎏', '㏄', 'º', '¹', '²', '³',
    '↑', '↓', '←', '→', '↖', '↗', '↘', '↙', '↔', '↕', '➻', '➼',
    '➽', '➸', '➳', '➺', '➴', '➵', '➶', '➷', '➹', '▶', '▷',
    '◁', '◀', '◄', '«', '»', '➩', '➪', '➫', '➬', '➭', '➮',
    '➯', '➱', '⏎', '➲', '➾', '➔', '➘', '➙', '➚', '➛', '➜',
    '➝', '➞', '➟', '➠', '➡', '➢', '➣', '➤', '➥', '➦', '➧',
    '➨', '↚', '↛', '↜', '↝', '↞', '↟', '↠', '↡', '↢', '↣', '↤', '↥',
    '↦', '↧', '↨', '⇄', '⇅', '⇆', '⇇', '⇈', '⇉', '⇊', '⇋', '⇌', '⇍',
    '⇎', '⇏', '⇐', '⇑', '⇓', '⇔', '⇖', '⇗', '⇘', '⇙', '⇜', '↩', '↪',
    '↫', '↬', '↭', '↮', '↯', '↰', '↱', '↲', '↳', '↴', '↵', '↶', '↷',
    '↸', '↹', '☇', '☈', '↼', '↽', '↾', '↿', '⇀', '⇁', '⇂', '⇃', '⇞',
    '⇟', '⇠', '⇡', '⇢', '⇣', '⇤', '⇥', '⇦', '⇧', '⇨', '⇩', '⇪', '↺',
    '↻', '⇚', '⇛', '♐', '┌', '┍', '┎', '┏', '┐', '┑', '┒', '┓', '└', '┕',
    '┖', '┗', '┘', '┙', '┚', '┛', '├', '┝', '┞', '┟', '┠', '┡', '┢', '┣',
    '┤', '┥', '┦', '┧', '┨', '┩', '┪', '┫', '┬', '┭', '┮', '┯', '┰', '┱',
    '┲', '┳', '┴', '┵', '┶', '┷', '┸', '┹', '┺', '┻', '┼', '┽', '┾', '┿',
    '╀', '╁', '╂', '╃', '╄', '╅', '╆', '╇', '╈', '╉', '╊', '╋', '╌', '╍',
    '╎', '╏', '═', '║', '╒', '╓', '╔', '╕', '╖', '╗', '╘', '╙', '╚', '╛',
    '╜', '╝', '╞', '╟', '╠', '╡', '╢', '╣', '╤', '╥', '╦', '╧', '╨', '╩',
    '╪', '╫', '╬', '◤', '◥', '◣', '◢', '▸', '◂', '▴', '▾', '▽', '⊿', '▻',
    '◅', '▵', '▿', '▹', '◃', '❏', '❐', '❑', '❒', '▀', '▁', '▂', '▃', '▄',
    '▅', '▆', '▇', '▉', '▊', '▋', '█', '▌', '▍', '▎', '▏', '▐', '░', '▒', '▓',
    '▔', '▕', '▣', '▤', '▥', '▦', '▧', '▨', '▩', '▪', '▫', '▬', '▭', '▮', '▯',
    '㋀', '㋁', '㋂', '㋃', '㋄', '㋅', '㋆', '㋇', '㋈', '㋉', '㋊', '㋋',
    '㏠', '㏡', '㏢', '㏣', '㏤', '㏥', '㏦', '㏧', '㏨', '㏩', '㏪', '㏫',
    '㏬', '㏭', '㏮', '㏯', '㏰', '㏱', '㏲', '㏳', '㏴', '㏵', '㏶', '㏷',
    '㏸', '㏹', '㏺', '㏻', '㏼', '㏽', '㏾', '㍙', '㍚', '㍛', '㍜', '㍝',
    '㍞', '㍟', '㍠', '㍡', '㍢', '㍣', '㍤', '㍥', '㍦', '㍧', '㍨', '㍩',
    '㍪', '㍫', '㍬', '㍭', '㍮', '㍯', '㍰', '㍘', '☰', '☲', '☱', '☴',
    '☵', '☶', '☳', '☷', '☯', '♠', '♣', '♧', '♡', '♥', '❤', '❥', '❣',
    '✲', '☀', '☼', '☾', '☽', '◐', '◑', '☺', '☻', '☎', '☏', '✿', '❀',
    '¿', '½', '✡', '㍿', '卍', '卐', '✚', '♪', '♫', '♩', '♬', '㊚', '㊛',
    '囍', '㊒', '㊖', 'Φ', 'Ψ', '♭', '♯', '♮', '¶', '€', '¥', '﹢', '﹣',
    '=', '≦', '≧', '≒', '﹤', '﹥', '㏒', '㏑', '⅟', '⅓', '⅕', '⅙',
    '⅛', '⅔', '⅖', '⅚', '⅜', '¾', '⅗', '⅝', '⅞', '⅘', '≂', '≃', '≄',
    '≅', '≆', '≇', '≉', '≊', '≋', '≍', '≎', '≏', '≐', '≑', '≓', '≔',
    '≕', '≖', '≗', '≘', '≙', '≚', '≛', '≜', '≝', '≞', '≟', '≢', '≣',
    '≨', '≩', '⊰', '⊱', '⋛', '⋚', '∬', '∭', '∯', '∰', '∱', '∲', '∳',
    '℅', '‱', 'ø', 'Ø', 'π', 'ღ', '♤', '＇', '〝', '〞', 'ˆ', '﹕', '︰',
    '﹔', '﹖', '﹑', '•', '¸', '´', '｜', '＂', '｀', '¡', '﹏', '﹋',
    '﹌', '︴', '﹟', '﹩', '﹠', '﹡', '﹦', '￣', '¯', '﹨', '˜', '﹍', '﹎',
    '﹉', '﹊', '‹', '›', '﹛', '﹜', '［', '］', '{', '}', '︵', '︷', '︿',
    '︹', '︽', '﹁', '﹃', '︻', '︶', '︸', '﹀', '︺', '︾', '﹂', '﹄',
    '︼', '❝', '❞', '£', 'Ұ', '₴', '₰', '¢', '₤', '₳', '₲', '₪', '₵',
    '₣', '₱', '฿', '₡', '₮', '₭', '₩', 'ރ', '₢', '₥', '₫', '₦',
    'z', 'ł', '﷼', '₠', '₧', '₯', '₨', 'K', 'č', 'र', '₹', 'ƒ', '₸',
    '✐', '✎', '✏', '✑', '✒', '✍', '✉', '✁', '✂', '✃', '✄', '✆',
    '☑', '✓', '✔', '☐', '☒', '✗', '✘', 'ㄨ', '✖', '☢', '☠', '☣', '✈',
    '☜', '☞', '☝', '☚', '☛', '☟', '✌', '♢', '♦', '☁', '☂', '❄', '☃',
    '♨', '웃', '유', '❖', '☪', '✪', '✯', '☭', '✙', '⚘', '♔', '♕', '♖',
    '♗', '♘', '♙', '♚', '♛', '♜', '♝', '♞', '♟', '◊', '◦', '◘', '◈', 'の',
    'Ю', '❈', '✣', '✤', '✥', '✦', '❉', '❦', '❧', '❃', '❂', '❁', '☄', '☊',
    '☋', '☌', '☍', '۰', '⊕', 'Θ', '㊣', '◙', '♈', '큐', '™', '◕', '‿', '｡'
    # "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
    # "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    ])


def get_diclist(text):
    try:
        savetxt_list = []
        mecab = MeCab.Tagger("-Ochasen")  # 有词性标注的
        cmp_nouns = mecab.parse(text)
        every_row = cmp_nouns.split('\n')
        save_word_list = []
        for every_attribute_line in every_row:
            every_attribute_array = every_attribute_line.split('\t')
            if len(every_attribute_array) > 3:
                save_word_list.append([every_attribute_array[0].strip(), every_attribute_array[3].strip(),every_attribute_array[2].strip()])#every_attribute_array[2]は語の辞書形
        length_save_word_list = len(save_word_list)
        for i in range(length_save_word_list - 4):
            if i == 0:
                if save_word_list[i][1].find('名詞') != -1 and save_word_list[i][1].find('名詞-数') == -1 \
                        and save_word_list[i][0] not in MULTIBYTE_MARK:
                    savetxt_list.append([save_word_list[i][0],save_word_list[i][2]])  # 保存名词,字典形
                elif save_word_list[i][1].find('名詞-数') != -1 and save_word_list[i + 1][1].find('名詞-数') != -1 \
                        and save_word_list[i + 2][0] not in MULTIBYTE_MARK and save_word_list[i + 2][1].find('名詞') != -1 \
                        and save_word_list[i + 2][1].find('名詞-数') == -1:
                    savetxt_list.append(
                        [save_word_list[i][0] + save_word_list[i + 1][0] + save_word_list[i + 2][0],save_word_list[i][2]+save_word_list[i + 1][2]+save_word_list[i + 2][2]])  # 保存数词+数词+名词,字典形
                elif save_word_list[i][1].find('名詞-数') != -1 and save_word_list[i + 1][1].find('名詞-数') != -1 \
                        and save_word_list[i + 2][1].find('名詞-数') != -1 and save_word_list[i + 3][0] not in MULTIBYTE_MARK \
                        and save_word_list[i + 3][1].find('名詞') != -1 and save_word_list[i + 3][1].find('名詞-数') == -1:
                    savetxt_list.append(
                        [save_word_list[i][0] + save_word_list[i + 1][0] + save_word_list[i + 2][0] + save_word_list[i + 3][0], save_word_list[i][2] + save_word_list[i + 1][2] + save_word_list[i + 2][2] + save_word_list[i + 3][2]])  # 保存数词+数词+数词+名词,字典形
                elif save_word_list[i][1].find('名詞-数') != -1 and save_word_list[i + 1][0] not in MULTIBYTE_MARK \
                        and save_word_list[i + 1][1].find('名詞') != -1 and save_word_list[i + 1][1].find('名詞-数') == -1:
                    savetxt_list.append([save_word_list[i][0] + save_word_list[i + 1][0], save_word_list[i][2] + save_word_list[i + 1][2]])  # 保存数词+名词,字典形

            if i > 0:
                if save_word_list[i][1].find('名詞') != -1 and save_word_list[i][1].find('名詞-数') == -1 \
                        and save_word_list[i - 1][1].find('名詞-数') == -1 and save_word_list[i][0] not in MULTIBYTE_MARK:
                    savetxt_list.append([save_word_list[i][0], save_word_list[i][2]])  # 保存名词,字典形
                elif save_word_list[i][1].find('名詞-数') != -1 and save_word_list[i + 1][1].find('名詞-数') != -1 \
                        and save_word_list[i + 2][0] not in MULTIBYTE_MARK and save_word_list[i + 2][1].find('名詞') != -1 \
                        and save_word_list[i + 2][1].find('名詞-数') == -1 and save_word_list[i - 1][1].find('名詞-数') == -1:
                    savetxt_list.append(
                        [save_word_list[i][0] + save_word_list[i + 1][0] + save_word_list[i + 2][0], save_word_list[i][2] + save_word_list[i + 1][2] + save_word_list[i + 2][2]])  # 保存数词+数词+名词,字典形
                elif save_word_list[i][1].find('名詞-数') != -1 and save_word_list[i + 1][1].find('名詞-数') != -1 \
                        and save_word_list[i + 2][1].find('名詞-数') != -1 and save_word_list[i + 3][0] not in MULTIBYTE_MARK \
                        and save_word_list[i + 3][1].find('名詞') != -1 and save_word_list[i + 3][1].find('名詞-数') == -1 \
                        and save_word_list[i - 1][1].find('名詞-数') == -1:
                    savetxt_list.append(
                        [save_word_list[i][0] + save_word_list[i + 1][0] + save_word_list[i + 2][0] + save_word_list[i + 3][0], save_word_list[i][2] + save_word_list[i + 1][2] + save_word_list[i + 2][2] + save_word_list[i + 3][2]])  # 保存数词+数词+数词+名词,字典形
                elif save_word_list[i][1].find('名詞-数') != -1 and save_word_list[i + 1][1].find('名詞-数') == -1 \
                        and save_word_list[i + 1][0] not in MULTIBYTE_MARK and save_word_list[i + 1][1].find('名詞') != -1 \
                        and save_word_list[i + 1][1].find('名詞-数') == -1 and save_word_list[i - 1][1].find('名詞-数') == -1:
                    savetxt_list.append([save_word_list[i][0] + save_word_list[i + 1][0], save_word_list[i][2] + save_word_list[i + 1][2]])  # 保存数词+名词,字典形

                '''
                #保存数词
                elif save_word_list[i][1].find('名詞-数') != -1 and save_word_list[i + 1][1].find('名詞-数') != -1 \
                    and save_word_list[i + 2][1].find('名詞') == -1 and save_word_list[i-1][1].find('名詞-数') == -1:
                    savetxt_list.append(save_word_list[i][0] + save_word_list[i + 1][0])#保存数词+数词
                elif save_word_list[i][1].find('名詞-数') != -1 and save_word_list[i + 1][1].find('名詞-数') != -1 \
                    and save_word_list[i + 2][1].find('名詞-数') != -1 and save_word_list[i + 3][1].find('名詞') == -1\
                    and save_word_list[i - 1][1].find('名詞-数') == -1:
                    savetxt_list.append(save_word_list[i][0] + save_word_list[i + 1][0] + save_word_list[i + 2][0])#保存数词+数词+数词
                elif save_word_list[i][1].find('名詞-数') != -1 and save_word_list[i+1][1].find('名詞') == -1\
                    and save_word_list[i-1][1].find('名詞-数') == -1:
                    savetxt_list.append(save_word_list[i][0])#保存数词
                '''

        # savetxt_list = [' '.join(i) for i in savetxt_list]  # 不加这一句,重要度就是频率

        new_txt_list = []
        for every_word in savetxt_list:  # 每个字符都不在特殊符号里并且不是数字的词语添加到new_txt_list
            append_flag = True
            if (every_word[0] is not None and len(every_word[0].strip()) > 1 and not (every_word[0].strip().isdigit())):
                for i in every_word[0]:
                    if i in MULTIBYTE_MARK:
                        append_flag = False
                        break
                if append_flag == True:
                    new_txt_list.append(every_word)

        new_txt_list2 = []
        for every_word in new_txt_list:  # 不包含no_need_words的词加入到new_txt_list2
            find_flag = False
            for word in no_need_words:
                if every_word[0].find(word) != -1:
                    find_flag = True
                    break
            if find_flag == False:
                new_txt_list2.append(every_word)

        new_txt_list3 = []
        for every_word in new_txt_list2:  # 去掉0和片假名长音'ー'开头的字符串
            if not every_word[0].startswith('0') and not every_word[0].startswith('０') and not every_word[0].startswith('ー'):
                new_txt_list3.append(every_word)
        cmp_nouns = new_txt_list3

        diclist = []
        for word in cmp_nouns:
            d = {'Surface':word[0], 'BaseForm':word[1]}
            diclist.append(d)
        return diclist

    except Exception as ex:
        logger.error("Call method get_diclist() has error.")
        logger.error("Exception:" + str(ex))
        raise ex


def add_pnvalue(diclist_old):
    diclist_new = []
    for word in diclist_old:
        base = word['BaseForm']        # 個々の辞書から基本形を取得
        if base in pn_dict:
            pn = float(pn_dict[base])  # 中身の型があれなので
        else:
            pn = 'null'            # その語がPN Tableになかった場合
        word['PN'] = pn
        diclist_new.append(word)
    return diclist_new


def write_log():
    '''
    :return: 返回logger对象
    '''
    # 获取logger实例，如果参数为空则返回root logger
    logger = logging.getLogger()
    now_date = datetime.datetime.now().strftime('%Y%m%d')
    log_file = now_date+".log"# 文件日志
    if not os.path.exists("log"):#python文件同级别创建log文件夹
        os.makedirs("log")
    # 指定logger输出格式
    formatter = logging.Formatter('%(asctime)s %(levelname)s line:%(lineno)s %(message)s')
    file_handler = logging.FileHandler("log" + os.sep + log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter) # 可以通过setFormatter指定输出格式
    # 为logger添加的日志处理器，可以自定义日志处理器让其输出到其他地方
    logger.addHandler(file_handler)
    # 指定日志的最低输出级别，默认为WARN级别
    logger.setLevel(logging.INFO)
    return logger


def read_dateConfig_file_set_database():
    if os.path.exists(os.path.join(os.path.dirname(__file__), "dateConfig.ini")):
        try:
            conf = configparser.ConfigParser()
            conf.read(os.path.join(os.path.normpath(os.path.dirname(__file__)), "dateConfig.ini"), encoding="utf-8-sig")
            server = conf.get("server", "server")
            user = conf.get("user", "user")
            password = conf.get("password", "password")
            database = conf.get("database", "database")
            return server,user,password,database
        except Exception as ex:
            logger.error("Content in dateConfig.ini about database has error.")
            logger.error("Exception:" + str(ex))
            raise ex


def get_year_week_from_Mst_date(server, user, password, database, current_date):
    '''
    :param server:服务器名称
    :param user:用户名
    :param password:密码
    :param database:数据库名
    :param current_date:系统当前日期年-月-日
    :return:Mst_date表返回的当前年和当前周
    '''
    try:
        conn = pymssql.connect(server, user, password, database)
        cur = conn.cursor()
        sql = " select year_no,week_no from Mst_date where date_mst='%s' "  % current_date
        cur.execute(sql)
        row = cur.fetchone()
        if row:
            current_year = row[0]
            current_week = row[1]
            return current_year,current_week
        else:
            return ""
    except pymssql.Error as ex:
        logger.error("dbException:" + str(ex))
        raise ex
    except Exception as ex:
        logger.error("Call method get_year_week_from_Mst_date() error!Can not query from table Mst_date!")
        logger.error("Exception:" + str(ex))
        raise ex
    finally:
        conn.close()


def read_dateConfig_file_set_year_week():
    global report_year
    global report_week
    if os.path.exists(os.path.join(os.path.dirname(__file__), "dateConfig.ini")):
        try:
            conf = configparser.ConfigParser()
            conf.read(os.path.join(os.path.dirname(__file__), "dateConfig.ini"), encoding="utf-8-sig")
            year = conf.get("execute_year", "year")
            week = conf.get("execute_week", "week")
            if  year:
                report_year = year
            if week:
                report_week = week
        except Exception as ex:
            logger.error("Content in dateConfig.ini about execute_year or execute_week has error.")
            logger.error("Exception:" + str(ex))
            raise ex


def get_report_employee_list(server, user, password, database, report_year,report_week):
    try:
        conn = pymssql.connect(server, user, password, database)
        cur = conn.cursor()
        sql = " select convert(int,employee_code) as employee_code,report_year,report_week,remark from report " \
              " where report_year = %s and report_week = %s " \
             % (report_year,report_week)
        cur.execute(sql)
        rows = cur.fetchall()
        if rows:
            employee_list = [list(row) for row in rows]
            return employee_list
        else:
            return ""
    except pymssql.Error as ex:
        logger.error("dbException:" + str(ex))
        raise ex
    except Exception as ex:
        logger.error("Call method get_report_employee_list() error!")
        logger.error("Exception:" + str(ex))
        raise ex
    finally:
        conn.close()

def generate_report_keyword_pn_list(data_list):
    '''
    :param data_list:要处理哪些年哪些周哪些人的list
    :return:data_to_report_keyword_pn_list:要插入report_keyword_pn表的数据
    '''
    if data_list:
        try:
            array_report_list = np.array(data_list)
            employee_list = array_report_list[:, 0]
            year_list = array_report_list[:, 1]
            week_list = array_report_list[:, 2]
            report_list = array_report_list[:, 3]
            data_to_report_keyword_pn_list = []#要插入的数据
            i = 0
            for report in report_list:
                dl_old = get_diclist(report)
                dl_new = add_pnvalue(dl_old)
                dl_result = [dict(k + (('frequency', v),)) for k, v in Counter(tuple(k.items()) for k in dl_new).items()]
                for item in dl_result:
                    data_to_report_keyword_pn_list.append([year_list[i],week_list[i],employee_list[i],item['Surface'],item['PN'],item['frequency']])
                i += 1
            return data_to_report_keyword_pn_list
        except Exception as ex:
            logger.error("Call method generate_report_keyword_pn_list() error!")
            logger.error("Exception:" + str(ex))
            raise ex
    else:
        logger.error("Call method generate_report_keyword_pn_list() error!The data_list is empty.")
        raise

def get_report_pn_dictionary_list(server, user, password, database):
    '''
    :param server:服务器名称
    :param user:用户名
    :param password:密码
    :param database:数据库名
    :return:从数据库返回的数据加工成字典
    '''
    try:
        conn = pymssql.connect(server, user, password, database)
        cur = conn.cursor()
        sql = " select dict_keyword,dict_pn from report_pn_dictionary "
        cur.execute(sql)
        rows = cur.fetchall()
        if rows:
            report_pn_dictionary_list = [list(row) for row in rows]
            pn_dict = dict(report_pn_dictionary_list)
            return pn_dict
        else:
            return ""
    except pymssql.Error as ex:
        logger.error("dbException:" + str(ex))
        raise ex
    except Exception as ex:
        logger.error("Call method get_report_pn_dictionary_list() error!")
        logger.error("Exception:" + str(ex))
        raise ex
    finally:
        conn.close()

def insert_report_keyword_pn(server, user, password, database, data_list):
    '''
    :param server:服务器名称
    :param user:用户名
    :param password:密码
    :param database:数据库名
    :param data_list:要插入到report_keyword_pn表的list
    '''
    if data_list:
        try:
            conn = pymssql.connect(server, user, password, database,charset='utf8')
            cur = conn.cursor()
            for one_row in data_list:
                report_year = one_row[0]
                report_week = one_row[1]
                employee_code = one_row[2]
                keyword = one_row[3]
                keyword = "'"+keyword+"'"
                pn = one_row[4]
                frequency = one_row[5]
                sql = ' if not exists (select 1 from report_keyword_pn where report_year = %s and report_week = %s and employee_code = %s and keyword = %s ) ' \
                      ' insert into report_keyword_pn (report_year, report_week, employee_code, keyword, pn, keyword_frequency) ' \
                      ' values(%s, %s, %s, %s, %s, %s) ' \
                      %(report_year, report_week, employee_code, keyword, report_year, report_week, employee_code, keyword, pn, frequency)
                cur.execute(sql)
                conn.commit()
        except pymssql.Error as ex:
            logger.error("dbException:" + str(ex))
            raise ex
        except Exception as ex:
            logger.error("Exception:"+str(ex))
            conn.rollback()
            raise ex
        finally:
            conn.close()
    else:
        logger.error("Call method insert_report_keyword_pn() error!The data_list is empty.")
        raise


if __name__=="__main__":
    logger = write_log()  # 获取日志对象
    time_start = datetime.datetime.now()
    start = time.clock()
    logger.info("Program start,now time is:" + str(time_start))
    server, user, password, database = read_dateConfig_file_set_database()  # 读取配置文件中的数据库信息
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # 系统当前日期
    current_year, current_week = get_year_week_from_Mst_date(server, user, password, database,current_date)  # 从Mst_date获取当前年和周
    report_year = str(current_year)  # 当前系统年
    report_week = str(current_week)  # 当前系统周
    read_dateConfig_file_set_year_week()  # 读配置文件设置report_year和report_week
    logger.info("report_year:" + report_year)
    logger.info("report_week:" + report_week)
    report_employee_list = get_report_employee_list(server, user, password, database, report_year,report_week)  # 从report表获取X社员、X年、X周、top报告列表
    pn_dict = get_report_pn_dictionary_list(server, user, password, database)#从数据库取出字典
    report_keyword_pn_list = generate_report_keyword_pn_list(report_employee_list) #生成要插入到report_keyword_pn表的数据
    insert_report_keyword_pn(server, user, password, database, report_keyword_pn_list)#插入到表report_keyword_pn
    time_end = datetime.datetime.now()
    end = time.clock()
    logger.info("Program end,now time is:" + str(time_end))
    logger.info("Program run : %f seconds" % (end - start))
