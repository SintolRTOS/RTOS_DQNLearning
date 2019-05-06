# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:26:17 2019

@author: wangjingyi
"""
import numpy as np
from openpyxl import load_workbook
import logging  # 引入logging模块
import time
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
logger.setLevel(level = logging.DEBUG)
handler = logging.FileHandler("KeyWords/log/log_" + str(time.time()) + '.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

class WordAgent(object):
    """docstring for ClassName"""
    def __init__(self,
                 filepath,
                 mode):
        super(WordAgent,self).__init__()
        self.filepath = filepath
        self.mode = mode
        self.parameter_size = 6
        self.result = []
        self.result_keywords = []
    
    def openExcel(self):
        logging.info('start openpyxl openExcel!')
        self.wb = load_workbook(self.filepath)
        print('form sheetname: ' + self.wb.sheetnames[0])
        self.keytitle = self.wb.sheetnames[0];
        sheet = self.wb.get_sheet_by_name(str(self.keytitle))
        self.max_row = sheet.max_row
        self.max_column = sheet.max_column
        logging.info("form max_row: " + str(self.max_row))
        logging.info("form max_column: " + str(self.max_column))
        self.data = []
        #read the excel file
#        print(sheet.cell(2,2).value)
#        for i in range(1,self.max_row):
#            for j in range(1,self.max_column):
#                print(sheet.cell(i,j).value)
#                self.data[i,j] = sheet.cell(i,j).value
        for row in sheet.iter_rows(min_col=1, min_row=1, max_row=self.max_row, max_col=self.max_column):
            ROW = []
            for cell in row:
                ROW.append(cell.value)
            self.data.append(ROW)
        logging.info("all form data: ")
        logging.info(self.data)
    def reset(self):
#        self.data.clear()
#        print('wordAgent clear!')
#        self.openExcel()
        self.init_observation()
        self.init_action_space()
        self.reward = 0
        self.result.clear()
        self.result_keywords.clear()
        return self.observation
        
    def init_observation(self):
        logging.info('------------------ init_observation-------------------------')
        self.keywords_length = 5
#        logger.info('self.keywords_length: '+ str(self.keywords_length))
        self.observation = np.zeros(self.keywords_length * self.parameter_size)
        for i in range(self.keywords_length):
            self.observation[i*self.parameter_size] = float(self.data[i][4])
            self.observation[i*self.parameter_size + 1] = float(self.data[i][5])
            self.observation[i*self.parameter_size + 2] = float(self.data[i][8])
            self.observation[i*self.parameter_size + 3] = float(self.data[i][9])
            self.observation[i*self.parameter_size + 4] = float(self.data[i][10])
            self.observation[i*self.parameter_size + 5] = i
#            logger.info('----------------self.observation i: ' + str(i))
#            logger.info('self.observation[i*self.parameter_size]: ' + str(self.observation[i*self.parameter_size]))
#            logger.info('self.observation[i*self.parameter_size + 1]: ' + str(self.observation[i*self.parameter_size + 1]))
#            logger.info('self.observation[i*self.parameter_size + 2]: ' + str(self.observation[i*self.parameter_size + 2]))
#            logger.info('self.observation[i*self.parameter_size + 3]: ' + str(self.observation[i*self.parameter_size + 3]))
#            logger.info('self.observation[i*self.parameter_size + 4]: ' + str(self.observation[i*self.parameter_size + 4]))
#            logger.info('self.observation[i*self.parameter_size + 5]: ' + str(self.observation[i*self.parameter_size + 5]))
#            logger.info('----------------end---------------- ')
#            ROW = []
#            ROW.append(self.data[i][5])
#            ROW.append(self.data[i][6])
#            ROW.append(self.data[i][9])
#            ROW.append(self.data[i][10])
#            ROW.append(self.data[i][11])
#            self.observation.append(ROW)
        return self.observation
    
    def get_observation(self):
        return self.observation
    
    def init_action_space(self):
        self.action_space = [2.,]
    
    def get_action_space(self):
        return self.action_space
    
    def get_result_keywords(self):
        logger.info('--------------get_result_keywords----------------')
        self.result_keywords.clear()
        count = len(self.result)
        logger.info('self.result: ' + str(self.result))
        logger.info('result count: ' + str(count))
        for i in range(count):
            key_id = int(self.result[i])
            key_words = self.data[key_id][1]
            self.result_keywords.append(key_words)
            logger.info('key_id: ' + str(key_id))
            logger.info('keywords: ' + str(key_words))
        
        logger.info('result keywords: ' + str(self.result_keywords))
        logger.info('--------------end get_result_keywords----------------')
        return self.result_keywords
    
    def step(self,u):
        logger.info('WordAgent select: ' + str(u))
        a_value =  float(u[0])
        pervalue = (a_value + 2.) / 4.
        logger.info('WordAgent pervalue: ' + str(pervalue))
        index = int(pervalue * float(self.keywords_length - 1))
        logger.info('--------------step----------------')
        logger.info('WordAgent select index: ' + str(index))
        popularity = self.observation[index*self.parameter_size]
        conversion = self.observation[index*self.parameter_size + 1]
        transform_1 = self.observation[index*self.parameter_size + 2]
        transform_2 = self.observation[index*self.parameter_size + 3]
        transform_3 = self.observation[index*self.parameter_size + 4]
        keywords_id = self.observation[index*self.parameter_size + 5]
        self.reward += popularity*conversion + popularity*transform_1*2 + popularity*transform_2 + popularity*transform_3
        logger.info('WordAgent self.reward: ' + str(self.reward))
        self.observation[index*self.parameter_size] = -1000
        self.observation[index*self.parameter_size + 1] = -1
        self.observation[index*self.parameter_size + 2] = -1
        self.observation[index*self.parameter_size + 3] = -1
        self.observation[index*self.parameter_size + 4] = -1
        self.result.append(int(keywords_id))
#        self.observation[index*self.parameter_size + 5] = 0
        logger.info('step self.observation index: ' + str(index))
        logger.info('step self.observation[index*self.parameter_size]: ' + str(self.observation[index*self.parameter_size]))
        logger.info('step self.observation[index*self.parameter_size + 1]: ' + str(self.observation[index*self.parameter_size + 1]))
        logger.info('step self.observation[index*self.parameter_size + 2]: ' + str(self.observation[index*self.parameter_size + 2]))
        logger.info('step self.observation[index*self.parameter_size + 3]: ' + str(self.observation[index*self.parameter_size + 3]))
        logger.info('step self.observation[index*self.parameter_size + 4]: ' + str(self.observation[index*self.parameter_size + 4]))
        logger.info('step self.observation[index*self.parameter_size + 5]: ' + str(self.observation[index*self.parameter_size + 5]))
        logger.info('step self.observation result: ' + str(self.result))
        logger.info('----------------step end---------------- ')
#        self.observation = np.delete(self.observation,
#                  (index*self.parameter_size,
#                   index*self.parameter_size+1,
#                   index*self.parameter_size+2,
#                   index*self.parameter_size+3,
#                   index*self.parameter_size+4,
#                   index*self.parameter_size+5),
#                   axis = 0)
#        print('self.observation.shape: ',self.observation.shape)
#        self.keywords_length -= 1
        return self.get_observation(),self.reward,False,{}
        
            
        
        

print('test openpyxl sucessful!')
agent = WordAgent('assert/keyword.xlsx','xlsx')
agent.openExcel()
agent.reset()
        
#info = np.array([1,2,3,4,5,6,7,8,9,0])
#print(info.shape)
#observation =  np.delete(info,
#                  (3,4,5),
#                   axis = 0)
#print(observation.shape)

    
        