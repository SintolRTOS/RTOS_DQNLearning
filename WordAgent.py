# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:26:17 2019

@author: wangjingyi
"""
from openpyxl import load_workbook

class WordAgent(object):
    """docstring for ClassName"""
    def __init__(self,
                 filepath,
                 mode):
        super(WordAgent,self).__init__()
        self.filepath = filepath
        self.mode = mode
    
    def openExcel(self):
        print('test openpyxl openExcel!')
        self.wb = load_workbook(self.filepath)
        print('form sheetname: ' + self.wb.sheetnames[0])
        self.keytitle = self.wb.sheetnames[0];
        sheet = self.wb.get_sheet_by_name(str(self.keytitle))
        self.max_row = sheet.max_row
        self.max_column = sheet.max_column
        print("form max_row: " + str(self.max_row))
        print("form max_column: " + str(self.max_column))
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
        print("all form data: ", end="")
        print(self.data)
        

print('test openpyxl sucessful!')
agent = WordAgent('assert/keyword.xlsx','xlsx')
agent.openExcel();

    
        