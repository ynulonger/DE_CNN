# -*- coding: utf-8 -*-
import xlrd
import xlwt
import numpy as np

arousal_or_valence = "valence"
output_file = "total.xlsx"
dir_path = "/home/yyl/DE_CNN/result/with_base/1234/LOSO/32_folds/"
model_name = "CNN"
# 10 folds cross validation
fold = 32

out_book = xlwt.Workbook(encoding='utf-8', style_compression=0)
out_sheet = out_book.add_sheet('accuracy', cell_overwrite_ok=True)
column_index = 0

def fill_cells(dir_path,filename,column_index,model_name,target_class,fold):
	# table header
	out_sheet.write(0,column_index,"model_name:")
	out_sheet.write(0,column_index+1,model_name)

	out_sheet.write(1,column_index,"target_class:")
	out_sheet.write(1,column_index+1,target_class)

	out_sheet.write(2,column_index,"subject")
	out_sheet.write(2,column_index+1,"accuracy")
	total_accuracy = 0

	subject = filename
	accuracy = 0
	for count in range(fold):
		input_file = dir_path+target_class+"/"+str(subject)+"_"+str(count)+".xlsx"
		in_book = xlrd.open_workbook(input_file)
		sheet = in_book.sheet_by_name("condition")
		accuracy += sheet.cell_value(1,0)
	accuracy = (accuracy/fold)*100
	total_accuracy += accuracy
	print(subject,":",accuracy)

# fill_cells(dir_path,0,"3D_CO","valence")
fill_cells(dir_path,"CO_3D",2,"3D_CO","arousal",32)
fill_cells(dir_path,"CO_3D",2,"3D_CO","valence",32)

fill_cells("/home/yyl/DE_CNN/result/with_base/1234/LOSO/8_folds/","CO_3D",2,"3D_CO","arousal",32)
fill_cells("/home/yyl/DE_CNN/result/with_base/1234/LOSO/8_folds/","CO_3D",2,"3D_CO","valence",32)

fill_cells("/home/yyl/DE_CNN/result/with_base/1234/LOSO/","LOSO",2,"3D_CO","arousal",10)
fill_cells("/home/yyl/DE_CNN/result/with_base/1234/LOSO/","LOSO",2,"3D_CO","valence",10)


# out_book.save("accuracies_LVO.xls")
print("end")