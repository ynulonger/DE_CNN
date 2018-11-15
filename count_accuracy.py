# -*- coding: utf-8 -*-
import xlrd
import xlwt
import numpy as np
import sys

arousal_or_valence = sys.argv[1]
with_or_without = sys.argv[2]

dir_path = "/home/yyl/DE_CNN/result/"+with_or_without+"_base/"
model_name = "CNN"
# 10 folds cross validation
fold = 10

out_book = xlwt.Workbook(encoding='utf-8', style_compression=0)
out_sheet = out_book.add_sheet('accuracy', cell_overwrite_ok=True)
column_index = 0

def fill_cells(dir_path,column_index,model_name,target_class):
	# table header
	out_sheet.write(0,column_index,"model_name:")
	out_sheet.write(0,column_index+1,model_name)

	out_sheet.write(1,column_index,"target_class:")
	out_sheet.write(1,column_index+1,target_class)

	out_sheet.write(2,column_index,"subject")
	out_sheet.write(2,column_index+1,"accuracy")
	total_accuracy = 0
	for sub in range(1,33):
		subject = "s%02d"%sub
		accuracy = 0
		for count in range(fold):
			input_file = dir_path+target_class+"/"+str(subject)+"_"+str(count)+".xlsx"
			in_book = xlrd.open_workbook(input_file)
			sheet = in_book.sheet_by_name("condition")
			accuracy += sheet.cell_value(1,0)
		accuracy = (accuracy/fold)*100
		total_accuracy += accuracy
		print(sub,":",accuracy)
		out_sheet.write(sub + 2,column_index, subject)
		out_sheet.write(sub + 2,column_index+1,accuracy)
	mean_accuracy = total_accuracy/32
	# std = np.std(accuracy,ddof=1)
	print("mean accuracy:",mean_accuracy)
	out_sheet.write(sub+3,column_index,"mean:")
	out_sheet.write(sub+3,column_index+1,mean_accuracy)
	# out_sheet.write(sub+4,column_index,"std:")
	# out_sheet.write(sub+4,column_index+1,std)

fill_cells(dir_path+"1/",0,"θ",arousal_or_valence)
fill_cells(dir_path+"2/",3,"α",arousal_or_valence)
fill_cells(dir_path+"3/",6,"β",arousal_or_valence)
fill_cells(dir_path+"4/",9,"γ",arousal_or_valence)
fill_cells(dir_path+"12/",12,"θ+α",arousal_or_valence)
fill_cells(dir_path+"13/",15,"θ+β",arousal_or_valence)
fill_cells(dir_path+"14/",18,"θ+γ",arousal_or_valence)
fill_cells(dir_path+"23/",21,"α+β",arousal_or_valence)
fill_cells(dir_path+"24/",24,"α+γ",arousal_or_valence)
fill_cells(dir_path+"34/",27,"β+γ",arousal_or_valence)
fill_cells(dir_path+"123/",30,"θ+α+β",arousal_or_valence)
fill_cells(dir_path+"124/",33,"θ+α+γ",arousal_or_valence)
fill_cells(dir_path+"134/",36,"θ+β+γ",arousal_or_valence)
fill_cells(dir_path+"234/",39,"α+β+γ",arousal_or_valence)
fill_cells(dir_path+"1234/",42,"θ+α+β+γ",arousal_or_valence)

out_book.save("./result/summary/acc_"+with_or_without+"_"+arousal_or_valence+".xls")
print("end")