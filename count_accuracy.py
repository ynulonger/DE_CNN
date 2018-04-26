# -*- coding: utf-8 -*-
import xlrd
import xlwt
import numpy as np

arousal_or_valence = "valence"
output_file = "total.xlsx"
dir_path = "/home/yyl/DE_CNN/result/"
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
		accuracy = (accuracy/10)*100
		total_accuracy += accuracy
		print(sub,":",accuracy)
		out_sheet.write(sub + 2,column_index, subject)
		out_sheet.write(sub + 2,column_index+1,accuracy)
	mean_accuracy = total_accuracy/32
	std = np.std(accuracy,ddof=1)
	print("mean accuracy:",mean_accuracy)
	print("std:",std)
	out_sheet.write(sub+3,column_index,"mean:")
	out_sheet.write(sub+3,column_index+1,mean_accuracy)
	out_sheet.write(sub+4,column_index,"std:")
	out_sheet.write(sub+4,column_index+1,std)
fill_cells(dir_path+"theta/",0,"CNN_theta","valence")
fill_cells(dir_path+"theta/",2,"CNN_theta","arousal")

fill_cells(dir_path+"alpha/",5,"CNN_alpha","valence")
fill_cells(dir_path+"alpha/",7,"CNN_alpha","arousal")

fill_cells(dir_path+"beta/",10,"CNN_beta","valence")
fill_cells(dir_path+"beta/",12,"CNN_beta","arousal")

fill_cells(dir_path+"gmma/",15,"CNN_gmma","valence")
fill_cells(dir_path+"gmma/",17,"CNN_gmma","arousal")

fill_cells(dir_path+"12/",20,"theta+alpha","valence")
fill_cells(dir_path+"12/",22,"theta+alpha","arousal")

fill_cells(dir_path+"13/",25,"theta+beta","valence")
fill_cells(dir_path+"13/",27,"theta+beta","arousal")

fill_cells(dir_path+"14/",30,"theta+gmma","valence")
fill_cells(dir_path+"14/",32,"theta+gmma","arousal")

fill_cells(dir_path+"23/",35,"alpha+beta","valence")
fill_cells(dir_path+"23/",37,"alpha+beta","arousal")

fill_cells(dir_path+"24/",40,"alpha+gmma","valence")
fill_cells(dir_path+"24/",42,"alpha+gmma","arousal")

fill_cells(dir_path+"34/",45,"beta+gmma","valence")
fill_cells(dir_path+"34/",47,"beta+gmma","arousal")

fill_cells(dir_path+"123/",50,"θ+α+β","valence")
fill_cells(dir_path+"123/",52,"θ+α+β","arousal")

fill_cells(dir_path+"124/",55,"θ+α+γ","valence")
fill_cells(dir_path+"124/",57,"θ+α+γ","arousal")

fill_cells(dir_path+"134/",60,"θ+β+γ","valence")
fill_cells(dir_path+"134/",62,"θ+β+γ","arousal")

fill_cells(dir_path+"234/",65,"α+β+γ","valence")
fill_cells(dir_path+"234/",67,"α+β+γ","arousal")

fill_cells(dir_path+"4_band/",70,"θ+α+β+γ","valence")
fill_cells(dir_path+"4_band/",72,"θ+α+β+γ","arousal")

out_book.save("accuracies_all.xls")