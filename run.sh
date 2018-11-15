echo "start sub：	$1";
echo "end sub：	$2";
echo "class:	$3";
echo "use baseline:	$4";
for((i=$1;i<=$2;i++));
do
	if [ $i -lt 10 ];
	then
		a=s0
	else
		a=s
	fi
	b=$i 
	sub=${a}${b}
	echo $sub
	python cnn.py $4 "$sub" $3 1
	python cnn.py $4 "$sub" $3 2
	python cnn.py $4 "$sub" $3 3
	python cnn.py $4 "$sub" $3 4
	python cnn.py $4 "$sub" $3 1 2
	python cnn.py $4 "$sub" $3 1 3
	python cnn.py $4 "$sub" $3 1 4
	python cnn.py $4 "$sub" $3 2 3
	python cnn.py $4 "$sub" $3 2 4
	python cnn.py $4 "$sub" $3 3 4
	python cnn.py $4 "$sub" $3 1 2 3
	python cnn.py $4 "$sub" $3 1 2 4
	python cnn.py $4 "$sub" $3 1 3 4
	python cnn.py $4 "$sub" $3 2 3 4
	python cnn.py $4 "$sub" $3 1 2 3 4
done 