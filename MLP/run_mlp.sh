echo "start sub：	$1";
echo "end sub：	$2";
echo "class:	$3";
echo "use baseline:	$4";
for((i=$1;i<=$2;i++));
do
	if [ $i -lt 10 ];
	then
		a=0
	fi
	b=$i 
	sub=${a}${b}
	echo $sub
	python3 ../tf_mlp_channels.py $sub $3 $4 1
	python3 ../tf_mlp_channels.py $sub $3 $4 2
	python3 ../tf_mlp_channels.py $sub $3 $4 3
	python3 ../tf_mlp_channels.py $sub $3 $4 4
	python3 ../tf_mlp_channels.py $sub $3 $4 1 2
	python3 ../tf_mlp_channels.py $sub $3 $4 1 3
	python3 ../tf_mlp_channels.py $sub $3 $4 1 4
	python3 ../tf_mlp_channels.py $sub $3 $4 2 3
	python3 ../tf_mlp_channels.py $sub $3 $4 2 4
	python3 ../tf_mlp_channels.py $sub $3 $4 3 4
	python3 ../tf_mlp_channels.py $sub $3 $4 1 2 3
	python3 ../tf_mlp_channels.py $sub $3 $4 1 2 4
	python3 ../tf_mlp_channels.py $sub $3 $4 1 3 4
	python3 ../tf_mlp_channels.py $sub $3 $4 2 3 4
	python3 ../tf_mlp_channels.py $sub $3 $4 1 2 3 4
	python3 ../tf_mlp_channels.py $sub $3 $4 1 2 3 4

done 

