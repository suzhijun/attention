
function readfile()
{
	for file in `ls $1` 
	do
		if [ -d $1"/"$file ]
		then 
			echo $file is directory
			readfile $1"/"$file
		else
			if [[ $file != *.py ]]
			then
				echo convert $1"/"$file to unix style
				sed -i 's/\r//g' $1"/"$file
			else
				echo $file is python file
			fi
		fi
	done
}

readfile $1
