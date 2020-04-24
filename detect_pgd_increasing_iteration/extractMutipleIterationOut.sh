 #!/bin/bash
        for i in `seq 1 2 23`;
        do
                echo iteration: $i
		            #python getLayerOutput.py --model=MNIST --it=$i --resume=./checkpoint/MnistNet_50.pth
		            python detection.py --model=MNIST --it=$i
        done  
